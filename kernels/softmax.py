import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────
# BASELINE: Plain PyTorch (3 passes over HBM)
# ─────────────────────────────────────────────

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Readable baseline. Correct but slow — touches HBM 3 times:
      Pass 1: read x → compute max
      Pass 2: read x → compute exp(x - max) and sum
      Pass 3: read x → divide
    """
    x_max = x.max(dim=-1, keepdim=True).values
    numerator = torch.exp(x - x_max)
    denominator = numerator.sum(dim=-1, keepdim=True)
    return numerator / denominator


# ─────────────────────────────────────────────
# TRITON KERNEL: Fused (1 pass over HBM)
# ─────────────────────────────────────────────

@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One Triton 'program' handles one full row.
    All computation (max, exp, sum, divide) happens in
    registers — we only touch HBM once to read, once to write.
    """
    row_idx = tl.program_id(0)

    # --- Load entire row into registers (the ONE HBM read) ---
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_ptr = input_ptr + row_idx * n_cols
    mask = col_offsets < n_cols

    x = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf'))

    # --- All math below happens in fast registers ---

    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # Exponentiate
    x = tl.exp(x)

    # Normalize
    x_sum = tl.sum(x, axis=0)
    x = x / x_sum

    # --- Write result back (the ONE HBM write) ---
    out_ptr = output_ptr + row_idx * n_cols
    tl.store(out_ptr + col_offsets, x, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)

    # Launch one program per row
    _softmax_kernel[(n_rows,)](
        x, output, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ─────────────────────────────────────────────
# CORRECTNESS CHECK
# ─────────────────────────────────────────────

def check_correctness():
    print("=" * 50)
    print("CORRECTNESS CHECK")
    print("=" * 50)

    torch.manual_seed(0)
    x = torch.randn(4, 8, device='cuda')

    ref = torch.softmax(x, dim=-1)
    ours = triton_softmax(x)

    match = torch.allclose(ref, ours, atol=1e-5)
    print(f"Triton vs PyTorch match: {match}")

    if not match:
        print("Max difference:", (ref - ours).abs().max().item())


# ─────────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────────

def benchmark():
    print("\n" + "=" * 50)
    print("BENCHMARK (GB/s = higher is better)")
    print("=" * 50)
    print(f"{'N cols':<10} {'Torch':>12} {'Triton':>12} {'Speedup':>10}")
    print("-" * 50)

    for n_cols in [128, 256, 512, 1024, 2048, 4096]:
        x = torch.randn(1024, n_cols, device='cuda', dtype=torch.float32)

        # Bytes read + written (2 passes: 1 read, 1 write)
        bytes_accessed = 2 * x.nelement() * x.element_size()

        torch_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, dim=-1)
        )
        triton_ms = triton.testing.do_bench(
            lambda: triton_softmax(x)
        )

        def to_gbs(ms):
            return bytes_accessed / (ms * 1e-3) / 1e9

        torch_gbs  = to_gbs(torch_ms)
        triton_gbs = to_gbs(triton_ms)
        speedup    = triton_gbs / torch_gbs

        print(f"{n_cols:<10} {torch_gbs:>10.1f}x  {triton_gbs:>10.1f}x  {speedup:>8.2f}x")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    check_correctness()
    benchmark()