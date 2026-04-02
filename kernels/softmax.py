import torch

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Pure Python/PyTorch softmax
    """
    # Step 1: subtract max for numerical stability
    # Without this, exp() of large numbers overflows to inf
    x_max = x.max(dim=-1, keepdim=True).values

    # Step 2: exponentiate
    numerator = torch.exp(x - x_max)

    # Step 3: normalize
    denominator = numerator.sum(dim=-1, keepdim=True)

    return numerator / denominator


if __name__ == "__main__":
    # Test it on CPU (your Mac can run this)
    x = torch.randn(4, 8)
    print("Input:\n", x)

    our_result = naive_softmax(x)
    torch_result = torch.softmax(x, dim=-1)

    print("\nOur softmax:\n", our_result)
    print("\nMatch:", torch.allclose(our_result, torch_result))
    # Should print True