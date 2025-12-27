import torch


def get_nash_welfare(u: torch.Tensor, allocation: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    u:          (B, n, m)
    allocation: (B, m, n)
    reduction:  "mean" | "sum" | "none"
    
    Returns: scalar if reduction != "none"; else (B,)
    """

    # (B, n, n)
    mat = torch.matmul(u, allocation)

    # (B, n)
    diag = torch.diagonal(mat, dim1=-2, dim2=-1)

    # Geometric mean = (prod diag)^(1/n)
    # Use logs for numerical stability
    gmean = diag.clamp(min=1e-9).log().mean(dim=-1).exp()  # (B,)

    if reduction == "mean":
        return gmean.mean()
    elif reduction == "sum":
        return gmean.sum()
    else:
        return gmean
