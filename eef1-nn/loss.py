# ------------------------------------------------------------
# EEF1‑NN  ·  Lagrangian loss   (Mishra et al., 2021 §4.1)
# ------------------------------------------------------------
import torch
import torch.nn.functional as F


def util_matrix(V: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Compute   U[b,i,k] = v_i( A_k )   for every batch b.
      V : [B, n, m]   (valuations)
      A : [B, n, m]   (allocations, soft)
    Returns
      U : [B, n, n]
    """
    # A_k is "column k" in the allocation matrix   [n, m]
    #   v_i(A_k) = Σ_j  V[i,j] · A[k,j]
    #   => batch‑matmul  V  @  Aᵀ
    return torch.bmm(V, A.transpose(1, 2))          # [B, n, n]


def envy_penalty(V: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Compute Σ_i envy_i  (Eq. 1)  for each batch element.
    Returns tensor shape [B]  (one scalar per batch).
    """
    U = util_matrix(V, A)                           # [B, n, n]
    self_util = torch.diagonal(U, dim1=1, dim2=2)   # [B, n]
    diff = U - self_util.unsqueeze(2)               # broadcast subtract
    envy = F.relu(diff).sum(dim=2)                  # Σ_k max(0,⋯)  → [B,n]
    return envy.sum(dim=1)                          # Σ_i …         → [B]


def usw(V: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Utilitarian social welfare per batch:  Σ_i v_i(A_i).
    Returns shape [B].
    """
    U = util_matrix(V, A)                           # [B, n, n]
    self_util = torch.diagonal(U, dim1=1, dim2=2)   # [B, n]
    return self_util.sum(dim=1)                     # Σ_i … → [B]


def eef1nn_lagrangian_loss(
        alloc: torch.Tensor,          # A  [B, n, m]   (output of model)
        valuations: torch.Tensor,     # V  [B, n, m]   (ground‑truth)
        lam: float = 1.0              # λ (Lagrange multiplier)
    ) -> torch.Tensor:
    """
    Lagrangian loss 𝓛  (batch mean; differentiable).

    𝓛 = ( −USW  +  λ · envy_penalty / n ) / (n·m)

    scaling by n·m is optional but matches Eq. (4).
    """
    if alloc.dim() != 3 or valuations.shape != alloc.shape:
        raise ValueError("alloc and valuations must both be [B,n,m]")

    B, n, m = alloc.shape

    # ----- utility & envy -----
    usw_val  = usw(valuations, alloc)                  # [B]
    envy_val = envy_penalty(valuations, alloc)         # [B]

    # ----- loss per batch -----
    loss_per_sample = (
        -usw_val + lam * (envy_val / n)
    ) / (n * m)

    # return mean over batch
    return loss_per_sample.mean()
