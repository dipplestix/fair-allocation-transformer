import torch


def one_hot_allocation(alloc: torch.Tensor) -> torch.Tensor:
    """Convert soft allocation [B,n,m] to discrete one-hot per item."""
    winners = alloc.argmax(dim=1)  # [B, m]
    B, n, m = alloc.shape
    discrete = torch.zeros_like(alloc)
    batch_idx = torch.arange(B).unsqueeze(1).expand(B, m)
    item_idx = torch.arange(m).expand(B, m)
    discrete[batch_idx, winners, item_idx] = 1
    return discrete


def ef1_fraction(alloc: torch.Tensor, valuations: torch.Tensor) -> float:
    """Return fraction of EF1 allocations in batch."""
    A = one_hot_allocation(alloc)
    B, n, m = A.shape
    count = 0
    for b in range(B):
        V = valuations[b]
        Ab = A[b]
        if _is_ef1(V, Ab):
            count += 1
    return count / B if B > 0 else 0.0


def _is_ef1(V: torch.Tensor, A: torch.Tensor) -> bool:
    n, m = V.shape
    for i in range(n):
        val_i_Ai = (V[i] * A[i]).sum()
        for j in range(n):
            if i == j:
                continue
            Aj_vals_i = V[i] * A[j]
            val_i_Aj = Aj_vals_i.sum()
            max_item = (
                Aj_vals_i.max()
                if Aj_vals_i.numel() > 0
                else torch.tensor(0.0, device=V.device)
            )
            if val_i_Ai < val_i_Aj - max_item:
                return False
    return True


def utilitarian_welfare(
    allocation: torch.Tensor, valuations: torch.Tensor
) -> torch.Tensor:
    """Utilitarian welfare of a batch of allocations."""
    return (allocation * valuations).sum(dim=(1, 2))


def max_welfare(valuations: torch.Tensor) -> torch.Tensor:
    """Welfare of the optimal discrete allocation via integer programming."""
    try:
        import pulp
    except Exception as e:  # pragma: no cover - fallback for missing pulp
        best_per_item = valuations.max(dim=1).values
        return best_per_item.sum(dim=1)

    B, n, m = valuations.shape
    device = valuations.device
    welfare = torch.zeros(B, device=device)

    for b in range(B):
        V = valuations[b].cpu().tolist()
        prob = pulp.LpProblem("max_welfare", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", (range(n), range(m)), cat=pulp.LpBinary)

        # Objective: maximise total welfare
        prob += pulp.lpSum(
            V[i][j] * x[i][j] for i in range(n) for j in range(m)
        )

        # Each item assigned to exactly one agent
        for j in range(m):
            prob += pulp.lpSum(x[i][j] for i in range(n)) == 1

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        welfare[b] = sum(
            V[i][j] * x[i][j].value() for i in range(n) for j in range(m)
        )

    return welfare


def welfare_ratio(alloc: torch.Tensor, valuations: torch.Tensor) -> float:
    """Return average welfare divided by max welfare, expressed as a percentage."""
    A = one_hot_allocation(alloc)
    welfare = utilitarian_welfare(A, valuations)
    ratio = welfare / max_welfare(valuations)
    return ratio.mean().item() * 100
