import torch


def encode_valuations(V: torch.Tensor) -> torch.Tensor:
    """
    Construct the 6‑channel EEF1‑NN input from an (additive) valuation tensor.

    Parameters
    ----------
    V : torch.Tensor                    # shape = [B, n_agents, m_items]
        Valuation matrix for each batch element.

    Returns
    -------
    I : torch.Tensor                    # shape = [B, 6, n_agents, m_items]
        Six‑channel input:
            ch‑0 : full valuation matrix  (V)
            ch‑1 : X restricted to items  j≡0 (mod 5)
            ch‑2 : X restricted to items  j≡0,1 (mod 5)
            ch‑3 : X restricted to items  j≡0,1,2 (mod 5)
            ch‑4 : X restricted to items  j≡0,1,2,3 (mod 5)
            ch‑5 : X  (all items)                      (= channel‑4 super‑set)
        where  X[i,j] = V[i,j]  if  i = argmax_k V[k,j]  else 0 .
    """
    if V.dim() != 3:
        raise ValueError("V must have shape [B, n_agents, m_items]")
    B, n, m = V.shape
    device = V.device

    # ---------- 1) Build X (“winner‑take‑all” valuation matrix) ----------
    winners = V.argmax(dim=1)                             # [B, m]
    # create a boolean mask: True at (batch, winning_agent, item)
    mask = torch.zeros_like(V, dtype=torch.bool)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, m)
    item_idx  = torch.arange(m, device=device).expand(B, m)
    mask[batch_idx, winners, item_idx] = True
    X = V * mask                                          # [B, n, m]

    # ---------- 2) Pre‑compute item‑index remainders mod 5 ----------
    mod5 = torch.arange(m, device=device) % 5             # [m]

    # ---------- 3) Assemble the 5 incremental X‑channels ----------
    chans = [V]                                           # channel‑0
    for k in range(5):                                    # k = 0 … 4
        keep = (mod5 <= k).view(1, 1, m)                  # broadcast mask
        chans.append(X * keep)                            # channel‑(k+1)

    # Stack along “channel” dimension → [B, 6, n, m]
    I = torch.stack(chans, dim=1)
    return I
