import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

import wandb

from eef1nn_input import encode_valuations
from net import EEF1NN
from loss import eef1nn_lagrangian_loss
from helper import ef1_fraction, welfare_ratio


def random_valuations(num_samples: int, n_agents: int, m_items: int) -> torch.Tensor:
    """Generate uniform random valuations in [0, 1]."""
    return torch.rand(num_samples, n_agents, m_items)


def main():
    parser = argparse.ArgumentParser(description="Train the EEF1-NN model")
    parser.add_argument("--agents", type=int, default=4, help="number of agents")
    parser.add_argument("--items", type=int, default=20, help="number of items")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0, help="Lagrange multiplier")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=10000, help="training samples")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--test-samples", type=int, default=512, help="number of validation samples")
    args = parser.parse_args()

    wandb.init(project="eef1nn", config=vars(args))

    device = torch.device(args.device)

    # Generate dataset
    valuations = random_valuations(args.samples, args.agents, args.items)
    dataset = TensorDataset(valuations)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    test_vals = random_valuations(args.test_samples, args.agents, args.items)

    model = EEF1NN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for (V,) in loader:
            V = V.to(device)
            inp = encode_valuations(V)
            alloc = model(inp, temperature=args.temperature)
            loss = eef1nn_lagrangian_loss(alloc, V, lam=args.lam)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * V.size(0)

        avg_loss = total_loss / args.samples

        model.eval()
        with torch.no_grad():
            V_t = test_vals.to(device)
            inp_t = encode_valuations(V_t)
            alloc_t = model(inp_t, temperature=args.temperature)
            alpha = ef1_fraction(alloc_t, V_t) * 100
            beta = welfare_ratio(alloc_t, V_t)

        wandb.log({"loss": avg_loss, "alpha": alpha, "beta": beta}, step=epoch)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | Alpha: {alpha:.2f}% | Beta: {beta:.2f}%")

    torch.save(model.state_dict(), "eef1nn.pt")


if __name__ == "__main__":
    main()
