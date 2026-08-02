#!/usr/bin/env python3
"""Run one bounded optimizer step on exactly one visible CUDA GPU."""

from __future__ import annotations

import json

import torch


def main() -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU must be visible")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    model = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = torch.randn(8, 4, device=device)
    targets = torch.randn(8, 2, device=device)
    weight_before = model.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(features), targets)
    if not torch.isfinite(loss):
        raise RuntimeError("optimizer smoke produced a non-finite loss")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    if torch.equal(weight_before, model.weight.detach()):
        raise RuntimeError("optimizer step did not update model weights")
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "loss": float(loss.detach().cpu()),
                "status": "GPU_SENTINEL:optimizer-step-complete",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
