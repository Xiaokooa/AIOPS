from __future__ import annotations

import os

import torch


def main() -> None:
    print("[cuda-check]")
    print(f"torch={torch.__version__}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit(1)
    print(f"cuda_device_count={torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(
            f"cuda:{idx} name={props.name} "
            f"total_gb={props.total_memory / (1024 ** 3):.2f} "
            f"capability={props.major}.{props.minor}"
        )
    x = torch.ones(1, device="cuda")
    print(f"allocation_ok device={x.device} value={float(x.item())}")


if __name__ == "__main__":
    main()
