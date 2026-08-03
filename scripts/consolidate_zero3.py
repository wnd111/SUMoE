from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sumoe.training.checkpoint import (
    validate_architecture_metadata,
    write_architecture_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate the last ZeRO-3 checkpoint to one FP32 state dict"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pointer = args.run_dir / "last-checkpoint.txt"
    if not pointer.is_file():
        raise FileNotFoundError(f"last checkpoint pointer is missing: {pointer}")
    checkpoint = args.run_dir / pointer.read_text(encoding="utf-8").strip()
    metadata = validate_architecture_metadata(checkpoint)
    output = args.output_dir / "pytorch_model.bin"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite consolidated model: {output}")
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    state = get_fp32_state_dict_from_zero_checkpoint(str(checkpoint))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    write_architecture_metadata(args.output_dir, metadata["variant"])
    print(f"saved consolidated state dict to {output}")


if __name__ == "__main__":
    main()
