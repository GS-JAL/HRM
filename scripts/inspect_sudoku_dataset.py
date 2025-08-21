#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np


def load_dataset_split(data_path: str, split: str):
    split_dir = os.path.join(data_path, split)
    with open(os.path.join(split_dir, "dataset.json"), "r") as f:
        meta = json.load(f)

    data = {}
    for field in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]:
        data[field] = np.load(os.path.join(split_dir, f"all__{field}.npy"))

    return meta, data


def decode_grid(seq: np.ndarray):
    # seq is length 81, values 1..10 where 1 represents blank (0)
    grid = (seq.reshape(9, 9).astype(int) - 1)
    return grid


def grid_to_str(grid: np.ndarray, show_zero_as: str = ".") -> str:
    lines = []
    for r in range(9):
        row_vals = []
        for c in range(9):
            v = int(grid[r, c])
            row_vals.append(str(v) if v != 0 else show_zero_as)
        line = " ".join(row_vals)
        lines.append(line)
    return "\n".join(lines)


def check_sudoku_validity(grid: np.ndarray):
    # Expect digits 1..9 in solution; 0 indicates blank (should not occur in labels)
    issues = []
    # Rows
    for r in range(9):
        row = grid[r, :]
        vals = [v for v in row.tolist() if v != 0]
        if len(vals) != len(set(vals)):
            issues.append(f"row_dup_{r}")
    # Cols
    for c in range(9):
        col = grid[:, c]
        vals = [v for v in col.tolist() if v != 0]
        if len(vals) != len(set(vals)):
            issues.append(f"col_dup_{c}")
    # 3x3 subgrids
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            block = grid[br:br+3, bc:bc+3].reshape(-1)
            vals = [v for v in block.tolist() if v != 0]
            if len(vals) != len(set(vals)):
                issues.append(f"box_dup_{br}_{bc}")
    return issues


def summarize_example(inputs: np.ndarray, labels: np.ndarray, idx: int):
    inp_seq = inputs[idx]
    out_seq = labels[idx]

    checks = []
    if len(inp_seq) != 81 or len(out_seq) != 81:
        checks.append("seq_len_mismatch")

    # Range checks (expect 1..10)
    if np.any(inp_seq < 0) or np.any(inp_seq > 10):
        checks.append("input_out_of_vocab")
    if np.any(out_seq < 0) or np.any(out_seq > 10):
        checks.append("label_out_of_vocab")

    inp_grid = decode_grid(inp_seq)
    out_grid = decode_grid(out_seq)

    # Clue preservation: where input has non-blank (value>0), label must match
    for r in range(9):
        for c in range(9):
            if inp_grid[r, c] != 0 and out_grid[r, c] != inp_grid[r, c]:
                checks.append("clue_mismatch")
                break
        else:
            continue
        break

    # Label validity checks
    if np.any((out_grid < 0) | (out_grid > 9)):
        checks.append("label_digit_out_of_range")
    else:
        checks.extend(check_sudoku_validity(out_grid))

    return {
        "decoded": {
            "input_grid": inp_grid.tolist(),
            "label_grid": out_grid.tolist(),
        },
        "pretty": {
            "input_grid": grid_to_str(inp_grid),
            "label_grid": grid_to_str(out_grid),
        },
        "raw": {
            "inputs": inp_seq.tolist(),
            "labels": out_seq.tolist(),
        },
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect Sudoku HRM dataset")
    parser.add_argument("--data-path", type=str, default="data/sudoku-extreme-full", help="Path to dataset root (containing train/test)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Split to inspect")
    parser.add_argument("--num", type=int, default=3, help="Number of examples to show")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Explicit indices to inspect")

    args = parser.parse_args()

    meta, data = load_dataset_split(args.data_path, args.split)
    inputs = data["inputs"]
    labels = data["labels"]

    print(f"Dataset: {args.data_path} | Split: {args.split}")
    print(f"Examples: {len(inputs)} | seq_len={meta['seq_len']} | vocab={meta['vocab_size']}")
    print()

    if args.indices is not None and len(args.indices) > 0:
        indices = args.indices
    else:
        end = min(args.start + args.num, len(inputs))
        indices = list(range(args.start, end))

    for i in indices:
        summary = summarize_example(inputs, labels, i)
        print(f"=== Example #{i} ===")
        print("Input grid:\n" + summary["pretty"]["input_grid"]) 
        print("Label grid:\n" + summary["pretty"]["label_grid"]) 
        if summary["checks"]:
            print(f"Checks: {summary['checks']}")
        print("Raw inputs:", summary["raw"]["inputs"]) 
        print("Raw labels:", summary["raw"]["labels"]) 
        print()


if __name__ == "__main__":
    main()


