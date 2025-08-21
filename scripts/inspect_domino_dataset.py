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


def decode_input_sequence(input_seq: np.ndarray):
    # Table tiles from indices 0..55 (pairs)
    table_tiles = []
    for i in range(0, 56, 2):
        a, b = int(input_seq[i]), int(input_seq[i + 1])
        if a == 0 and b == 0:
            continue
        if a == 0 or b == 0:
            # malformed pair, still try to decode
            tile = (a - 1 if a > 0 else None, b - 1 if b > 0 else None)
        else:
            tile = (a - 1, b - 1)
        table_tiles.append(tile)

    # Player tiles from indices 56..69 (pairs)
    player_tiles = []
    for i in range(56, 70, 2):
        a, b = int(input_seq[i]), int(input_seq[i + 1])
        if a == 0 and b == 0:
            continue
        if a == 0 or b == 0:
            tile = (a - 1 if a > 0 else None, b - 1 if b > 0 else None)
        else:
            tile = (a - 1, b - 1)
        player_tiles.append(tile)

    left_end = int(input_seq[70])
    right_end = int(input_seq[71])
    left_end = (left_end - 1) if left_end > 0 else None
    right_end = (right_end - 1) if right_end > 0 else None

    return table_tiles, player_tiles, left_end, right_end


def decode_label_sequence(label_seq: np.ndarray):
    # First 3 positions encode the target; rest are padding/ignored
    a = int(label_seq[0])
    b = int(label_seq[1])
    pos = int(label_seq[2])
    tile = None
    if a > 0 and b > 0:
        tile = (a - 1, b - 1)
    pos_map = {0: None, 1: "left", 2: "right", 3: "first"}
    return tile, pos_map.get(pos, None)


def summarize_example(inputs: np.ndarray, labels: np.ndarray, idx: int):
    inp = inputs[idx]
    out = labels[idx]

    table_tiles, player_tiles, left_end, right_end = decode_input_sequence(inp)
    tile, pos = decode_label_sequence(out)

    checks = []
    # Validations
    if len(inp) != 72 or len(out) != 72:
        checks.append("seq_len_mismatch")
    if np.any(inp < 0) or np.any(inp > 7):
        checks.append("input_out_of_vocab")
    if np.any(out < 0) or np.any(out > 7):
        checks.append("label_out_of_vocab")
    if out[2] not in (0, 1, 2, 3):
        checks.append("position_invalid")

    # Basic validity checks: target tile must be in player's hand and consistent with ends
    if tile is not None:
        if tile not in player_tiles:
            checks.append("target_not_in_hand")
        # Consistency with ends
        if pos == "left":
            if left_end is not None and (tile[0] != left_end and tile[1] != left_end):
                checks.append("target_not_match_left_end")
        elif pos == "right":
            if right_end is not None and (tile[0] != right_end and tile[1] != right_end):
                checks.append("target_not_match_right_end")
        elif pos == "first":
            if left_end is not None or right_end is not None:
                checks.append("first_on_nonempty_table")

    return {
        "decoded": {
            "table_tiles": table_tiles,
            "player_tiles": player_tiles,
            "left_end": left_end,
            "right_end": right_end,
            "target_tile": tile,
            "target_position": pos,
        },
        "raw": {
            "inputs": inp.tolist(),
            "labels": out.tolist(),
        },
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect Domino HRM dataset")
    parser.add_argument("--data-path", type=str, default="data/domino-optimal-play", help="Path to dataset root (containing train/test)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Split to inspect")
    parser.add_argument("--num", type=int, default=5, help="Number of examples to show")
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
        d = summary["decoded"]
        print(f"Table: {d['table_tiles']}")
        print(f"Player: {d['player_tiles']}")
        print(f"Ends: {d['left_end']} - {d['right_end']}")
        print(f"Target: tile={d['target_tile']} position={d['target_position']}")
        if summary["checks"]:
            print(f"Checks: {summary['checks']}")
        print("Raw inputs:", summary["raw"]["inputs"]) 
        print("Raw labels:", summary["raw"]["labels"]) 
        print()


if __name__ == "__main__":
    main()


