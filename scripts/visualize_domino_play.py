#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Tuple, Optional
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
    table_tiles: List[Tuple[int, int]] = []
    for i in range(0, 56, 2):
        a, b = int(input_seq[i]), int(input_seq[i + 1])
        if a == 0 and b == 0:
            continue
        table_tiles.append(((a - 1) if a > 0 else 0, (b - 1) if b > 0 else 0))

    player_tiles: List[Tuple[int, int]] = []
    for i in range(56, 70, 2):
        a, b = int(input_seq[i]), int(input_seq[i + 1])
        if a == 0 and b == 0:
            continue
        player_tiles.append(((a - 1) if a > 0 else 0, (b - 1) if b > 0 else 0))

    left_end = int(input_seq[70])
    right_end = int(input_seq[71])
    left_end = (left_end - 1) if left_end > 0 else None
    right_end = (right_end - 1) if right_end > 0 else None

    return table_tiles, player_tiles, left_end, right_end


def decode_label_sequence(label_seq: np.ndarray):
    a = int(label_seq[0])
    b = int(label_seq[1])
    pos = int(label_seq[2])
    tile = None
    if a > 0 and b > 0:
        tile = (a - 1, b - 1)
    pos_map = {0: None, 1: "left", 2: "right", 3: "first"}
    return tile, pos_map.get(pos, None)


def orient_chain(table_tiles: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not table_tiles:
        return []
    oriented: List[Tuple[int, int]] = [(table_tiles[0][0], table_tiles[0][1])]
    if len(table_tiles) >= 2:
        next_tile = table_tiles[1]
        # Prefer orientation so that right side of first matches one value in next
        if oriented[0][1] not in next_tile:
            if oriented[0][0] in next_tile:
                oriented[0] = (oriented[0][1], oriented[0][0])
    for i in range(1, len(table_tiles)):
        prev_right = oriented[i - 1][1]
        a, b = table_tiles[i]
        if a == prev_right:
            oriented.append((a, b))
        elif b == prev_right:
            oriented.append((b, a))
        else:
            # Fallback: keep as-is (should be rare)
            oriented.append((a, b))
    return oriented


def draw_tile(a: int, b: int) -> List[str]:
    def v(x: int) -> str:
        return str(x)
    top = "┌───┬───┐"
    mid = f"│ {v(a)} │ {v(b)} │"
    bot = "└───┴───┘"
    return [top, mid, bot]


def join_tiles_horiz(tiles_lines: List[List[str]], sep: str = " ") -> List[str]:
    if not tiles_lines:
        return []
    lines = [sep.join(parts) for parts in zip(*tiles_lines)]
    return lines


def center_text(width: int, text: str) -> str:
    if len(text) >= width:
        return text[:width]
    pad_left = (width - len(text)) // 2
    pad_right = width - len(text) - pad_left
    return (" " * pad_left) + text + (" " * pad_right)


def draw_chain_ascii(table_tiles: List[Tuple[int, int]], left_end: Optional[int], right_end: Optional[int]) -> str:
    oriented = orient_chain(table_tiles)
    tiles_drawn = [draw_tile(a, b) for (a, b) in oriented]
    chain_lines = join_tiles_horiz(tiles_drawn)

    # Add ends annotation line above
    if chain_lines:
        tile_width = len(tiles_drawn[0][0])
        left_label = f"L={left_end if left_end is not None else '-'}"
        right_label = f"R={right_end if right_end is not None else '-'}"
        left_block = center_text(tile_width, left_label)
        right_block = center_text(tile_width, right_label)
        spacer = " " * (len(chain_lines[0]) - 2 * tile_width)
        header = left_block + spacer + right_block
        return "\n".join([header] + chain_lines)
    else:
        # Empty chain
        return f"[Empty table] (L={left_end}, R={right_end})"


def compute_playable(player_tiles: List[Tuple[int, int]], left_end: Optional[int], right_end: Optional[int]) -> List[bool]:
    playable = []
    if left_end is None and right_end is None:
        # First tile can be any
        return [True for _ in player_tiles]
    for (a, b) in player_tiles:
        ok = False
        if left_end is not None and (a == left_end or b == left_end):
            ok = True
        if right_end is not None and (a == right_end or b == right_end):
            ok = True
        playable.append(ok)
    return playable


def draw_hand(player_tiles: List[Tuple[int, int]], playable_mask: List[bool], target_tile: Optional[Tuple[int, int]]) -> str:
    # Build per-tile blocks with marker line
    tiles_blocks: List[List[str]] = []
    for idx, (a, b) in enumerate(player_tiles):
        block = draw_tile(a, b)
        marker = ""
        if target_tile is not None and ((a, b) == target_tile or (b, a) == target_tile):
            marker = ">>"
        elif playable_mask[idx]:
            marker = "*"
        marker_line = center_text(len(block[0]), marker)
        tiles_blocks.append(block + [marker_line])

    if not tiles_blocks:
        return "[Empty hand]"
    lines = [" ".join(parts) for parts in zip(*tiles_blocks)]
    return "\n".join(lines)


def summarize_example(inputs: np.ndarray, labels: np.ndarray, idx: int):
    inp = inputs[idx]
    out = labels[idx]
    table_tiles, player_tiles, left_end, right_end = decode_input_sequence(inp)
    tile, pos = decode_label_sequence(out)
    return table_tiles, player_tiles, left_end, right_end, tile, pos


def main():
    parser = argparse.ArgumentParser(description="Console Domino Visualizer")
    parser.add_argument("--data-path", type=str, default="data/domino-opt", help="Path to domino dataset root")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Split")
    parser.add_argument("--index", type=int, default=0, help="Example index to visualize")
    args = parser.parse_args()

    _, data = load_dataset_split(args.data_path, args.split)
    inputs = data["inputs"]
    labels = data["labels"]

    if args.index < 0 or args.index >= len(inputs):
        raise IndexError(f"Index {args.index} out of range (0..{len(inputs)-1})")

    table_tiles, player_tiles, left_end, right_end, target_tile, target_pos = summarize_example(inputs, labels, args.index)

    print(f"=== Domino Visualization: {args.split} #{args.index} ===")
    print()
    print("Table (left → right):")
    print(draw_chain_ascii(table_tiles, left_end, right_end))
    print()
    print("Hand:")
    playable_mask = compute_playable(player_tiles, left_end, right_end)
    print(draw_hand(player_tiles, playable_mask, target_tile))
    print()
    print(f"Target move: tile={target_tile} position={target_pos}")


if __name__ == "__main__":
    main()


