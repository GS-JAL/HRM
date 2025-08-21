from typing import List, Tuple, Optional
import os
import json
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
from argdantic import ArgParser

# Make runnable both as module and as script
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.common import PuzzleDatasetMetadata, dihedral_transform

# Token definitions (shared across inputs/labels)
PAD = 0
SEP = 1

# Grid tokens (inputs)
TOK_DOT = 10
TOK_WALL = 11
TOK_S = 12
TOK_T = 13

# Label mask tokens
TOK_FALSE = 20
TOK_TRUE = 21

# Digit tokens (for scalar outputs encoded as two digits)
TOK_DIGITS = {d: 30 + d for d in range(10)}

HEADER_LEN = 8  # [SEP, OP_CODE, H, W, R1, R2, R3, SEP]

# OP codes (for interpretability/visualization only)
OP_REACHABLE = 20
OP_SHORTEST_PATH = 21
OP_FLOODFILL_SIZE = 22

# puzzle_identifiers (task IDs): keep 0 reserved for <blank>
PID_REACHABLE = 1
PID_SHORTEST_PATH = 2
PID_FLOODFILL_SIZE = 3


cli = ArgParser()


class BuildConfig(BaseModel):
    output_dir: str = "data/grid-ops-lite"

    # Train sizes per curriculum stage
    train_easy_size: int = 2000
    train_medium_size: int = 2000
    train_hard_size: int = 2000

    # Test size (single set)
    test_size: int = 1000

    # Grid parameter ranges per stage (inclusive bounds)
    easy_H_min: int = 4
    easy_H_max: int = 8
    easy_W_min: int = 4
    easy_W_max: int = 8
    easy_obstacle_prob_min: float = 0.05
    easy_obstacle_prob_max: float = 0.12

    medium_H_min: int = 8
    medium_H_max: int = 16
    medium_W_min: int = 8
    medium_W_max: int = 16
    medium_obstacle_prob_min: float = 0.12
    medium_obstacle_prob_max: float = 0.22

    hard_H_min: int = 16
    hard_H_max: int = 30
    hard_W_min: int = 16
    hard_W_max: int = 30
    hard_obstacle_prob_min: float = 0.20
    hard_obstacle_prob_max: float = 0.30

    # Data augmentation (shape-preserving dihedral transforms)
    aug: bool = True
    seed: int = 42


@dataclass
class EncodedExample:
    inputs: np.ndarray  # [seq_len]
    labels: np.ndarray  # [seq_len]
    puzzle_identifier: int  # 1..3


# ------------------------ Grid generation utils ------------------------ #

def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.Philox(seed=seed))


def generate_grid(H: int, W: int, p_wall: float, rng: np.random.Generator) -> np.ndarray:
    grid = np.full((H, W), TOK_DOT, dtype=np.int32)
    walls = rng.random((H, W)) < p_wall
    grid[walls] = TOK_WALL
    return grid


def place_S_T(grid: np.ndarray, need_T: bool, rng: np.random.Generator) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
    H, W = grid.shape
    free_cells = np.argwhere(grid == TOK_DOT)
    if free_cells.size == 0:
        raise ValueError("No free cells to place S/T")
    s_idx = rng.integers(0, free_cells.shape[0])
    sy, sx = free_cells[s_idx]
    grid[sy, sx] = TOK_S

    t_pos = None
    if need_T:
        # Ensure T is different from S
        candidates = np.argwhere(grid == TOK_DOT)
        if candidates.size == 0:
            # Fallback: convert one wall to dot and place T
            wy, wx = rng.integers(0, H), rng.integers(0, W)
            grid[wy, wx] = TOK_DOT
            candidates = np.array([[wy, wx]], dtype=np.int64)
        t_idx = rng.integers(0, candidates.shape[0])
        ty, tx = candidates[t_idx]
        grid[ty, tx] = TOK_T
        t_pos = (ty, tx)

    return (sy, sx), t_pos


def bfs_all_reachable(grid: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    H, W = grid.shape
    qy = [start[0]]
    qx = [start[1]]
    head = 0
    visited = np.zeros((H, W), dtype=bool)
    visited[start] = True

    def can(y: int, x: int) -> bool:
        return 0 <= y < H and 0 <= x < W and (grid[y, x] in (TOK_DOT, TOK_S, TOK_T)) and not visited[y, x]

    while head < len(qy):
        y, x = qy[head], qx[head]
        head += 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if can(ny, nx):
                visited[ny, nx] = True
                qy.append(ny)
                qx.append(nx)
    return visited


def bfs_shortest_path(grid: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    H, W = grid.shape
    from collections import deque

    q = deque([start])
    prev = {start: None}

    def can(y: int, x: int) -> bool:
        return 0 <= y < H and 0 <= x < W and (grid[y, x] in (TOK_DOT, TOK_S, TOK_T))

    # Fixed neighbor order for canonical path
    neigh = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while q:
        y, x = q.popleft()
        if (y, x) == target:
            # reconstruct
            path = []
            cur = target
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        for dy, dx in neigh:
            ny, nx = y + dy, x + dx
            if can(ny, nx) and (ny, nx) not in prev:
                prev[(ny, nx)] = (y, x)
                q.append((ny, nx))
    return None


# ------------------------ Encoding ------------------------ #

def build_header(op_code: int, H: int, W: int) -> np.ndarray:
    header = np.zeros((HEADER_LEN,), dtype=np.int32)
    header[0] = SEP
    header[1] = op_code
    header[2] = H
    header[3] = W
    header[-1] = SEP
    return header


def encode_reachable(grid: np.ndarray) -> EncodedExample:
    H, W = grid.shape
    # Find S
    sy, sx = map(int, np.argwhere(grid == TOK_S)[0])
    reachable = bfs_all_reachable(grid, (sy, sx))

    header = build_header(OP_REACHABLE, H, W)
    payload = grid.reshape(-1)

    inputs = np.concatenate([header, payload])

    labels = np.zeros_like(inputs)
    mask = np.where(reachable.reshape(-1), TOK_TRUE, TOK_FALSE).astype(np.int32)
    labels[HEADER_LEN:HEADER_LEN + H * W] = mask

    return EncodedExample(inputs=inputs, labels=labels, puzzle_identifier=PID_REACHABLE)


def encode_shortest_path(grid: np.ndarray) -> EncodedExample:
    H, W = grid.shape
    sy, sx = map(int, np.argwhere(grid == TOK_S)[0])
    ty, tx = map(int, np.argwhere(grid == TOK_T)[0])
    path = bfs_shortest_path(grid, (sy, sx), (ty, tx))

    header = build_header(OP_SHORTEST_PATH, H, W)
    payload = grid.reshape(-1)
    inputs = np.concatenate([header, payload])

    labels = np.zeros_like(inputs)
    mask = np.full((H * W,), TOK_FALSE, dtype=np.int32)
    if path is not None:
        for (y, x) in path:
            mask[y * W + x] = TOK_TRUE
    labels[HEADER_LEN:HEADER_LEN + H * W] = mask

    return EncodedExample(inputs=inputs, labels=labels, puzzle_identifier=PID_SHORTEST_PATH)


def encode_floodfill_size(grid: np.ndarray) -> EncodedExample:
    H, W = grid.shape
    sy, sx = map(int, np.argwhere(grid == TOK_S)[0])
    reachable = bfs_all_reachable(grid, (sy, sx))
    size = int(reachable.sum())

    header = build_header(OP_FLOODFILL_SIZE, H, W)
    payload = grid.reshape(-1)
    inputs = np.concatenate([header, payload])

    labels = np.zeros_like(inputs)
    # Encode size as three digits (hundreds, tens, ones) at the first three payload positions
    hundreds = size // 100
    tens = (size % 100) // 10
    ones = size % 10
    labels[HEADER_LEN + 0] = TOK_DIGITS[hundreds]
    labels[HEADER_LEN + 1] = TOK_DIGITS[tens]
    labels[HEADER_LEN + 2] = TOK_DIGITS[ones]

    return EncodedExample(inputs=inputs, labels=labels, puzzle_identifier=PID_FLOODFILL_SIZE)


# ------------------------ Dataset building ------------------------ #

def sample_reachable(H: int, W: int, p_wall: float, rng: np.random.Generator) -> EncodedExample:
    # Ensure at least one free cell for S
    while True:
        grid = generate_grid(H, W, p_wall, rng)
        try:
            (sy, sx), _ = place_S_T(grid, need_T=False, rng=rng)
            # Avoid trivial all walls around S by regenerating a few times if isolated
            reachable = bfs_all_reachable(grid, (sy, sx))
            if reachable.sum() >= 1:
                return encode_reachable(grid)
        except ValueError:
            continue


def sample_shortest_path(H: int, W: int, p_wall: float, rng: np.random.Generator) -> EncodedExample:
    # Ensure an existing path S->T by retrying
    for _ in range(100):
        grid = generate_grid(H, W, p_wall, rng)
        try:
            start, t_pos = place_S_T(grid, need_T=True, rng=rng)
        except ValueError:
            continue
        path = bfs_shortest_path(grid, start, t_pos)  # type: ignore
        if path is not None:
            return encode_shortest_path(grid)
    # Fallback: carve a simple corridor
    grid = np.full((H, W), TOK_DOT, dtype=np.int32)
    sy, sx = 0, 0
    ty, tx = H - 1, W - 1
    grid[sy, sx] = TOK_S
    grid[ty, tx] = TOK_T
    return encode_shortest_path(grid)


def sample_floodfill(H: int, W: int, p_wall: float, rng: np.random.Generator) -> EncodedExample:
    while True:
        grid = generate_grid(H, W, p_wall, rng)
        try:
            start, _ = place_S_T(grid, need_T=False, rng=rng)
        except ValueError:
            continue
        # Accept any grid (count will vary)
        return encode_floodfill_size(grid)


def build_split(split: str, sets_spec: List[Tuple[str, int]], config: BuildConfig, global_seed: int):
    """Build a split with multiple curriculum sets.

    sets_spec: list of tuples (set_name, count)
    For each set, H, W, obstacle_prob are sampled per-example within configured ranges.
    """
    save_dir = os.path.join(config.output_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    # Determine split-wide seq_len for padding
    if split == "train":
        split_max_HW = max(
            config.hard_H_max * config.hard_W_max,
            config.medium_H_max * config.medium_W_max,
            config.easy_H_max * config.easy_W_max,
        )
    else:
        # Use medium bounds for test by default
        split_max_HW = config.medium_H_max * config.medium_W_max
    split_seq_len = HEADER_LEN + split_max_HW

    # For each set, collect examples and indices
    set_data = {}
    for set_name, count in sets_spec:
        rng = _rng(global_seed ^ (hash((split, set_name)) & 0xFFFFFFFF))

        inputs_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        puzzle_ids: List[int] = []
        puzzle_indices: List[int] = [0]
        group_indices: List[int] = [0]

        for i in tqdm(range(count), desc=f"{split}/{set_name}"):
            # Sample size and density per curriculum stage
            if set_name == "easy":
                H = int(rng.integers(config.easy_H_min, config.easy_H_max + 1))
                W = int(rng.integers(config.easy_W_min, config.easy_W_max + 1))
                p_wall = float(rng.uniform(config.easy_obstacle_prob_min, config.easy_obstacle_prob_max))
            elif set_name == "medium":
                H = int(rng.integers(config.medium_H_min, config.medium_H_max + 1))
                W = int(rng.integers(config.medium_W_min, config.medium_W_max + 1))
                p_wall = float(rng.uniform(config.medium_obstacle_prob_min, config.medium_obstacle_prob_max))
            elif set_name == "hard":
                H = int(rng.integers(config.hard_H_min, config.hard_H_max + 1))
                W = int(rng.integers(config.hard_W_min, config.hard_W_max + 1))
                p_wall = float(rng.uniform(config.hard_obstacle_prob_min, config.hard_obstacle_prob_max))
            else:
                # Unknown set name: default to medium bounds so it fits test split
                H = int(rng.integers(config.medium_H_min, config.medium_H_max + 1))
                W = int(rng.integers(config.medium_W_min, config.medium_W_max + 1))
                p_wall = float(rng.uniform(config.medium_obstacle_prob_min, config.medium_obstacle_prob_max))

            # Round-robin tasks for balance
            mod = i % 3
            if mod == 0:
                ex = sample_reachable(H, W, p_wall, rng)
            elif mod == 1:
                ex = sample_shortest_path(H, W, p_wall, rng)
            else:
                ex = sample_floodfill(H, W, p_wall, rng)

            # Optional dihedral augmentation for training only
            aug_times = 8 if (split == "train" and config.aug) else 1
            for aug_idx in range(aug_times):
                if aug_idx == 0:
                    in_arr = ex.inputs
                    lb_arr = ex.labels
                else:
                    # Only payload is transformed; header stays
                    Hh, Ww = H, W
                    payload = ex.inputs[HEADER_LEN:HEADER_LEN + Hh * Ww].reshape(Hh, Ww)
                    payload_t = dihedral_transform(payload, aug_idx)

                    in_arr = ex.inputs.copy()
                    in_arr[HEADER_LEN:HEADER_LEN + Hh * Ww] = payload_t.reshape(-1)

                    # Labels: transform masks for PID 1/2; keep scalar for PID 3
                    if ex.puzzle_identifier in (PID_REACHABLE, PID_SHORTEST_PATH):
                        label_payload = ex.labels[HEADER_LEN:HEADER_LEN + Hh * Ww].reshape(Hh, Ww)
                        label_payload_t = dihedral_transform(label_payload, aug_idx)
                        lb_arr = ex.labels.copy()
                        lb_arr[HEADER_LEN:HEADER_LEN + Hh * Ww] = label_payload_t.reshape(-1)
                    else:
                        lb_arr = ex.labels.copy()

                # Pad to split-wide seq_len
                cur_len = HEADER_LEN + H * W
                if cur_len < split_seq_len:
                    pad_in = np.zeros((split_seq_len - cur_len,), dtype=np.int32)
                    pad_lb = np.zeros((split_seq_len - cur_len,), dtype=np.int32)
                    in_arr = np.concatenate([in_arr.astype(np.int32), pad_in])
                    lb_arr = np.concatenate([lb_arr.astype(np.int32), pad_lb])

                inputs_list.append(in_arr.astype(np.int32))
                labels_list.append(lb_arr.astype(np.int32))
                puzzle_ids.append(ex.puzzle_identifier)

                # One example per puzzle
                next_example_index = len(inputs_list)
                puzzle_indices.append(next_example_index)
                group_indices.append(len(puzzle_ids))

        # Stack arrays
        inputs = np.vstack(inputs_list).astype(np.int32)
        labels = np.vstack(labels_list).astype(np.int32)
        puzzle_ids_arr = np.array(puzzle_ids, dtype=np.int32)
        puzzle_indices_arr = np.array(puzzle_indices, dtype=np.int32)
        group_indices_arr = np.array(group_indices, dtype=np.int32)

        # Save per-set arrays
        np.save(os.path.join(save_dir, f"{set_name}__inputs.npy"), inputs)
        np.save(os.path.join(save_dir, f"{set_name}__labels.npy"), labels)
        np.save(os.path.join(save_dir, f"{set_name}__puzzle_identifiers.npy"), puzzle_ids_arr)
        np.save(os.path.join(save_dir, f"{set_name}__puzzle_indices.npy"), puzzle_indices_arr)
        np.save(os.path.join(save_dir, f"{set_name}__group_indices.npy"), group_indices_arr)

    # Metadata (shared for split)
    # seq_len = HEADER_LEN + H*W; H, W vary, so take worst-case per split
    seq_len = split_seq_len

    vocab_size = max(
        SEP,
        TOK_T,
        TOK_TRUE,
        max(TOK_DIGITS.values())
    ) + 1

    # Train split may have multiple sets for curriculum; test uses single set "all"
    metadata = PuzzleDatasetMetadata(
        pad_id=PAD,
        ignore_label_id=PAD,
        blank_identifier_id=0,  # 0 reserved for <blank>
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=4,  # 0 blank + 3 tasks
        total_groups=sum(count * (8 if (split == "train" and config.aug) else 1) for _name, count, *_rest in sets_spec),
        mean_puzzle_examples=1,
        sets=[name for name, *_rest in sets_spec],
    )

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # identifiers mapping at root
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>", "REACHABLE_MASK", "SHORTEST_PATH_MASK", "FLOODFILL_SIZE"], f)


@cli.command(singleton=True)
def build(config: BuildConfig):
    rng = _rng(config.seed)
    del rng  # seed only used for determinism in internal RNGs

    # Train split with curriculum sets
    train_sets = [
        ("easy", config.train_easy_size),
        ("medium", config.train_medium_size),
        ("hard", config.train_hard_size),
    ]
    build_split("train", train_sets, config, global_seed=config.seed)

    # Test split single set
    test_sets = [("all", config.test_size)]
    build_split("test", test_sets, config, global_seed=config.seed + 1)


if __name__ == "__main__":
    cli()
