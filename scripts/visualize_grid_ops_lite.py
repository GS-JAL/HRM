import os
import json
import argparse
import numpy as np

PAD = 0
SEP = 1
TOK_DOT = 10
TOK_WALL = 11
TOK_S = 12
TOK_T = 13
TOK_FALSE = 20
TOK_TRUE = 21

TOKEN_TO_CHAR = {
    TOK_DOT: '.',
    TOK_WALL: '#',
    TOK_S: 'S',
    TOK_T: 'T',
}

LABEL_TO_CHAR = {
    TOK_FALSE: '·',
    TOK_TRUE: '*',
}


def load_set(dataset_dir: str, split: str, set_name: str):
    inputs = np.load(os.path.join(dataset_dir, split, f"{set_name}__inputs.npy"))
    labels = np.load(os.path.join(dataset_dir, split, f"{set_name}__labels.npy"))
    pids   = np.load(os.path.join(dataset_dir, split, f"{set_name}__puzzle_identifiers.npy"))
    return inputs, labels, pids


def decode_header(arr):
    return {
        "SEP0": int(arr[0]),
        "OP_CODE": int(arr[1]),
        "H": int(arr[2]),
        "W": int(arr[3]),
        "R1": int(arr[4]),
        "R2": int(arr[5]),
        "R3": int(arr[6]),
        "SEP1": int(arr[7]),
    }


def render_example(inp: np.ndarray, lab: np.ndarray, pid: int) -> str:
    hdr = decode_header(inp[:8])
    H, W = hdr["H"], hdr["W"]
    payload = inp[8:8+H*W]
    grid_chars = [TOKEN_TO_CHAR.get(int(t), '?') for t in payload]

    # labels
    if pid in (1, 2):  # masks
        mask = lab[8:8+H*W]
        mask_chars = [LABEL_TO_CHAR.get(int(t), ' ') for t in mask]
    else:  # floodfill size encoded in first positions as digits (optional)
        mask = lab[8:8+H*W]
        mask_chars = [' ' for _ in range(H*W)]

    # Build lines
    lines = []
    lines.append(f"pid={pid} op={hdr['OP_CODE']} HxW={H}x{W}")
    for r in range(H):
        row_g = ' '.join(grid_chars[r*W:(r+1)*W])
        row_m = ' '.join(mask_chars[r*W:(r+1)*W])
        lines.append(f"G {row_g}    M {row_m}")

    # Show scalar if floodfill
    if pid == 3:
        # Decode three-digit number if present
        h_tok = int(lab[8])
        t_tok = int(lab[9])
        o_tok = int(lab[10])
        digits = []
        for tok in (h_tok, t_tok, o_tok):
            if tok >= 30:
                digits.append(tok - 30)
            else:
                digits.append(0)
        size = digits[0] * 100 + digits[1] * 10 + digits[2]
        lines.append(f"floodfill_size={size}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--set", dest="set_name", default="easy")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    inputs, labels, pids = load_set(args.dataset_dir, args.split, args.set_name)

    end = min(inputs.shape[0], args.start + args.num)
    for i in range(args.start, end):
        print("="*60)
        print(f"Example {i}")
        print(render_example(inputs[i], labels[i], int(pids[i])))


if __name__ == "__main__":
    main()
