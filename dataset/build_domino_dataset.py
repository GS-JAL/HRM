from typing import Optional, List, Tuple
import os
import csv
import json
import numpy as np
import random

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/domino-optimal-play"
    num_games: int = 10000
    subsample_size: Optional[int] = None
    num_aug: int = 0


class DominoGame:
    """Simple domino game state representation"""
    
    def __init__(self):
        # Create all domino tiles (0,0) to (6,6)
        self.all_tiles = [(i, j) for i in range(7) for j in range(i, 7)]
        self.reset_game()
    
    def reset_game(self):
        """Reset game state for a new game"""
        # Shuffle tiles and distribute
        tiles = self.all_tiles.copy()
        random.shuffle(tiles)
        
        # Each player gets 7 tiles (4 players), remaining go to boneyard
        self.player_tiles = tiles[:7]
        self.opponent_tiles = tiles[7:28]  # We don't know these during play
        
        # Table starts empty
        self.table_tiles = []
        self.left_end = None
        self.right_end = None
        
    def get_playable_ends(self) -> List[int]:
        """Get the values where tiles can be played"""
        if not self.table_tiles:
            return [0, 1, 2, 3, 4, 5, 6]  # First tile can be any value
        return [self.left_end, self.right_end] if self.left_end != self.right_end else [self.left_end]
    
    def can_play_tile(self, tile: Tuple[int, int]) -> List[str]:
        """Check if tile can be played and where"""
        playable_positions = []
        playable_ends = self.get_playable_ends()
        
        if not self.table_tiles:  # First tile
            playable_positions.append("first")
        else:
            if tile[0] in playable_ends or tile[1] in playable_ends:
                if self.left_end in tile:
                    playable_positions.append("left")
                if self.right_end in tile:
                    playable_positions.append("right")
                    
        return playable_positions
    
    def play_tile(self, tile: Tuple[int, int], position: str):
        """Play a tile at the specified position"""
        if position == "first":
            self.table_tiles.append(tile)
            self.left_end, self.right_end = tile
        elif position == "left":
            # Connect to left end
            if tile[0] == self.left_end:
                self.left_end = tile[1]
            else:
                self.left_end = tile[0]
            self.table_tiles.insert(0, tile)
        elif position == "right":
            # Connect to right end  
            if tile[0] == self.right_end:
                self.right_end = tile[1]
            else:
                self.right_end = tile[0]
            self.table_tiles.append(tile)
    
    def get_playable_tiles(self) -> List[Tuple[Tuple[int, int], List[str]]]:
        """Get all tiles that can be played and their positions"""
        playable = []
        for tile in self.player_tiles:
            positions = self.can_play_tile(tile)
            if positions:
                playable.append((tile, positions))
        return playable
    
    def simple_scoring_heuristic(self, tile: Tuple[int, int]) -> float:
        """Simple heuristic: prefer tiles with higher pip count to get rid of heavy tiles"""
        return sum(tile) + random.random() * 0.1  # Small random component for variety


def generate_domino_game_data(count: int) -> List[dict]:
    """Generate exactly `count` domino game scenarios with optimal moves."""
    scenarios: List[dict] = []
    attempts = 0
    max_attempts = count * 20  # safety
    
    while len(scenarios) < count and attempts < max_attempts:
        attempts += 1
        game = DominoGame()
        game.reset_game()
        
        # Play a few random moves to create different game states
        for _ in range(random.randint(0, 10)):
            playable = game.get_playable_tiles()
            if not playable:
                break
            
            tile, positions = random.choice(playable)
            position = random.choice(positions)
            game.play_tile(tile, position)
            game.player_tiles.remove(tile)
            
            # Simulate opponent plays (remove random tiles from their hand)
            if game.opponent_tiles:
                game.opponent_tiles.pop(random.randint(0, len(game.opponent_tiles)-1))
        
        # Now get the current state and optimal move
        playable = game.get_playable_tiles()
        if not playable:
            continue
        
        # Find "optimal" move using simple heuristic
        best_tile = None
        best_position = None
        best_score = -1.0
        
        for tile, positions in playable:
            score = game.simple_scoring_heuristic(tile)
            if score > best_score:
                best_score = score
                best_tile = tile
                best_position = positions[0]  # Take first valid position
        
        scenarios.append({
            'table_tiles': game.table_tiles.copy(),
            'player_tiles': game.player_tiles.copy(),
            'left_end': game.left_end,
            'right_end': game.right_end,
            'optimal_tile': best_tile,
            'optimal_position': best_position
        })
    
    return scenarios


def encode_game_state(scenario: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Encode game state into input/output arrays.

    Inputs: length-72 sequence
      - [0..55]: table representation (28 tiles max, each tile = 2 values)
      - [56..69]: player tiles (7 tiles max, each tile = 2 values)
      - [70..71]: left and right ends

    Labels: length-72 sequence (to match HRM seq_len requirements)
      - [0]: optimal tile value A (1..7), 0 if none
      - [1]: optimal tile value B (1..7), 0 if none
      - [2]: optimal position code (1=left, 2=right, 3=first), 0 if none
      - [3..71]: 0 (ignored by loss via ignore_label_id)
    """
    
    # Input encoding
    input_seq = np.zeros(72, dtype=np.int32)
    
    # Encode table tiles (max 28 tiles on table, each takes 2 positions)
    table_pos = 0
    for tile in scenario['table_tiles'][:28]:
        input_seq[table_pos] = tile[0] + 1      # +1 to avoid 0 (reserved for padding)
        input_seq[table_pos + 1] = tile[1] + 1
        table_pos += 2
    
    # Encode player tiles (max 7 tiles, each takes 2 positions)
    player_pos = 56
    for tile in scenario['player_tiles'][:7]:
        input_seq[player_pos] = tile[0] + 1
        input_seq[player_pos + 1] = tile[1] + 1
        player_pos += 2
    
    # Encode ends
    input_seq[70] = (scenario['left_end'] + 1) if scenario['left_end'] is not None else 0
    input_seq[71] = (scenario['right_end'] + 1) if scenario['right_end'] is not None else 0
    
    # Output encoding aligned to seq_len=72
    output_seq = np.zeros(72, dtype=np.int32)
    if scenario['optimal_tile']:
        output_seq[0] = scenario['optimal_tile'][0] + 1
        output_seq[1] = scenario['optimal_tile'][1] + 1
        position_map = {'left': 1, 'right': 2, 'first': 3}
        output_seq[2] = position_map.get(scenario['optimal_position'], 0)
    
    return input_seq, output_seq


def convert_subset(set_name: str, config: DataProcessConfig):
    """Convert domino game data to the required format"""
    
    # Generate game scenarios
    print(f"Generating {config.num_games} domino game scenarios...")
    scenarios = generate_domino_game_data(config.num_games)
    
    # If subsample_size is specified for training set
    if set_name == "train" and config.subsample_size is not None:
        if config.subsample_size < len(scenarios):
            scenarios = random.sample(scenarios, config.subsample_size)
    
    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0
    
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for scenario in tqdm(scenarios):
        for aug_idx in range(1 + num_augments):
            # Encode the game state
            inp, out = encode_game_state(scenario)
            
            # Add some augmentation by rotating tile values (simple augmentation)
            if aug_idx > 0:
                # Simple augmentation: add random offset to all tile values (mod 7)
                # Apply to inputs everywhere (nonzero entries)
                offset = random.randint(1, 6)
                inp = np.where(inp > 0, ((inp - 1 + offset) % 7) + 1, inp)
                # Apply ONLY to the first two output slots (tile values). Keep position code intact.
                if out[0] > 0:
                    out[0] = ((out[0] - 1 + offset) % 7) + 1
                if out[1] > 0:
                    out[1] = ((out[1] - 1 + offset) % 7) + 1
            
            # Add to results
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)  # All domino games have same identifier
        
        # Push group
        results["group_indices"].append(puzzle_id)
    
    # Convert to numpy
    results = {
        "inputs": np.array(results["inputs"], dtype=np.int32),
        "labels": np.array(results["labels"], dtype=np.int32),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    
    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=72,  # Match input and label sequence lengths
        vocab_size=8,  # 0 (pad) + 1..7 for domino values and position codes within 1..3
        
        pad_id=0,
        ignore_label_id=0,  # Ignore zeros in labels beyond the first 3 positions
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1.0 + num_augments,
        sets=["all"]
    )
    
    # Save metadata as JSON
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    # Save identifiers mapping (for visualization)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<domino_game>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Preprocess domino game data"""
    random.seed(42)  # For reproducible results
    np.random.seed(42)
    
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
