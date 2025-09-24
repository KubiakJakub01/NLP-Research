import json
import os
from typing import Any

import chess
import chess.pgn
import numpy as np
import zstandard
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from .common import PuzzleDatasetMetadata

cli = ArgParser()


class DataProcessConfig(BaseModel):
    dataset_dirs: list[str] = ['data/chess/downloads']
    output_dir: str = 'data/chess-1000'
    seed: int = 42
    train_eval_split_ratio: float = 0.95
    max_games_per_file: int = 1000


ChessMaxGridSize = 8
HRMMaxGridSize = 30

PIECE_TO_INT = {
    (chess.PAWN, chess.WHITE): 1,
    (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3,
    (chess.ROOK, chess.WHITE): 4,
    (chess.QUEEN, chess.WHITE): 5,
    (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 7,
    (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9,
    (chess.ROOK, chess.BLACK): 10,
    (chess.QUEEN, chess.BLACK): 11,
    (chess.KING, chess.BLACK): 12,
}


def board_to_np(board: chess.Board) -> np.ndarray:
    arr = np.zeros((ChessMaxGridSize, ChessMaxGridSize), dtype=np.uint8)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row = i // 8
            col = i % 8
            arr[row, col] = PIECE_TO_INT[(piece.piece_type, piece.color)]
    return arr


def np_grid_to_seq(inp: np.ndarray, out: np.ndarray):
    pad_r = (HRMMaxGridSize - ChessMaxGridSize) // 2
    pad_c = (HRMMaxGridSize - ChessMaxGridSize) // 2

    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        grid_copy = grid.copy()
        grid_copy[grid_copy > 0] += 2
        grid_copy[grid_copy == 0] = 2

        padded_grid = np.pad(
            grid_copy,
            ((pad_r, HRMMaxGridSize - pad_r - nrow), (pad_c, HRMMaxGridSize - pad_c - ncol)),
            constant_values=0,
        )

        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < HRMMaxGridSize:
            padded_grid[eos_row, pad_c:eos_col] = 1
        if eos_col < HRMMaxGridSize:
            padded_grid[pad_r:eos_row, eos_col] = 1

        result.append(padded_grid.flatten())

    return result


def process_pgn_file(pgn_path: str, max_games: int):
    examples = []
    games_count = 0
    with open(pgn_path, 'rb') as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            pgn_file = chess.pgn.io.TextIOWrapper(reader, encoding='utf-8')  # pylint: disable=no-member
            while True:
                if max_games != -1 and games_count >= max_games:
                    break
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                game_examples = []
                board = game.board()
                for move in game.mainline_moves():
                    board_before = board.copy()
                    board.push(move)
                    board_after = board.copy()
                    game_examples.append((board_to_np(board_before), board_to_np(board_after)))

                if game_examples:
                    examples.append(game_examples)
                games_count += 1
    return examples


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)

    pgn_files = []
    for dataset_dir in config.dataset_dirs:
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.pgn.zst'):
                    pgn_files.append(os.path.join(root, file))

    print(f'Found {len(pgn_files)} PGN files.')

    all_games = []
    for pgn_file in tqdm(pgn_files, desc='Processing PGN files'):
        all_games.extend(process_pgn_file(pgn_file, config.max_games_per_file))

    np.random.shuffle(all_games)

    split_idx = int(len(all_games) * config.train_eval_split_ratio)
    train_games = all_games[:split_idx]
    eval_games = all_games[split_idx:]

    data = {'train': {'all': train_games}, 'test': {'all': eval_games}}

    num_identifiers = 1

    for split_name, split in data.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        total_examples = 0
        total_puzzles = 0

        for subset_name, subset in split.items():
            results: dict[str, Any] = {
                k: []
                for k in [
                    'inputs',
                    'labels',
                    'puzzle_identifiers',
                    'puzzle_indices',
                ]
            }
            results['puzzle_indices'].append(0)

            example_id = 0

            for game in subset:
                for board_before, board_after in game:
                    seqs = np_grid_to_seq(board_before, board_after)
                    results['inputs'].append(seqs[0])
                    results['labels'].append(seqs[1])
                    example_id += 1
                    total_examples += 1

                results['puzzle_indices'].append(example_id)
                results['puzzle_identifiers'].append(num_identifiers)
                num_identifiers += 1
                total_puzzles += 1

            for k, v in results.items():
                v = np.stack(v, 0) if k in {'inputs', 'labels'} else np.array(v, dtype=np.int32)
                np.save(os.path.join(config.output_dir, split_name, f'{subset_name}__{k}.npy'), v)

        metadata = PuzzleDatasetMetadata(
            seq_len=HRMMaxGridSize * HRMMaxGridSize,
            vocab_size=12 + 2 + 2,
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_puzzles,
            mean_puzzle_examples=total_examples / total_puzzles if total_puzzles > 0 else 0,
            sets=list(split.keys()),
        )

        with open(
            os.path.join(config.output_dir, split_name, 'dataset.json'), 'w', encoding='utf-8'
        ) as f:
            json.dump(metadata.model_dump(), f)

    print('Dataset conversion finished.')


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == '__main__':
    cli()
