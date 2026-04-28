import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import typer
from cshogi import (
    BLACK,
    WHITE,
    NOT_REPETITION,
    REPETITION_DRAW,
    REPETITION_WIN,
    REPETITION_LOSE,
    REPETITION_SUPERIOR,
    REPETITION_INFERIOR,
    move_to_usi,
)

from player.mcts_player import MCTSPlayer
from util.datawriter import HcpeDataWriter
from util.directory import ensure_directory_exists

DEFAULT_SELFPLAY_NODES = 800
DEFAULT_SELFPLAY_GAMES = 10
DEFAULT_MAX_MOVES = 512
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TEMPERATURE_MOVES = 30
DEFAULT_DIRICHLET_ALPHA = 0.15
DEFAULT_DIRICHLET_EPSILON = 0.25

selfplay_app = typer.Typer()

_BLOCK_KEY_PATTERN = re.compile(r"^blocks\.(\d+)\.")


class SelfPlayMCTSPlayer(MCTSPlayer):
    def __init__(self) -> None:
        super().__init__()
        self.selfplay_temperature: float = DEFAULT_TEMPERATURE
        self.selfplay_temperature_moves: int = DEFAULT_TEMPERATURE_MOVES
        self.selfplay_dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
        self.selfplay_dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON

    def _apply_root_dirichlet_noise(self) -> None:
        current_node = self.tree.current_head
        if current_node is None or current_node.policy is None or current_node.child_move is None:
            return
        if len(current_node.child_move) <= 1:
            return
        alpha = max(1e-6, self.selfplay_dirichlet_alpha)
        epsilon = min(1.0, max(0.0, self.selfplay_dirichlet_epsilon))
        eta = np.random.dirichlet([alpha] * len(current_node.policy)).astype(np.float32)
        current_node.policy = (1.0 - epsilon) * current_node.policy + epsilon * eta

    def _policy_from_visit_counts(self, counts: np.ndarray, use_temperature: bool) -> np.ndarray:
        counts = counts.astype(np.float64)
        total = float(np.sum(counts))
        if total <= 0.0:
            return np.repeat(1.0 / len(counts), len(counts))

        if use_temperature and self.selfplay_temperature > 0:
            inv_temp = 1.0 / max(1e-6, self.selfplay_temperature)
            weights = np.power(np.maximum(counts, 1e-12), inv_temp)
            weight_sum = float(np.sum(weights))
            if weight_sum > 0.0:
                return weights / weight_sum

        return counts / total

    def _winner_from_special(self, token: str) -> int:
        # resign: 手番側の負け, win: 手番側の勝ち
        if token == "resign":
            return -1 if self.root_board.turn == BLACK else 1
        if token == "win":
            return 1 if self.root_board.turn == BLACK else -1
        return 0

    def select_selfplay_move(self, nodes: int) -> str:
        self.begin_time = 0.0
        current_node = self.tree.current_head
        if current_node is None:
            raise ValueError("tree.current_head is None")

        if self.root_board.is_game_over():
            return "resign"
        if self.root_board.is_nyugyoku():
            return "win"

        self.set_limits(nodes=nodes)
        self.playout_count = 0

        if current_node.child_move is None:
            current_node.expand_node(self.root_board)
        if current_node.policy is None:
            self.current_batch_index = 0
            self.queue_node(self.root_board, current_node)
            self.eval_node()

        self._apply_root_dirichlet_noise()
        self.search()

        if current_node.child_move is None or current_node.child_move_count is None:
            return "resign"
        if len(current_node.child_move) == 0:
            return "resign"
        if len(current_node.child_move) == 1:
            return move_to_usi(current_node.child_move[0])

        use_temperature = self.root_board.move_number <= self.selfplay_temperature_moves
        probs = self._policy_from_visit_counts(current_node.child_move_count, use_temperature=use_temperature)
        selected_index = int(np.random.choice(np.arange(len(current_node.child_move)), p=probs))
        return move_to_usi(current_node.child_move[selected_index])

    def _winner_from_repetition(self, draw: int) -> int:
        if draw in (REPETITION_DRAW, REPETITION_SUPERIOR, REPETITION_INFERIOR):
            return 0
        if draw == REPETITION_WIN:
            return 1 if self.root_board.turn == BLACK else -1
        if draw == REPETITION_LOSE:
            return -1 if self.root_board.turn == BLACK else 1
        return 0

    def selfplay_one_game(self, output_path: str, nodes: int, max_moves: int) -> int:
        writer = HcpeDataWriter()
        writer.reset()
        usi_moves: list[str] = []
        winner = 0

        for _ in range(max_moves):
            self.position("startpos", usi_moves)
            move = self.select_selfplay_move(nodes=nodes)
            if move in ("resign", "win"):
                winner = self._winner_from_special(move)
                break

            writer.push(move)
            self.root_board.push_usi(move)
            usi_moves.append(move)

            draw = self.root_board.is_draw()
            if draw != NOT_REPETITION:
                winner = self._winner_from_repetition(draw)
                break

            if self.root_board.is_game_over():
                winner = -1 if self.root_board.turn == BLACK else 1
                break
            if self.root_board.is_nyugyoku():
                winner = 1 if self.root_board.turn == BLACK else -1
                break

        writer.finalize(winner=winner, filename=output_path)
        return winner


def infer_resnet_blocks_from_checkpoint(modelfile: str) -> Optional[int]:
    if not os.path.exists(modelfile):
        return None
    checkpoint = torch.load(modelfile, map_location="cpu")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        return None

    max_index = -1
    for key in model_state.keys():
        match = _BLOCK_KEY_PATTERN.match(key)
        if match:
            max_index = max(max_index, int(match.group(1)))
    if max_index < 0:
        return None
    return max_index + 1


@selfplay_app.command()
def generate(
    output: str = typer.Option("train_data/selfplay_latest.hcpe", help="Output HCPE file path"),
    games: int = typer.Option(DEFAULT_SELFPLAY_GAMES, help="Number of self-play games"),
    nodes: int = typer.Option(DEFAULT_SELFPLAY_NODES, help="Playouts per move"),
    max_moves: int = typer.Option(DEFAULT_MAX_MOVES, help="Max plies per game"),
    overwrite: bool = typer.Option(True, help="Overwrite output file before generation"),
    modelfile: Optional[str] = typer.Option(None, help="Model checkpoint path"),
    gpu_id: int = typer.Option(0, help="GPU ID (-1 for CPU)"),
    batchsize: int = typer.Option(32, help="Evaluation batch size"),
    blocks: Optional[int] = typer.Option(None, help="ResNet blocks (auto if omitted)"),
    activation_function: str = typer.Option("relu", help="Activation function"),
    c_puct: float = typer.Option(1.0, help="PUCT constant"),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE, help="Self-play temperature"),
    temperature_moves: int = typer.Option(DEFAULT_TEMPERATURE_MOVES, help="Temperature plies"),
    dirichlet_alpha: float = typer.Option(DEFAULT_DIRICHLET_ALPHA, help="Dirichlet alpha"),
    dirichlet_epsilon: float = typer.Option(DEFAULT_DIRICHLET_EPSILON, help="Dirichlet epsilon"),
    seed: Optional[int] = typer.Option(None, help="Random seed"),
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    ensure_directory_exists(output)
    if overwrite and os.path.exists(output):
        os.remove(output)

    player = SelfPlayMCTSPlayer()
    if modelfile:
        player.modelfile = modelfile
    inferred_blocks = infer_resnet_blocks_from_checkpoint(player.modelfile)
    player.gpu_id = gpu_id
    player.batch_size = batchsize
    if blocks is not None:
        player.blocks = blocks
    elif inferred_blocks is not None:
        player.blocks = inferred_blocks
    player.activation_function = activation_function
    player.c_puct = c_puct
    player.selfplay_temperature = temperature
    player.selfplay_temperature_moves = temperature_moves
    player.selfplay_dirichlet_alpha = dirichlet_alpha
    player.selfplay_dirichlet_epsilon = dirichlet_epsilon
    print(
        f"info string selfplay init model={player.modelfile} blocks={player.blocks} "
        f"inferred_blocks={inferred_blocks} activation_function={player.activation_function}",
        flush=True,
    )
    player.isready()

    black_wins = 0
    white_wins = 0
    draws = 0
    for i in range(games):
        result = player.selfplay_one_game(output_path=output, nodes=nodes, max_moves=max_moves)
        if result == 1:
            black_wins += 1
        elif result == -1:
            white_wins += 1
        else:
            draws += 1
        print(
            f"info string selfplay game={i + 1}/{games} result={result} "
            f"score black={black_wins} white={white_wins} draw={draws}",
            flush=True,
        )

    print(
        f"info string selfplay finished output={output} games={games} "
        f"black={black_wins} white={white_wins} draw={draws}",
        flush=True,
    )


if __name__ == "__main__":
    selfplay_app()
