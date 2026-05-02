"""
search/spsa.py
SPSA (Simultaneous Perturbation Stochastic Approximation) による
UW-NPLSパラメータ自動調整。

使い方：
  python -m search.spsa --games_per_iter 4 --total_iter 200 \\
                         --output spsa_result.json

SPSAの概要：
  - 全パラメータを同時にランダム摂動（±c_k）して2回対局
  - 勝率差から勾配を近似してパラメータを更新
  - Stockfish / Leela Chess Zero でも使われている実績ある手法
  - 数式：θ_{k+1} = θ_k - a_k × g_k
      a_k = a / (k + 1 + A)^alpha    （学習率：徐々に減衰）
      c_k = c / (k + 1)^gamma         （摂動幅：徐々に縮小）
      g_k ≈ (L_+ - L_-) / (2 c_k Δθ) （勾配近似）
"""

import math
import random
import json
import copy
import argparse
import subprocess
import threading
import time
from typing import Dict, Tuple, List, Optional, Callable


# ===========================================================
# パラメータ定義
# ===========================================================

# SPSAで調整する対象パラメータとその探索範囲
SPSA_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    # 優先度重み
    "alpha":       (0.0, 3.0),
    "beta":        (0.0, 2.0),
    "gamma":       (0.0, 1.0),
    "delta":       (0.0, 1.0),
    "epsilon":     (0.0, 2.0),
    "zeta":        (0.0, 2.0),
    # RootMoveStatsスコアリング（正規化するので合計!=1でもOK）
    "w_max":       (0.0, 2.0),   # 追加
    "w_mean":      (0.0, 2.0),   # 追加
    "w_count":     (0.0, 1.0),   # 追加
    "var_penalty": (0.0, 1.0),
    # 探索部uncertainty ミキシング重み（正規化するので合計!=1でもOK）
    "unc_w_var":   (0.0, 1.0),
    "unc_w_visit": (0.0, 1.0),
    "unc_w_range": (0.0, 1.0),
}

# パラメータごとの摂動スケール倍率（問題点の修正）
# スケールが大きいパラメータには大きいc_scaleを設定して勾配精度を均一化
SPSA_C_SCALE: Dict[str, float] = {
    "alpha":       2.0,   # range 0~3 → 大きめ
    "beta":        1.5,
    "gamma":       0.8,
    "delta":       0.8,
    "epsilon":     1.5,
    "zeta":        1.5,
    "w_max":       1.5,
    "w_mean":      1.5,
    "w_count":     0.8,
    "var_penalty": 0.8,
    "unc_w_var":   0.8,
    "unc_w_visit": 0.8,
    "unc_w_range": 0.8,
}

SPSA_INITIAL_PARAMS: Dict[str, float] = {
    "alpha":       1.0,
    "beta":        0.5,
    "gamma":       0.2,
    "delta":       0.3,
    "epsilon":     0.5,
    "zeta":        0.7,
    "w_max":       0.4,   # 追加
    "w_mean":      0.4,   # 追加
    "w_count":     0.1,   # 追加
    "var_penalty": 0.1,
    "unc_w_var":   0.5,
    "unc_w_visit": 0.3,
    "unc_w_range": 0.2,
}

# ===========================================================
# SPSATuner
# ===========================================================

class SPSATuner:
    """
    SPSA によるNPLSパラメータ最適化クラス。

    Parameters
    ----------
    initial_params : 初期パラメータ dict
    param_bounds   : 各パラメータの (min, max)
    a              : 学習率スケール（大きいほど更新幅が大きい）
    c              : 摂動スケール（大きいほど探索幅が広い）
    alpha          : 学習率の減衰指数（推奨：0.602）
    gamma          : 摂動の減衰指数（推奨：0.101）
    A              : 安定化定数（総イテレーション数の約10%が目安）
    """

    def __init__(
        self,
        initial_params: Dict[str, float] = None,
        param_bounds:   Dict[str, Tuple[float, float]] = None,
        a:     float = 0.02,
        c:     float = 0.05,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A:     float = 20.0,
    ):
        self.params    = copy.deepcopy(initial_params or SPSA_INITIAL_PARAMS)
        self.bounds    = param_bounds or SPSA_PARAM_BOUNDS
        self.a         = a
        self.c         = c
        self.alpha     = alpha
        self.gamma     = gamma
        self.A         = A
        self.iteration = 0
        self.history:  List[Dict] = []

        # 収束モニタリング用
        self._win_plus_history:  List[float] = []
        self._win_minus_history: List[float] = []

    # ----------------------------------------------------------
    # 内部計算
    # ----------------------------------------------------------

    def _lr(self) -> float:
        """a_k = a / (k + 1 + A)^alpha"""
        return self.a / (self.iteration + 1 + self.A) ** self.alpha

    def _pert(self) -> float:
        """c_k = c / (k + 1)^gamma"""
        return self.c / (self.iteration + 1) ** self.gamma

    def _clip(self, name: str, value: float) -> float:
        lo, hi = self.bounds[name]
        return max(lo, min(hi, value))

    def _bernoulli_delta(self, c_k: float) -> Dict[str, float]:
        """
        各パラメータを ±(c_k × c_scale) でランダム摂動（ベルヌーイ分布）。
        c_scaleによってパラメータごとに摂動幅を調整し、勾配精度を均一化する。
        """
        c_scale = SPSA_C_SCALE  # モジュールレベル定数を参照
        return {
            name: c_k * c_scale.get(name, 1.0) * random.choice([-1.0, 1.0])
            for name in self.params
        }

    def _add_delta(
        self,
        base:  Dict[str, float],
        delta: Dict[str, float],
        sign:  float,
    ) -> Dict[str, float]:
        return {
            name: self._clip(name, base[name] + sign * delta[name])
            for name in base
        }

    # ----------------------------------------------------------
    # 1イテレーション
    # ----------------------------------------------------------

    def step(
        self,
        evaluate_fn: Callable[
            [Dict[str, float], Dict[str, float]],
            Tuple[int, int]
        ],
    ) -> Dict[str, float]:
        """
        1イテレーション実行。

        Parameters
        ----------
        evaluate_fn :
            (params_plus, params_minus) を受け取り
            (wins_plus, wins_minus) を返す関数。
            wins は 0〜games_per_iter の整数。

        Returns
        -------
        更新後の params dict
        """
        self.iteration += 1
        a_k = self._lr()
        c_k = self._pert()

        delta        = self._bernoulli_delta(c_k)
        params_plus  = self._add_delta(self.params, delta, +1.0)
        params_minus = self._add_delta(self.params, delta, -1.0)

        wins_plus, wins_minus = evaluate_fn(params_plus, params_minus)
        self._win_plus_history.append(wins_plus)
        self._win_minus_history.append(wins_minus)

        # 損失差：plus が勝てば loss_diff < 0（良い方向）
        # 最大化問題として扱うため wins を正の目的関数と定義
        # g_k ≈ (wins_minus - wins_plus) / (2 * delta)  ← 最大化の勾配
        grad: Dict[str, float] = {}
        for name in self.params:
            d = delta[name]
            if abs(d) > 1e-12:
                # 最大化：wins_plus > wins_minus なら theta を delta 方向に動かす
                grad[name] = (wins_minus - wins_plus) / (2.0 * d)

        for name in self.params:
            if name in grad:
                self.params[name] = self._clip(
                    name,
                    self.params[name] - a_k * grad[name],
                )

        record = {
            "iteration":   self.iteration,
            "a_k":         a_k,
            "c_k":         c_k,
            "params":      copy.deepcopy(self.params),
            "wins_plus":   wins_plus,
            "wins_minus":  wins_minus,
        }
        self.history.append(record)

        return copy.deepcopy(self.params)

    # ----------------------------------------------------------
    # 収束診断
    # ----------------------------------------------------------

    def convergence_score(self, window: int = 20) -> Optional[float]:
        """
        直近 window イテレーションの wins_plus / wins_minus 差の移動平均。
        0に近いほど収束している。
        """
        if len(self._win_plus_history) < window:
            return None
        diffs = [
            abs(p - m)
            for p, m in zip(
                self._win_plus_history[-window:],
                self._win_minus_history[-window:],
            )
        ]
        return sum(diffs) / len(diffs)

    # ----------------------------------------------------------
    # 保存・読み込み
    # ----------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "iteration":   self.iteration,
            "best_params": self.params,
            "bounds":      self.bounds,
            "c_scale":     SPSA_C_SCALE,   # 追加
            "history":     self.history,
            "spsa_config": {
                "a": self.a, "c": self.c,
                "alpha": self.alpha, "gamma": self.gamma, "A": self.A,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[SPSA] 保存完了 → {path}  (iter={self.iteration})")

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.params    = data["best_params"]
        self.bounds    = data.get("bounds", self.bounds)
        self.history   = data.get("history", [])
        self.iteration = data.get("iteration", len(self.history))
        cfg = data.get("spsa_config", {})
        self.a     = cfg.get("a",     self.a)
        self.c     = cfg.get("c",     self.c)
        self.alpha = cfg.get("alpha", self.alpha)
        self.gamma = cfg.get("gamma", self.gamma)
        self.A     = cfg.get("A",     self.A)
        print(f"[SPSA] 読み込み完了 ← {path}  (iter={self.iteration})")

    def print_summary(self) -> None:
        print("\n=== SPSA 最適化結果 ===")
        for name, val in self.params.items():
            print(f"  {name:15s} = {val:.4f}")
        score = self.convergence_score()
        if score is not None:
            print(f"  収束スコア（低いほど良い）: {score:.3f}")
        print("======================\n")


# ===========================================================
# USIエンジン対局ヘルパー
# ===========================================================

class USIGameRunner:
    """
    2つのUSIエンジン（NPLSPlayer）を subprocess で動かして対局させる。
    NPLSPlayer を直接インポートして使う場合はこのクラスを使わずに
    直接 NPLSPlayer.go() を呼び出すことも可能。
    """

    def __init__(
        self,
        engine_cmd:     List[str],   # エンジン起動コマンド
        opponent_cmd:   List[str],   # 対戦相手起動コマンド
        time_limit_ms:  int = 3000,  # 持ち時間(ms)
        byoyomi_ms:     int = 1000,  # 秒読み(ms)
        max_moves:      int = 400,   # 最大手数（引き分け）
    ):
        self.engine_cmd    = engine_cmd
        self.opponent_cmd  = opponent_cmd
        self.time_limit_ms = time_limit_ms
        self.byoyomi_ms    = byoyomi_ms
        self.max_moves     = max_moves

    def _send(self, proc: subprocess.Popen, cmd: str) -> None:
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def _recv_until(
        self,
        proc: subprocess.Popen,
        keyword: str,
        timeout: float = 30.0,
    ) -> List[str]:
        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = proc.stdout.readline().strip()
            if line:
                lines.append(line)
                if keyword in line:
                    break
        return lines

    def _init_engine(self, proc: subprocess.Popen) -> None:
        self._send(proc, "usi")
        self._recv_until(proc, "usiok")
        self._send(proc, "isready")
        self._recv_until(proc, "readyok")

    def play_one_game(
        self,
        engine_params: Dict[str, float],
        engine_plays_black: bool = True,
    ) -> Optional[bool]:
        """
        1局対局する。

        Returns
        -------
        True  : engine が勝利
        False : engine が敗北
        None  : 引き分け / エラー
        """
        try:
            ep = subprocess.Popen(
                self.engine_cmd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1,
            )
            op = subprocess.Popen(
                self.opponent_cmd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1,
            )

            self._init_engine(ep)
            self._init_engine(op)

            # パラメータを setoption で渡す
            for name, val in engine_params.items():
                self._send(ep, f"setoption name {name} value {int(val * 100)}")

            self._send(ep, "usinewgame")
            self._send(op, "usinewgame")

            moves: List[str] = []
            board_sfen = "startpos"

            for move_num in range(self.max_moves):
                pos_cmd = f"position {board_sfen}"
                if moves:
                    pos_cmd += " moves " + " ".join(moves)

                # どちらが指す番か
                is_engine_turn = (
                    (move_num % 2 == 0) == engine_plays_black
                )
                current = ep if is_engine_turn else op

                self._send(current, pos_cmd)
                time_cmd = (
                    f"go byoyomi {self.byoyomi_ms}"
                )
                self._send(current, time_cmd)
                lines = self._recv_until(current, "bestmove", timeout=30.0)

                bestmove = None
                for line in lines:
                    if line.startswith("bestmove"):
                        parts = line.split()
                        bestmove = parts[1] if len(parts) > 1 else None
                        break

                if bestmove is None or bestmove == "resign":
                    # 投了 → 現在の手番のエンジンが負け
                    engine_lost = is_engine_turn
                    return not engine_lost

                moves.append(bestmove)

            return None  # 最大手数で引き分け

        except Exception as e:
            print(f"[USIGameRunner] エラー: {e}")
            return None
        finally:
            for proc in (ep, op):
                try:
                    self._send(proc, "quit")
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

    def play_n_games(
        self,
        engine_params: Dict[str, float],
        n: int,
    ) -> int:
        """n局対局してengineの勝利数を返す"""
        wins = 0
        for i in range(n):
            # 先後交互
            result = self.play_one_game(engine_params, engine_plays_black=(i % 2 == 0))
            if result is True:
                wins += 1
        return wins


# ===========================================================
# メインチューニングループ
# ===========================================================

def run_spsa_tuning(
    engine_cmd:      List[str],
    opponent_cmd:    List[str],
    games_per_iter:  int   = 4,
    total_iter:      int   = 200,
    output:          str   = "spsa_result.json",
    resume:          bool  = False,
    time_limit_ms:   int   = 3000,
    byoyomi_ms:      int   = 1000,
    a:               float = 0.02,
    c:               float = 0.05,
) -> Dict[str, float]:
    """
    SPSAチューニングのメインループ。

    Parameters
    ----------
    engine_cmd     : NPLSエンジンの起動コマンド（例：["python", "main.py", "--engine", "npls"]）
    opponent_cmd   : 対戦相手の起動コマンド
    games_per_iter : 1イテレーションあたりの対局数（plus / minus それぞれ）
    total_iter     : 総イテレーション数
    output         : 結果保存先JSONファイル
    resume         : Trueなら output から途中再開
    """

    tuner = SPSATuner(
        initial_params = SPSA_INITIAL_PARAMS,
        param_bounds   = SPSA_PARAM_BOUNDS,
        a     = a,
        c     = c,
        A     = total_iter * 0.1,
    )

    if resume and output:
        try:
            tuner.load(output)
            print(f"[SPSA] iter {tuner.iteration} から再開します")
        except FileNotFoundError:
            print("[SPSA] 保存ファイルなし。最初から開始します")

    runner = USIGameRunner(
        engine_cmd    = engine_cmd,
        opponent_cmd  = opponent_cmd,
        time_limit_ms = time_limit_ms,
        byoyomi_ms    = byoyomi_ms,
    )

    def evaluate_fn(
        params_plus:  Dict[str, float],
        params_minus: Dict[str, float],
    ) -> Tuple[int, int]:
        wins_p = runner.play_n_games(params_plus,  n=games_per_iter)
        wins_m = runner.play_n_games(params_minus, n=games_per_iter)
        print(
            f"[SPSA] iter {tuner.iteration}: "
            f"plus {wins_p}/{games_per_iter}, "
            f"minus {wins_m}/{games_per_iter}"
        )
        return wins_p, wins_m

    remaining = total_iter - tuner.iteration
    for i in range(remaining):
        params = tuner.step(evaluate_fn)

        if (i + 1) % 10 == 0:
            tuner.save(output)
            tuner.print_summary()

        score = tuner.convergence_score()
        if score is not None and score < 0.05:
            print(f"[SPSA] 収束検出（score={score:.4f}）。早期終了します。")
            break

    tuner.save(output)
    tuner.print_summary()
    return tuner.params


# ===========================================================
# 直接利用インターフェース（NPLSPlayerを直接使う場合）
# ===========================================================

def run_spsa_direct(
    player_factory:   Callable[[Dict[str, float]], object],
    evaluate_game_fn: Callable[[object, object], Optional[bool]],
    opponent_factory: Callable[[], object],
    games_per_iter:   int   = 4,
    total_iter:       int   = 200,
    output:           str   = "spsa_result.json",
    resume:           bool  = False,
) -> Dict[str, float]:
    """
    NPLSPlayer オブジェクトを直接生成して対局させる版。
    subprocess を使わないため高速。

    Parameters
    ----------
    player_factory   : params dict → NPLSPlayer を返す関数
    evaluate_game_fn : (player, opponent) → True/False/None を返す関数
    opponent_factory : () → 対戦相手オブジェクト を返す関数
    """

    tuner = SPSATuner(
        initial_params = SPSA_INITIAL_PARAMS,
        param_bounds   = SPSA_PARAM_BOUNDS,
        A = total_iter * 0.1,
    )

    if resume and output:
        try:
            tuner.load(output)
        except FileNotFoundError:
            pass

    def evaluate_fn(params_plus, params_minus):
        wins_p = 0
        wins_m = 0
        for i in range(games_per_iter):
            black_first = (i % 2 == 0)
            p = player_factory(params_plus)
            o = opponent_factory()
            r = evaluate_game_fn(p, o) if black_first else evaluate_game_fn(o, p)
            if r is True:
                wins_p += 1
            p = player_factory(params_minus)
            o = opponent_factory()
            r = evaluate_game_fn(p, o) if black_first else evaluate_game_fn(o, p)
            if r is True:
                wins_m += 1
        return wins_p, wins_m

    for i in range(total_iter - tuner.iteration):
        params = tuner.step(evaluate_fn)
        print(f"[SPSA] iter {tuner.iteration}/{total_iter}: {params}")
        if (i + 1) % 10 == 0:
            tuner.save(output)

    tuner.save(output)
    tuner.print_summary()
    return tuner.params


# ===========================================================
# CLI エントリポイント
# ===========================================================

def main():
    parser = argparse.ArgumentParser(description="UW-NPLS SPSA パラメータチューニング")
    parser.add_argument("--engine",        nargs="+", required=True,
                        help="NPLSエンジンの起動コマンド（例：python main.py --engine npls）")
    parser.add_argument("--opponent",      nargs="+", required=True,
                        help="対戦相手エンジンの起動コマンド")
    parser.add_argument("--games_per_iter", type=int, default=4,
                        help="1イテレーションあたりの対局数")
    parser.add_argument("--total_iter",    type=int, default=200,
                        help="総イテレーション数")
    parser.add_argument("--output",        type=str, default="spsa_result.json",
                        help="結果保存先JSONファイル")
    parser.add_argument("--resume",        action="store_true",
                        help="保存ファイルから途中再開")
    parser.add_argument("--time_limit_ms", type=int, default=3000,
                        help="持ち時間(ms)")
    parser.add_argument("--byoyomi_ms",    type=int, default=1000,
                        help="秒読み(ms)")
    parser.add_argument("--a",             type=float, default=0.02,
                        help="SPSA学習率スケール")
    parser.add_argument("--c",             type=float, default=0.05,
                        help="SPSA摂動スケール")
    args = parser.parse_args()

    run_spsa_tuning(
        engine_cmd      = args.engine,
        opponent_cmd    = args.opponent,
        games_per_iter  = args.games_per_iter,
        total_iter      = args.total_iter,
        output          = args.output,
        resume          = args.resume,
        time_limit_ms   = args.time_limit_ms,
        byoyomi_ms      = args.byoyomi_ms,
        a               = args.a,
        c               = args.c,
    )


if __name__ == "__main__":
    main()
