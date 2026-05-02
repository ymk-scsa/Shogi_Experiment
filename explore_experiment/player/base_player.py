import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, TypedDict
from typing_extensions import TypeAlias

# USIコマンドの戻り値の型
UsiResponse: TypeAlias = tuple[str, Optional[str]]


class SetLimitsArgDict(TypedDict, total=False):
    btime: Optional[int]
    wtime: Optional[int]
    byoyomi: Optional[int]
    binc: Optional[int]
    winc: Optional[int]
    nodes: Optional[int]
    infinite: bool
    ponder: bool


class BasePlayer:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future: Optional[Future[UsiResponse]] = None

    def _usi_debug(self, msg: str) -> None:
        print(f"[usi] {msg}", file=sys.stderr, flush=True)

    def _wait_pending_go_on_shutdown(self) -> None:
        """stdin 終了（パイプ）時に、非同期の go が未完了なら完了を待ち bestmove を出させる。"""
        fut = self.future
        if fut is None or fut.done():
            return
        self._usi_debug("stdin closed: waiting for background go to finish...")
        try:
            fut.result()
        except Exception as e:
            print(f"info string go thread error: {e}", flush=True)
            traceback.print_exc(file=sys.stderr)

    def usi(self) -> None:
        pass

    def usinewgame(self) -> None:
        pass

    def setoption(self, args: list[str]) -> None:
        pass

    def isready(self) -> None:
        pass

    def position(self, sfen: str, usi_moves: list[str]) -> None:
        pass

    def set_limits(
        self,
        btime: Optional[int] = None,
        wtime: Optional[int] = None,
        byoyomi: Optional[int] = None,
        binc: Optional[int] = None,
        winc: Optional[int] = None,
        nodes: Optional[int] = None,
        infinite: bool = False,
        ponder: bool = False,
    ) -> None:
        pass

    def go(self) -> UsiResponse:
        raise NotImplementedError

    def stop(self) -> None:
        pass

    def ponderhit(self, last_limits: SetLimitsArgDict) -> None:
        pass

    def quit(self) -> None:
        pass

    def run(self) -> None:
        while True:
            try:
                cmd_line = input()
            except EOFError:
                self._wait_pending_go_on_shutdown()
                break

            cmd_line = cmd_line.strip()
            if not cmd_line:
                continue

            cmd = cmd_line.split(" ", 1)
            if not cmd:
                continue

            if cmd[0] == "usi":
                self.usi()
                print("usiok", flush=True)
            elif cmd[0] == "setoption":
                option = cmd[1].split(" ")
                self.setoption(option)
            elif cmd[0] == "isready":
                self.isready()
                print("readyok", flush=True)
            elif cmd[0] == "usinewgame":
                self.usinewgame()
            elif cmd[0] == "position":
                if len(cmd) < 2:
                    self._usi_debug("ignored: position with no arguments")
                    continue
                args = cmd[1].split("moves")
                self.position(args[0].strip(), args[1].split() if len(args) > 1 else [])
            elif cmd[0] == "go":
                kwargs: SetLimitsArgDict = {}
                if len(cmd) > 1:
                    args = cmd[1].split(" ")
                    if args[0] == "infinite":
                        kwargs["infinite"] = True
                    else:
                        if args[0] == "ponder":
                            kwargs["ponder"] = True
                            args = args[1:]
                        for i in range(0, len(args) - 1, 2):
                            key = args[i]
                            if key == "btime":
                                kwargs["btime"] = int(args[i + 1])
                            elif key == "wtime":
                                kwargs["wtime"] = int(args[i + 1])
                            elif key == "byoyomi":
                                kwargs["byoyomi"] = int(args[i + 1])
                            elif key == "binc":
                                kwargs["binc"] = int(args[i + 1])
                            elif key == "winc":
                                kwargs["winc"] = int(args[i + 1])
                            elif key == "nodes":
                                kwargs["nodes"] = int(args[i + 1])
                self.set_limits(**kwargs)
                # ponderhitのために条件と経過時間を保存
                last_limits = kwargs
                need_print_bestmove = "ponder" not in kwargs and "infinite" not in kwargs

                def go_and_print_bestmove() -> UsiResponse:
                    bestmove, ponder_move = self.go()
                    if need_print_bestmove:
                        print("bestmove " + bestmove + (" ponder " + ponder_move if ponder_move else ""), flush=True)
                    return bestmove, ponder_move

                # CUDA 推論は isready と同じスレッド（メイン）で実行する。
                # 別スレッドだとドライバ次第でセグフォや不正終了することがある。
                use_background_go = kwargs.get("infinite", False) or kwargs.get("ponder", False)
                if use_background_go:
                    self.future = self.executor.submit(go_and_print_bestmove)
                else:
                    self.future = None
                    try:
                        go_and_print_bestmove()
                    except Exception as e:
                        print(f"info string go error: {e}", flush=True)
                        traceback.print_exc(file=sys.stderr)
            elif cmd[0] == "stop":
                need_print_bestmove = False
                self.stop()
                if self.future is None:
                    self._usi_debug("stop: no active go")
                    continue
                bestmove, _ = self.future.result()
                print("bestmove " + bestmove, flush=True)
            elif cmd[0] == "ponderhit":
                if self.future is None:
                    self._usi_debug("ponderhit: no active go")
                    continue
                last_limits["ponder"] = False
                self.ponderhit(last_limits)
                bestmove, ponder_move = self.future.result()
                print("bestmove " + bestmove + (" ponder " + ponder_move if ponder_move else ""), flush=True)
            elif cmd[0] == "quit":
                self.quit()
                break
