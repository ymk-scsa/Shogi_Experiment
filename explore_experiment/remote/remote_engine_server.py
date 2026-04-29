import argparse
import socket
import subprocess
import threading
from typing import Optional


def _recv_lines(sock: socket.socket):
    f = sock.makefile("r", encoding="utf-8", newline="\n")
    while True:
        line = f.readline()
        if line == "":
            break
        yield line.rstrip("\r\n")


def _send_line(sock: socket.socket, line: str) -> None:
    sock.sendall((line + "\n").encode("utf-8"))


def handle_client(client: socket.socket, engine_command: str, token: Optional[str]) -> None:
    engine = subprocess.Popen(
        engine_command,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    assert engine.stdin is not None
    assert engine.stdout is not None
    assert engine.stderr is not None

    stop_event = threading.Event()
    authed = token is None

    def forward_engine_stdout() -> None:
        try:
            for out in engine.stdout:
                if stop_event.is_set():
                    break
                _send_line(client, out.rstrip("\r\n"))
        except Exception:
            pass

    def forward_engine_stderr() -> None:
        try:
            for err in engine.stderr:
                if stop_event.is_set():
                    break
                msg = err.rstrip("\r\n")
                if msg:
                    _send_line(client, f"info string remote stderr: {msg}")
        except Exception:
            pass

    t_out = threading.Thread(target=forward_engine_stdout, daemon=True)
    t_err = threading.Thread(target=forward_engine_stderr, daemon=True)
    t_out.start()
    t_err.start()

    try:
        if token is not None:
            _send_line(client, "info string remote auth required")
        for line in _recv_lines(client):
            if not authed:
                if line == f"AUTH {token}":
                    authed = True
                    _send_line(client, "info string remote auth ok")
                else:
                    _send_line(client, "info string remote auth failed")
                    break
                continue

            engine.stdin.write(line + "\n")
            engine.stdin.flush()
            if line == "quit":
                break
    finally:
        stop_event.set()
        try:
            if engine.poll() is None:
                engine.stdin.write("quit\n")
                engine.stdin.flush()
        except Exception:
            pass
        try:
            engine.terminate()
        except Exception:
            pass
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote USI engine server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=49001)
    parser.add_argument(
        "--engine-command",
        required=True,
        help='Example: "python player/mcts_player.py"',
    )
    parser.add_argument("--token", default=None, help="Optional simple shared token")
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"[server] listening on {args.host}:{args.port}", flush=True)
        while True:
            client, addr = server.accept()
            print(f"[server] connected: {addr}", flush=True)
            try:
                handle_client(client, args.engine_command, args.token)
            except Exception as e:
                print(f"[server] client error: {e}", flush=True)
            print("[server] disconnected", flush=True)


if __name__ == "__main__":
    main()
