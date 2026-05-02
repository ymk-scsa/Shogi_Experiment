import argparse
import os
import signal
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


def handle_client(
    client: socket.socket,
    engine_command: str,
    token: Optional[str],
    trace_io: bool,
    shutdown_event: threading.Event,
) -> None:
    popen_kwargs = dict(
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    if os.name == "nt":
        # Ctrl+C が親サーバに届いたとき、子プロセス群を確実に停止しやすくする
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    engine = subprocess.Popen(
        engine_command,
        **popen_kwargs,
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
                msg = out.rstrip("\r\n")
                if trace_io:
                    print(f"[server][engine->client] {msg}", flush=True)
                _send_line(client, msg)
        except Exception:
            pass

    def forward_engine_stderr() -> None:
        try:
            for err in engine.stderr:
                if stop_event.is_set():
                    break
                msg = err.rstrip("\r\n")
                if msg:
                    if trace_io:
                        print(f"[server][engine:stderr] {msg}", flush=True)
                    _send_line(client, f"info string remote stderr: {msg}")
        except Exception:
            pass

    t_out = threading.Thread(target=forward_engine_stdout, daemon=True)
    t_err = threading.Thread(target=forward_engine_stderr, daemon=True)
    t_out.start()
    t_err.start()

    try:
        if token is not None:
            if trace_io:
                print("[server][auth] required", flush=True)
            _send_line(client, "info string remote auth required")
        for line in _recv_lines(client):
            if shutdown_event.is_set():
                break
            if trace_io:
                print(f"[server][client->server] {line}", flush=True)
            if not authed:
                if line == f"AUTH {token}":
                    authed = True
                    if trace_io:
                        print("[server][auth] ok", flush=True)
                    _send_line(client, "info string remote auth ok")
                else:
                    if trace_io:
                        print("[server][auth] failed", flush=True)
                    _send_line(client, "info string remote auth failed")
                    break
                continue

            if trace_io:
                print(f"[server][server->engine] {line}", flush=True)
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
            if engine.poll() is None:
                engine.terminate()
                engine.wait(timeout=2.0)
        except Exception:
            try:
                engine.kill()
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
    parser.add_argument("--trace-io", action="store_true", help="Print line-level relay logs")
    args = parser.parse_args()

    shutdown_event = threading.Event()

    def _request_shutdown(*_args) -> None:
        if not shutdown_event.is_set():
            print("[server] shutdown requested", flush=True)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_shutdown)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        server.settimeout(1.0)
        print(f"[server] listening on {args.host}:{args.port}", flush=True)
        while not shutdown_event.is_set():
            try:
                client, addr = server.accept()
            except socket.timeout:
                continue
            print(f"[server] connected: {addr}", flush=True)
            try:
                handle_client(client, args.engine_command, args.token, args.trace_io, shutdown_event)
            except Exception as e:
                print(f"[server] client error: {e}", flush=True)
            print("[server] disconnected", flush=True)
        print("[server] shutdown complete", flush=True)


if __name__ == "__main__":
    main()
