import argparse
import queue
import socket
import sys
import threading
from typing import Optional


def _send_line(sock: socket.socket, line: str) -> None:
    sock.sendall((line + "\n").encode("utf-8"))


def _reader(sock: socket.socket, out_queue: queue.Queue[str], stop_event: threading.Event) -> None:
    f = sock.makefile("r", encoding="utf-8", newline="\n")
    try:
        while not stop_event.is_set():
            line = f.readline()
            if line == "":
                out_queue.put("info string proxy disconnected from remote")
                break
            out_queue.put(line.rstrip("\r\n"))
    except Exception as e:
        out_queue.put(f"info string proxy reader error: {e}")


def connect_with_retry(host: str, port: int, retry_ms: int) -> socket.socket:
    while True:
        try:
            s = socket.create_connection((host, port), timeout=5.0)
            s.settimeout(None)
            return s
        except OSError as e:
            print(f"info string proxy connect failed: {e}", flush=True)
            threading.Event().wait(retry_ms / 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="USI proxy engine client")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=49001)
    parser.add_argument("--token", default=None, help="Optional auth token")
    parser.add_argument("--retry-ms", type=int, default=1000)
    args = parser.parse_args()

    sock = connect_with_retry(args.host, args.port, args.retry_ms)
    if args.token:
        _send_line(sock, f"AUTH {args.token}")
    print("info string proxy connected", flush=True)

    out_queue: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()
    t = threading.Thread(target=_reader, args=(sock, out_queue, stop_event), daemon=True)
    t.start()

    def flush_remote_output(non_block: bool) -> None:
        while True:
            try:
                line = out_queue.get_nowait() if non_block else out_queue.get(timeout=5.0)
                print(line, flush=True)
            except queue.Empty:
                break

    try:
        while True:
            flush_remote_output(non_block=True)

            cmd = sys.stdin.readline()
            if cmd == "":
                break
            line = cmd.rstrip("\r\n")
            if not line:
                continue

            try:
                _send_line(sock, line)
            except OSError:
                print("info string proxy reconnecting...", flush=True)
                sock.close()
                sock = connect_with_retry(args.host, args.port, args.retry_ms)
                if args.token:
                    _send_line(sock, f"AUTH {args.token}")
                _send_line(sock, line)

            if line == "quit":
                break

            # isready は readyok を受け取るまで待つと GUI 側の応答が安定する
            if line == "isready":
                while True:
                    x = out_queue.get()
                    print(x, flush=True)
                    if x == "readyok":
                        break
    finally:
        stop_event.set()
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
