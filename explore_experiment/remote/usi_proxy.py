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


# ★ 追加: 常時出力スレッド
def _writer(out_queue: queue.Queue[str], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            line = out_queue.get(timeout=0.1)
            print(line, flush=True)
        except queue.Empty:
            continue


def connect_with_retry(host: str, port: int, retry_ms: int) -> socket.socket:
    while True:
        try:
            print(f"info string proxy connecting to {host}:{port}", flush=True)
            s = socket.create_connection((host, port), timeout=5.0)
            s.settimeout(None)
            print(f"info string proxy connected to {host}:{port}", flush=True)
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

    print(f"info string proxy starting with host {args.host}, port {args.port}, retry_ms {args.retry_ms}", flush=True)
    sock = connect_with_retry(args.host, args.port, args.retry_ms)
    if args.token:
        _send_line(sock, f"AUTH {args.token}")
    print("info string proxy connected", flush=True)

    out_queue: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()

    # ★ readerスレッド
    t_reader = threading.Thread(target=_reader, args=(sock, out_queue, stop_event), daemon=True)
    t_reader.start()

    # ★ 常時出力スレッド（これが修正の核心）
    t_writer = threading.Thread(target=_writer, args=(out_queue, stop_event), daemon=True)
    t_writer.start()

    try:
        while True:
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

    finally:
        stop_event.set()
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
