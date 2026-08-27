import os
import sys
import argparse
import socket
import uvicorn

# 한국어 Windows 콘솔(cp949)에서 이모지 출력 시 UnicodeEncodeError가 나므로 UTF-8로 고정한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

def get_local_ip():
    """다른 랩탑에서 접속할 때 쓸 이 PC의 LAN IP. 실패 시 localhost."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "localhost"

server_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server")
sys.path.insert(0, server_dir)

from app import app
from ssl_helper import generate_self_signed_cert

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--no-ssl", action="store_true", help="HTTP 모드로 실행")
    args = parser.parse_args()

    use_ssl = not args.no_ssl
    protocol = "https" if use_ssl else "http"

    cert_path, key_path = None, None
    if use_ssl:
        cert_path, key_path = generate_self_signed_cert(server_dir)

    lan_ip = get_local_ip()

    print("=" * 70)
    print(f"🥊 [HOST] 4-Player AR Shadow Boxing & Battle Arena ({protocol.upper()} 모드)")
    print("=" * 70)
    print(f"[1] Host 대형 스크린 3D 링 주소: {protocol}://localhost:{args.port}/arena")
    print(f"[2] 4인 파이터 웹캠 접속 주소 (다른 랩탑 브라우저 접속):")
    print(f"    - Fighter 1 (Red)   : {protocol}://{lan_ip}:{args.port}/client?id=client_1")
    print(f"    - Fighter 2 (Cyan)  : {protocol}://{lan_ip}:{args.port}/client?id=client_2")
    print(f"    - Fighter 3 (Gold)  : {protocol}://{lan_ip}:{args.port}/client?id=client_3")
    print(f"    - Fighter 4 (Green) : {protocol}://{lan_ip}:{args.port}/client?id=client_4")
    print("=" * 70)

    if use_ssl:
        uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=cert_path, ssl_keyfile=key_path, timeout_graceful_shutdown=0)
    else:
        uvicorn.run(app, host="0.0.0.0", port=args.port, timeout_graceful_shutdown=0)

