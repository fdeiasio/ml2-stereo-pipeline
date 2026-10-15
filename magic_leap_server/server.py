import socket
import threading

from .client import handle_client_connection
from .config import HOST, PORT, SOCKET_TIMEOUT


def run_server() -> None:
    print("Starting Magic Leap Server...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        server_socket.settimeout(SOCKET_TIMEOUT)
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            try:
                client_socket, client_address = server_socket.accept()
            except socket.timeout:
                continue
            
            client_thread = threading.Thread(
                target=handle_client_connection,
                args=(client_socket, client_address),
                daemon=True
            )
            client_thread.start()
