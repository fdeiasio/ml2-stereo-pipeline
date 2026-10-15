import socket


"""
this file is a simple test to test if the server is sending packets after computing the pose
you can also use this as a blueprint for any other application that wished to receive
the datas streamed by the server.
"""


LOCALHOST = '127.0.0.1'
PORT = 12345

socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
socket.bind((LOCALHOST, PORT))
print(f"Server listening on {LOCALHOST}:{PORT}")
while True:
    try:
        data, addr = socket.recvfrom(2048)
        print(f"Received message from {addr}: {data.decode()}")
    except Exception as e:
        print(f"Error receiving message: {e}")