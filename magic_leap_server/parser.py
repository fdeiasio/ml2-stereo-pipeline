import json
import socket
import struct
from typing import Any

from .config import EXPECTED_FRAME_COUNT, MAGIC_HEADER


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    """Read exactly `size` bytes from socket or raise if stream ends early."""
    chunks: list[bytes] = []
    remaining = size

    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Connection closed while reading packet")
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def _read_int32_le(sock: socket.socket) -> int:
    """Read a little-endian int32 value from socket."""
    return struct.unpack("<i", _recv_exact(sock, 4))[0]


def parse_wcam_packet(sock: socket.socket) -> list[tuple[dict[str, Any], bytes]]:
    """Parse one WCAM packet from a stream socket."""
    magic = _recv_exact(sock, 4)
    if magic != MAGIC_HEADER:
        raise ValueError(f"Invalid magic header: {magic!r}")

    frames: list[tuple[dict[str, Any], bytes]] = []
    for _ in range(EXPECTED_FRAME_COUNT):
        metadata_length = _read_int32_le(sock)

        if metadata_length < 0:
            raise ValueError(f"Invalid metadata length: {metadata_length}")

        metadata_raw = _recv_exact(sock, metadata_length)
        try:
            metadata = json.loads(metadata_raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError("Failed to decode metadata JSON") from exc

        image_length = _read_int32_le(sock)
        if image_length < 0:
            raise ValueError(f"Invalid image length: {image_length}")

        image_raw = _recv_exact(sock, image_length)
        frames.append((metadata, image_raw))

    return frames
