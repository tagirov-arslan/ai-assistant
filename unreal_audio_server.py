"""Dependency-free local WebSocket server for streaming PCM audio to Unreal."""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading
from typing import Any


_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class UnrealAudioServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = int(port)
        self._thread: threading.Thread | None = None
        self._listener: socket.socket | None = None
        self._clients: set[socket.socket] = set()
        self._clients_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._started = threading.Event()
        self._stop_requested = threading.Event()
        self._audio_format: tuple[int, int, int] | None = None
        self._audio_packets = 0
        self._audio_bytes = 0
        self._last_error = ""

    @property
    def available(self) -> bool:
        return True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.running:
            return True
        self._stop_requested.clear()
        self._started.clear()
        self._thread = threading.Thread(target=self._run, name="unreal-audio-ws", daemon=True)
        self._thread.start()
        self._started.wait(timeout=2.0)
        return self.running and self._listener is not None

    def stop(self) -> None:
        self._stop_requested.set()
        listener = self._listener
        self._listener = None
        if listener is not None:
            self._close_socket(listener)
        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            self._close_socket(client)

    def send_state(self, state: str) -> None:
        self._broadcast_json({"type": "state", "value": state})

    def send_event(self, event_type: str) -> None:
        self._broadcast_json({"type": event_type})

    def send_audio(self, pcm: bytes, sample_rate: int, channels: int, sample_width: int = 2) -> None:
        if not pcm:
            return
        if not self.running and not self.start():
            self._last_error = f"WebSocket server is not running on {self.host}:{self.port}"
            return
        audio_format = (int(sample_rate), int(channels), int(sample_width))
        if audio_format != self._audio_format:
            self._audio_format = audio_format
            self._broadcast_json({
                "type": "audio_format",
                "sample_rate": audio_format[0],
                "channels": audio_format[1],
                "sample_width": audio_format[2],
            })
        self._broadcast(bytes(pcm), opcode=0x2)
        self._audio_packets += 1
        self._audio_bytes += len(pcm)

    def get_stats(self) -> dict[str, int | str | bool]:
        with self._clients_lock:
            client_count = len(self._clients)
        return {
            "running": self.running,
            "clients": client_count,
            "audio_packets": self._audio_packets,
            "audio_bytes": self._audio_bytes,
            "last_error": self._last_error,
        }

    def _run(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.host, self.port))
            listener.listen(4)
            listener.settimeout(0.5)
            self._listener = listener
            self._started.set()
            while not self._stop_requested.is_set():
                try:
                    client, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=self._serve_client, args=(client,), daemon=True).start()
        except OSError as error:
            self._last_error = str(error)
            self._started.set()
        finally:
            self._listener = None
            self._close_socket(listener)

    def _serve_client(self, client: socket.socket) -> None:
        client.settimeout(3.0)
        try:
            request = self._read_http_request(client)
            headers = self._parse_headers(request)
            key = headers.get("sec-websocket-key")
            if not key:
                return
            accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()).decode("ascii")
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            )
            with self._clients_lock:
                self._clients.add(client)
            client.sendall(response.encode("ascii"))
            client.settimeout(0.5)
            if self._audio_format is not None:
                rate, channels, width = self._audio_format
                client.sendall(self._frame(json.dumps({
                    "type": "audio_format", "sample_rate": rate,
                    "channels": channels, "sample_width": width,
                }).encode("utf-8"), 0x1))
            while not self._stop_requested.is_set():
                try:
                    header = self._recv_exact(client, 2)
                except socket.timeout:
                    continue
                if not header:
                    break
                opcode = header[0] & 0x0F
                length = header[1] & 0x7F
                masked = bool(header[1] & 0x80)
                if length == 126:
                    length = struct.unpack("!H", self._recv_exact(client, 2))[0]
                elif length == 127:
                    length = struct.unpack("!Q", self._recv_exact(client, 8))[0]
                mask = self._recv_exact(client, 4) if masked else b""
                payload = self._recv_exact(client, length)
                if masked:
                    payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
                if opcode == 0x8:
                    break
                if opcode == 0x9:
                    client.sendall(self._frame(payload, 0xA))
        except (ConnectionError, OSError, ValueError, struct.error):
            pass
        finally:
            with self._clients_lock:
                self._clients.discard(client)
            self._close_socket(client)

    def _broadcast_json(self, payload: dict[str, Any]) -> None:
        self._broadcast(json.dumps(payload, ensure_ascii=False).encode("utf-8"), opcode=0x1)

    def _broadcast(self, payload: bytes, opcode: int) -> None:
        frame = self._frame(payload, opcode)
        dead: list[socket.socket] = []
        with self._send_lock:
            with self._clients_lock:
                clients = list(self._clients)
            for client in clients:
                try:
                    client.sendall(frame)
                except OSError as error:
                    self._last_error = str(error)
                    dead.append(client)
        if dead:
            with self._clients_lock:
                for client in dead:
                    self._clients.discard(client)
                    self._close_socket(client)

    @staticmethod
    def _frame(payload: bytes, opcode: int) -> bytes:
        first = 0x80 | opcode
        length = len(payload)
        if length < 126:
            return bytes((first, length)) + payload
        if length <= 0xFFFF:
            return bytes((first, 126)) + struct.pack("!H", length) + payload
        return bytes((first, 127)) + struct.pack("!Q", length) + payload

    @staticmethod
    def _read_http_request(client: socket.socket) -> bytes:
        data = bytearray()
        while b"\r\n\r\n" not in data and len(data) < 16384:
            chunk = client.recv(2048)
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    @staticmethod
    def _parse_headers(request: bytes) -> dict[str, str]:
        lines = request.decode("latin-1").split("\r\n")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()
        return headers

    @staticmethod
    def _recv_exact(client: socket.socket, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = client.recv(size - len(data))
            if not chunk:
                raise ConnectionError("WebSocket client disconnected")
            data.extend(chunk)
        return bytes(data)

    @staticmethod
    def _close_socket(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
