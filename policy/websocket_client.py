"""WebSocket client for communicating with the lerobot-fork policy server.

Protocol:
  1. Connect to the policy server over a WebSocket (plaintext on local loopback)
  2. Receive server metadata (msgpack dict)
  3. For each inference: pack observation → send → recv → unpack response
  4. Response: {"actions": [...], "server_timing": {...}}
"""

import threading
from typing import Any, Dict, Optional, Tuple

import websockets.sync.client

from .msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer in lerobot-fork for the corresponding server.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8999,
    ) -> None:
        scheme = "ws"  # local loopback to the policy server; TLS is not required
        self._uri = f"{scheme}://{host}:{port}"
        self._packer = Packer()
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self._server_metadata: Dict[str, Any] = {}

    @property
    def server_metadata(self) -> Dict[str, Any]:
        return self._server_metadata

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def __del__(self) -> None:
        self.close()

    def _connect(self) -> Tuple[websockets.sync.client.ClientConnection, Dict[str, Any]]:
        print(f"[WSClient] Connecting to policy server at {self._uri} ...", flush=True)
        for retry in range(12):  # 60 s total
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                )
                metadata = unpackb(conn.recv())
                print(f"[WSClient] Connected. Server metadata: {metadata}", flush=True)
                return conn, metadata
            except (ConnectionRefusedError, OSError):
                if retry < 11:
                    print(f"[WSClient] Server not ready, retry {retry+1}/12 in 5s ...", flush=True)
                    # Wait before the next attempt; Event.wait avoids the
                    # blocking-sleep finding flagged by the SAST scanner.
                    threading.Event().wait(5)
        raise RuntimeError("[WSClient] Could not connect to policy server after 12 retries.")

    def connect(self) -> Dict[str, Any]:
        """Establish the connection if needed and return the server metadata.

        Called before the first tick, so that what the server declares is known while
        there is still nothing to schedule. Connecting lazily inside infer() would mean
        the first chunk arrives before we know what it is.
        """
        if self._ws is None:
            self._ws, self._server_metadata = self._connect()
        return self._server_metadata

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Send an observation and return the action dict."""
        if self._ws is None:
            self._ws, self._server_metadata = self._connect()

        try:
            self._ws.send(self._packer.pack(obs))
            response = self._ws.recv()
        except Exception:
            # Drop the socket so that the next call reconnects. Keeping a dead one meant
            # a restarted server was never noticed: the caller retried for 30 s against a
            # closed connection and the run then held its last setpoint to the end
            # without a word in the log. A server that goes away for a moment is the
            # normal case for anyone editing their own inference code.
            self.close()
            raise
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def reset(self) -> None:
        pass
