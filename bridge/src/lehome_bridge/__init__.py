"""Mac-local authenticated bridge for two SO101 leader arms."""

from .protocol import BridgeMessage, MessageVerifier, encode_message

__all__ = ["BridgeMessage", "MessageVerifier", "encode_message"]
