"""
session-bound sequence enforcement.
"""

import time


class ControlFlowViolation(Exception):
    pass


class ControlFlowIntegrity:

    # Process-wide registry: intent_hash -> creation timestamp.
    _active_sessions: dict[str, float] = {}

    def __init__(self, allowed_sequence: list[str], intent_hash: str | None = None,
                 allow_rebind: bool = False):
        self.allowed_sequence = allowed_sequence
        self.current_index = 0
        self.intent_hash = intent_hash

        if intent_hash is not None:
            if intent_hash in ControlFlowIntegrity._active_sessions and not allow_rebind:
                raise ControlFlowViolation(
                    f"A ControlFlowIntegrity instance already exists for "
                    f"IntentHash {intent_hash[:16]}... - refusing to create a "
                    f"second one, which would silently reset sequence progress "
                    f"for an active session. Call end_session() when the "
                    f"session terminates, or pass allow_rebind=True if the "
                    f"reset is intentional."
                )
            ControlFlowIntegrity._active_sessions[intent_hash] = time.time()

    def validate_step(self, action: str) -> bool:
        if self.current_index >= len(self.allowed_sequence):
            raise ControlFlowViolation("No further actions allowed - sequence exhausted.")

        expected = self.allowed_sequence[self.current_index]
        if action != expected:
            raise ControlFlowViolation(
                f"Step {self.current_index}: expected '{expected}' but got '{action}'"
            )

        self.current_index += 1
        return True

    def end_session(self) -> None:
        """
        Release this session's IntentHash from the registry
        """
        if self.intent_hash:
            ControlFlowIntegrity._active_sessions.pop(self.intent_hash, None)

    def reset(self, allow_rebind: bool = False) -> None:
        """
        Reset sequence progress for reuse.\
        Explicit allow_rebind is required when this instance is bound to an
        IntentHash
        """
        if self.intent_hash is not None and not allow_rebind:
            raise ControlFlowViolation(
                "Cannot reset a session-bound flow without allow_rebind=True. "
                "This is intentional - silent resets of a sealed session's "
                "control flow are the vulnerability this binding closes."
            )
        self.current_index = 0

    def is_complete(self) -> bool:
        return self.current_index >= len(self.allowed_sequence)

    @classmethod
    def clear_session_registry(cls) -> None:
        """
        Testing and teardown only. Never call from a production path: it
        releases every active binding at once and defeats the mechanism.
        """
        cls._active_sessions.clear()