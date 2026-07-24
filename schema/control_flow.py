import hashlib
import time


class ControlFlowViolation(Exception):
    pass


class ControlFlowIntegrity:
    """
    SECURITY FIX (red-team finding): the original design
    allowed any caller to instantiate a fresh ControlFlowIntegrity object
    and silently reset sequence-progress state, because the object had no
    binding to the session's sealed IntentHash. An attacker (or buggy
    calling code) could bypass "already completed" sequence protection by
    simply constructing a new flow instance mid-session.

    Fix: every ControlFlowIntegrity instance is now bound to the session's
    IntentHash at construction. A process-wide registry tracks which
    IntentHash values have already had a flow instance created; a second
    attempt to create a NEW flow object for an IntentHash that already has
    one raises ControlFlowViolation instead of silently succeeding. This
    closes the gap without requiring every caller to manually enforce
    one-flow-per-session discipline.
    """

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
                    f"IntentHash {intent_hash[:16]}... — refusing to create "
                    f"a second one, which would silently reset sequence "
                    f"progress for an active session. If this is intentional "
                    f"(e.g. session explicitly ended and restarted), pass "
                    f"allow_rebind=True."
                )
            ControlFlowIntegrity._active_sessions[intent_hash] = time.time()

    def validate_step(self, action: str) -> bool:
        if self.current_index >= len(self.allowed_sequence):
            raise ControlFlowViolation("No further actions allowed — sequence exhausted.")

        expected = self.allowed_sequence[self.current_index]

        if action != expected:
            raise ControlFlowViolation(
                f"Step {self.current_index}: expected '{expected}' but got '{action}'"
            )

        self.current_index += 1
        return True

    def reset(self, allow_rebind: bool = False):
        """
        Allow reuse of the same flow for a new session.
        Explicit allow_rebind required if this instance is bound to an
        IntentHash — resetting a BOUND flow without acknowledging it is
        the exact silent-reset pattern the red-team finding identified.
        """
        if self.intent_hash is not None and not allow_rebind:
            raise ControlFlowViolation(
                "Cannot reset a session-bound flow without allow_rebind=True. "
                "This is intentional — silent resets of a sealed session's "
                "control flow are exactly the vulnerability this binding "
                "closes."
            )
        self.current_index = 0

    def is_complete(self) -> bool:
        return self.current_index >= len(self.allowed_sequence)

    @classmethod
    def clear_session_registry(cls):
        """Testing/teardown helper only — never call in production code
        paths, as it defeats the entire binding mechanism."""
        cls._active_sessions.clear()