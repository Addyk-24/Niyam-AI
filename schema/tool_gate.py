import jsonschema
from schema.intent_contract import IntentContract


class ToolAuthorizationError(Exception):
    """Raised when a tool call violates the intent contract."""
    pass


class ToolAuthorityGate:
    def __init__(self, contract: IntentContract):
        self.contract = contract

    def validate_tool(self, tool_name: str) -> bool:
        """
        Returns True if tool is permitted.
        BUG FIX: original silently returned False on exception — exceptions swallowed
                 security violations. Now raises ToolAuthorizationError so callers
                 cannot accidentally ignore a blocked action.
        """
        if tool_name in self.contract.forbidden_tools:
            raise ToolAuthorizationError(
                f"Tool '{tool_name}' is explicitly forbidden by intent contract."
            )

        if tool_name not in self.contract.allowed_tools:
            raise ToolAuthorizationError(
                f"Tool '{tool_name}' is not in the allowed list. "
                f"Allowed: {self.contract.allowed_tools}"
            )

        return True

    def validate_schema(self, tool_name: str, payload: dict) -> bool:
        """
        Validate the payload shape for known tools.

        SECURITY FIX (red-team finding): the original
        jsonschema numberr type accepts NaN and Infinity (json
        module round-trips them, and jsonschema's default number check
        does not reject them), and had no upper bound on string length
        or byte-level control characters. This allowed three confirmed
        bypasses: nan_amount, oversized_recipient (1MB string), and a
        null-byte-embedded recipient field reaching the Judge model
        unfiltered. All three are now rejected before the Judge is invoked.
        """
        schemas = {
            "proceed_transaction": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1e15,
                    },
                    "recipient": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 256,    # reject oversized-payload DoS attempts
                        # Reject embedded control chars (null byte, zero-width
                        # space) at the schema layer, not just in PayloadInspector
                        "pattern": r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\u200b]*$",
                    },
                },
                "required": ["amount", "recipient"],
                "additionalProperties": False,
            }
        }

        if tool_name in schemas:
            amount = payload.get("amount")
            if isinstance(amount, float):
                import math
                if math.isnan(amount) or math.isinf(amount):
                    raise ToolAuthorizationError(
                        f"Rejected non-finite amount value: {amount!r}"
                    )

            jsonschema.validate(instance=payload, schema=schemas[tool_name])

        return True

    def authorize(self, tool_name: str, payload: dict) -> bool:
        """
        Full authorization: tool allowed + payload valid.
        Call this single method from the middleware — it does both checks.
        """
        self.validate_tool(tool_name)
        self.validate_schema(tool_name, payload)
        return True