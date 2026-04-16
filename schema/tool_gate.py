import jsonschema
from schema.intent_contract import IntentContract


class ToolAuthorizationError(Exception):
    """Raised when a tool call violates the intent contract."""
    pass

class ToolAuthorityGate:
    def __init__(self,contract:IntentContract):
        self.contract = contract
    
    def validate_tool(self, tool_name: str):

        """
        Args: tool_name: string
        Returns: bool : True if tool is permitted
        """

        
        if tool_name in self.contracts.forbidden_tools:
            raise ToolAuthorizationError(
                f"Tool '{tool_name}' is explicitly forbidden by intent contract."
            )            

        if tool_name not in self.contracts.allowed_tools:
            raise ToolAuthorizationError(
                f"Tool '{tool_name}' is not in the allowed list. "
                f"Allowed: {self.contract.allowed_tools}"
            )


        print("Tool validation succeeded")
        return True
    

    # DEMO:
    def validate_schema(self, tool_name: str, payload: dict):
        """Validate the payload shape for known tools."""

        schemas = {
            "proceed_transaction": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "recipient": {"type": "string"},
                },
                "required": ["amount", "recipient"],
                "additionalProperties": False,
            }
        }
        if tool_name in schemas:
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
    



