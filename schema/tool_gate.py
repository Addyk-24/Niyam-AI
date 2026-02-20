import jsonschema

class ToolAuthorityGate:
    def __init__(self,contracts):
        self.contracts = contracts
    
    def validate_tool(self, tool_name: str):

        try:
            if tool_name in self.contracts.forbidden_tools:
                raise Exception(f"Tool '{tool_name}' is explicitly forbidden")
            

            if tool_name not in self.contracts.allowed_tools:
                raise Exception(f"Tool '{tool_name}' is not allowed by intent")

        except Exception as e:
            print(f"Tool validation failed: {e}")
            return False

        print("Tool validation succeeded")
        return True
    

    # DEMO:
    def validate_schema(self, tool_name: str, payload: dict):
        
        if tool_name == "proceed_transaction":
            schema = {
                "type": "object",
                "properties": {
                    "amount": {"type":"number"},
                    "recipient": {"type": "string"},

                },
                "required": ["amount","recipient"],
            }
            jsonschema.validate(instance=payload,schema=schema)
        
        return True
    



