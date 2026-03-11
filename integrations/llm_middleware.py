
class LlmMiddleware:
    def __init__(self, contract,cfi,tool_gate,ledger):
        self.contract = contract
        self.cfi = cfi
        self.tool_gate = tool_gate
        self.ledger = ledger
    
    def execute_tool(self,tool):
        pass
            