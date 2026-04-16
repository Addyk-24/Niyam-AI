

class ControlFlowViolation(Exception):
    pass


class ControlFlowIntegrity:
    def __init__(self,allowed_sequence: list[str]):

        self.allowed_sequence = allowed_sequence
        self.current_index = 0
        
    def validate_step(self,action:str):

        if self.current_index >= len(self.allowed_sequence):
            raise ControlFlowViolation("No further actions allowed — sequence exhausted.")
        
        expected = self.allowed_sequence[self.current_index]

        if action != expected:
            raise ControlFlowViolation(
                f"Step {self.current_index}: expected '{expected}' but got '{action}'"
            )
          
        self.current_index += 1
        return True
    def reset(self):
        """Allow reuse of the same flow for a new session."""
        self.current_index = 0

    def is_complete(self) -> bool:
        return self.current_index >= len(self.allowed_sequence)

        