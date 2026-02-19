

class ControlFlowIntegrity:
    def __init__(self,allowed_sequence):

        self.allowed_sequence = allowed_sequence
        self.current_index = 0
        
    def validate_step(self,action:str):
        if self.current_index >= len(self.allowed_sequence):
            raise Exception("No further actions allowed")
        
        expected = self.allowed_sequence[self.current_index]

        if action != expected:
            raise Exception(f"Expected action {expected} but got {action}")
        
        self.current_index += 1
        return True


        