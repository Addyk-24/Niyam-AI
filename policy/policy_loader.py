import yaml

class PolicyLoader:
    def __init__(self):
        self.policies = {}

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
        