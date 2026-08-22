import yaml
from pathlib import Path


class PolicyLoader:
    def __init__(self):
        self.policies: dict = {}

    @staticmethod
    def load(path: str | Path) -> dict:
        """Load a YAML guardrails file and return structured policy dict."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # Normalize to a consistent shape downstream code can rely on
        return {
            "agent_name": raw.get("agent", "Fin-gent"),
            "allowed_tools": raw.get("allow", []),
            "forbidden_tools": raw.get("deny", []),
            "flow": raw.get("flow", []),
        }

    def load_into(self, path: str | Path):
        """Load and store policy on this instance."""
        self.policies = self.load(path)
        return self.policies