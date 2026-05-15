import os


class Prompt:
    def __init__(self, template: str, keys: list[str], version: int | None = None):
        self.template = template
        self.keys = keys
        self.version = version

    @classmethod
    def from_file(cls, path: str, keys: list[str], version: int | None = None) -> "Prompt":
        with open(path, "r") as f:
            return cls(f.read(), keys, version=version)

    @classmethod
    def load(cls, name: str, version: int, keys: list[str], prompts_dir: str = "app/prompts") -> "Prompt":
        """Load a versioned prompt from app/prompts/{name}/v{version}.txt."""
        path = os.path.join(prompts_dir, name, f"v{version}.txt")
        return cls.from_file(path, keys, version=version)

    @staticmethod
    def available_versions(name: str, prompts_dir: str = "app/prompts") -> list[int]:
        """Return sorted list of available version numbers for a given prompt name."""
        prompt_dir = os.path.join(prompts_dir, name)
        if not os.path.isdir(prompt_dir):
            return []
        versions = []
        for fname in os.listdir(prompt_dir):
            if fname.startswith("v") and fname.endswith(".txt"):
                try:
                    versions.append(int(fname[1:-4]))
                except ValueError:
                    pass
        return sorted(versions)

    def format(self, **kwargs) -> str:
        result = self.template
        for key in self.keys:
            result = result.replace(f"{{{key}}}", str(kwargs[key]))
        return result

    @property
    def overhead(self) -> int:
        """Template length minus placeholder characters (fixed cost of the prompt)."""
        return len(self.template) - sum(len(f"{{{k}}}") for k in self.keys)