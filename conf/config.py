import yaml
from pathlib import Path

class ConfigNode:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

class ZeroGraspConfig:
    def __init__(self, config_path: str="conf/config.yaml"):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(path, 'r', 'utf-8') as f:
            raw_cfg = yaml.safe_load(f)
        
        if raw_cfg is None:
            raise ValueError(f"Config file at {config_path} is empty or invalid.")
        
        self.dataset = ConfigNode(raw_cfg.get('dataset', {}))
        self.model = ConfigNode(raw_cfg.get('model', {}))
        self.rl_agent = ConfigNode(raw_cfg.get('rl_agent', {}))
        self.train = ConfigNode(raw_cfg.get('train', {}))

    def __getattr__(self, name):
        return getattr(self.config, name)
    
cfg = ZeroGraspConfig()