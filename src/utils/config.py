# TIDES/src/utils/config.py
import yaml
from pathlib import Path
import logging

class ConfigManager:
    def __init__(self, args):
        self.args = args
        self.config = self._load_config()

    def _load_config(self):
        config = self._load_default_config()
        config.update(self._load_dataset_config())
        if self.args.config:
            config.update(self._load_custom_config())
        config.update(self._get_args_config())
        return config

    def _load_default_config(self):
        default_path = Path('config/default_config.yaml')
        if not default_path.exists():
            raise FileNotFoundError(f"Default config not found at {default_path}")
        with open(default_path) as f:
            return yaml.safe_load(f)

    def _load_dataset_config(self):
        dataset_path = Path(f'config/{self.args.dataset}_config.yaml')
        if dataset_path.exists():
            with open(dataset_path) as f:
                return yaml.safe_load(f)
        return {}

    def _load_custom_config(self):
        custom_path = Path(self.args.config)
        if custom_path.exists():
            with open(custom_path) as f:
                return yaml.safe_load(f)
        return {}

    def _get_args_config(self):
        args_config = {
            'model': {
                'type': self.args.model_type,
                'api_key': self.args.api_key
            },
            'retrieval': {
                'method': self.args.retriever
            }
        }
        
        if self.args.model_name:
            args_config['model']['name'] = self.args.model_name
        if self.args.top_k:
            args_config['retrieval']['top_k'] = self.args.top_k
        if self.args.output_dir:
            args_config['data']['output_dir'] = self.args.output_dir
            
        return args_config

    def get_config(self):
        return self.config