"""
Configuration and Environment Utilities
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def get_project_root() -> Path:
    """Returns absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_root(default_subpath: str = "data/raw") -> Path:
    """
    Resolves data root path with precedence:
    1. DATA_ROOT environment variable (if set)
    2. UTA_RLDD_ROOT environment variable (if set)
    3. Local project default path: <project_root>/data/raw
    """
    env_data_root = os.environ.get("DATA_ROOT") or os.environ.get("UTA_RLDD_ROOT")
    if env_data_root:
        return Path(env_data_root).resolve()
    return (get_project_root() / default_subpath).resolve()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file safely."""
    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / path
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}
