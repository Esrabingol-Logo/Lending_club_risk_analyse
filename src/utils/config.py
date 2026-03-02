from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import copy
import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(base_config_path: str | Path, dataset_config_path: str | Path) -> Dict[str, Any]:
    base_config_path = Path(base_config_path)
    dataset_config_path = Path(dataset_config_path)

    with base_config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    with dataset_config_path.open("r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f) or {}

    cfg = _deep_merge(base_cfg, ds_cfg)

    if "io" not in cfg or "file_path" not in cfg["io"]:
        raise ValueError("Config must include io.file_path (set it in lending_club.yaml).")
    if "paths" not in cfg:
        raise ValueError("Config must include paths section (comes from base_config.yaml).")

    return cfg
