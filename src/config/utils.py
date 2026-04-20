import dataclasses
import typing
from pathlib import Path
from typing import Any, Type

import tyro
import yaml


def _build(cls: Type, data: Any) -> Any:
    """Recursively construct a dataclass instance from a nested dict."""
    if not dataclasses.is_dataclass(cls) or not isinstance(data, dict):
        return data
    # use get_type_hints so string annotations (PEP 563) resolve to real types
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _build(hints[f.name], data[f.name])
    return cls(**kwargs)


def load_config(config_cls: Type, yaml_path: str | Path = "conf/config.yaml"):
    """Load defaults from YAML, then layer CLI overrides on top via tyro."""
    path = Path(yaml_path)
    default = config_cls()
    if path.exists():
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        default = _build(config_cls, raw)
    return tyro.cli(config_cls, default=default)


def flatten(d: dict, sep: str = ".") -> dict:
    """Flatten a nested dict, joining keys with `sep`."""

    def _helper(d: dict, prefix: str = "") -> dict:
        items: dict = {}
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                items.update(_helper(v, key))
            else:
                items[key] = v
        return items

    return _helper(d)
