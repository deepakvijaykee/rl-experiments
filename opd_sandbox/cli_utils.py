"""Small argparse helpers for OPD appendix entry points."""

from __future__ import annotations

import argparse
import dataclasses
from typing import TypeVar, get_type_hints


ConfigT = TypeVar("ConfigT")
TYPE_MAP = {int: int, float: float, str: str, bool: bool}


def parse_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_int_list(text: str, name: str, min_value: int = 1) -> list[int]:
    try:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be comma-separated ints") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    if any(value < min_value for value in values):
        raise argparse.ArgumentTypeError(
            f"all {name} must be >= {min_value}")
    return sorted(set(values))


def add_dataclass_args(
        parser: argparse.ArgumentParser,
        config_type: type,
        skip: set[str] | None = None) -> None:
    """Expose dataclass fields as simple CLI flags."""
    skip = skip or set()
    type_hints = get_type_hints(config_type)
    for field in dataclasses.fields(config_type):
        if field.name in skip:
            continue
        field_type = type_hints[field.name]
        if field_type is bool:
            parser.add_argument(
                f"--{field.name}",
                type=parse_bool,
                default=field.default,
                metavar="BOOL",
            )
        else:
            parser.add_argument(
                f"--{field.name}",
                type=TYPE_MAP.get(field_type, str),
                default=field.default,
            )


def config_from_args(
        config_type: type[ConfigT],
        args: argparse.Namespace,
        skip: set[str] | None = None) -> ConfigT:
    skip = skip or set()
    return config_type(**{
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(config_type)
        if field.name not in skip
    })
