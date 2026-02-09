from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar


class ErrorKind:
    MOCASA = "Oriole error"
    IO = "I/O error"
    TOML_DE = "TOML deserialization error"
    TOML_SER = "TOML serialization error"
    PARSE_FLOAT = "parse float error"
    SEND = "send error"
    RECEIVE = "receive error"
    RECEIVE_TIMEOUT = "receive timeout error"
    SYSTEM_TIME = "system time error"
    SERDE_JSON = "Serde JSON"


@dataclass
class MocasaError(Exception):
    kind: str
    message: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.message}"


T = TypeVar("T")


def new_error(message: str) -> MocasaError:
    return MocasaError(ErrorKind.MOCASA, message)


def for_file(file: str, exc: Exception) -> MocasaError:
    return MocasaError(ErrorKind.IO, f"{file}: {exc}")


def for_context(context: str, exc: Exception) -> MocasaError:
    if isinstance(exc, MocasaError):
        return MocasaError(exc.kind, f"{context}: {exc.message}")
    return MocasaError(ErrorKind.MOCASA, f"{context}: {exc}")
