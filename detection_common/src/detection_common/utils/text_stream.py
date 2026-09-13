"""
Utilities for filtering text streams.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TextIO, cast
from uuid import NAMESPACE_URL, uuid5


class _FilteredTextStream:
    _ID = uuid5(
        NAMESPACE_URL,
        f"python://{__name__}/filtered_text_stream",
    )

    def __init__(self, stream: TextIO, map: dict[str, str]) -> None:
        """
        Initialize a stream with its ordered string substitutions.
        """
        self._stream: TextIO = stream
        self._map = tuple(map.items())
        self._stream_filter_id = self._ID

    def write(self, text: str) -> int:
        """
        Write text after applying each configured substitution.
        """
        filtered_text = text
        for old, new in self._map:
            filtered_text = filtered_text.replace(old, new)

        self._stream.write(filtered_text)
        return len(text)

    def writelines(self, lines: Iterable[str]) -> None:
        """
        Write lines after filtering each line independently.
        """
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        """
        Flush the wrapped stream.
        """
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        """
        Delegate unsupported attributes to the wrapped stream.
        """
        return getattr(self._stream, name)


def set_text_stream_filter(stream: TextIO, *, map: dict[str, str]) -> TextIO:
    """
    Return a text stream that substitutes strings independently in each write.
    """
    stream = unset_text_stream_filter(stream)
    return cast(TextIO, _FilteredTextStream(stream, map))


def unset_text_stream_filter(stream: _FilteredTextStream | TextIO) -> TextIO:
    """
    Unset text filter from a stream.
    """
    while getattr(stream, "_stream_filter_id", None) == _FilteredTextStream._ID:
        wrapped_stream = getattr(stream, "_stream", None)
        if wrapped_stream is None:
            break
        stream = cast(TextIO, wrapped_stream)

    return cast(TextIO, stream)


def aligned_print(
        keyvals: Mapping[str, object],
        *,
        stream: TextIO | None = None
) -> None:
    """
    Print key/value pairs with keys left-aligned before their colons.
    """
    if not keyvals:
        return

    key_width = max(len(key) for key in keyvals)
    for key, value in keyvals.items():
        print(f"{key:<{key_width}}: {value}", file=stream)
