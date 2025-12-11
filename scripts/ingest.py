import json
from typing import Any


class IngestReader:
    """IngestReader is the reader that read the ingested data from a file."""

    _comments: list[dict[str, Any]]
    _pointer: int

    def __init__(self, file_path: str):
        """
        Creates and returns a new reader.

        Ensure that comments will have the
        payload corresponding to the ingested
        data after this method is called.

        Args:
            file_path (str): The path point to the ingested data file.
        """
        self._read_all(file_path)
        self._pointer = 0

    def _read_all(self, file_path: str) -> None:
        """
        _read_all reads all the ingested data from the file,
        and store the read data into the `self.comments`.

        Args:
            file_path (str): The path point to the ingested data file.
        """
        streams: list[dict[str, Any]] = []
        with open(file_path, "r+", encoding="utf-8") as file:
            for line in file:
                streams.append(json.loads(line))
        self._comments = streams

    def read_next(self) -> dict[str, Any] | None:
        """
        read_next reads the next comment from the ingested data.

        Returns:
            dict[str, Any] | None:
                The next comment if exists;
                otherwise, the underlying pointer meet EOF,
                and then return None.
        """
        if self._pointer >= len(self._comments):
            return None
        result = self._comments[self._pointer]
        self._pointer += 1
        return result

    def unread(self) -> IngestReader:
        """
        unread unread one comment from the ingested data.

        Returns:
            IngestReader: Returns `IngestReader` itself.
        """
        self._pointer = max(0, self._pointer - 1)
        return self
