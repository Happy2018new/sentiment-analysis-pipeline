import json
from typing import Any


class IngestReader:
    """IngestReader is the reader that read the ingested data from a file."""

    comments: list[dict[str, Any]]

    def __init__(self, file_path: str):
        """
        Creates and returns a new reader.

        Ensure that comments will have the
        payload corresponding to the ingested
        data after this method is called.

        Args:
            file_path (str): The path point to the ingested data file.
        """
        self._read(file_path)

    def _read(self, file_path: str) -> None:
        """
        _read reads all the ingested data from the file,
        and store the read data into the `self.comments`.

        Args:
            file_path (str): The path point to the ingested data file.
        """
        streams: list[dict[str, Any]] = []
        with open(file_path, "r+", encoding="utf-8") as file:
            for line in file:
                streams.append(json.loads(line))
        self.comments = streams
