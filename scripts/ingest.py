import json


class IngestReader:
    """IngestReader is the reader that read the ingested data from a file."""

    file_path: str

    def __init__(self, file_path: str):
        """Creates and returns a new reader

        Args:
            file_path (str): The path point to the ingested data file.
        """
        self.file_path = file_path

    def read(self) -> list[dict]:
        """read reads all the ingested data from the file.

        Returns:
            list[dict]: All the ingested data from the file.
                        Each element in this list is a basic JSON dict.
        """
        streams: list[dict] = []
        with open(self.file_path, "r+", encoding="utf-8") as file:
            for line in file:
                streams.append(json.loads(line))
        return streams
