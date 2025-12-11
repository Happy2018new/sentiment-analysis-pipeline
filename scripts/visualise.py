import csv
import matplotlib.pyplot as plt
from pathlib import Path
from adjustText import adjust_text
from analyse import (
    TokenWithScore,
    StemToLemMapping,
    CommentWithScore,
)


class Visualizer:
    """Visualizer is the visualizer that visualise the analysis results."""

    @staticmethod
    def save_comments_trend(
        data: list[float], output_path: str, steps: int = 5
    ) -> None:
        """
        save_comments_trend saves the comments sentiment
        trend chart to the given output path.

        Args:
            data (list[float]): The sentiment scores of all comments.
            output_path (str): The path to save the chart.
            steps (int, optional):
                The chart will split to `steps` chunks.
                And it will show the comments count of each chunk.
                Defaults to 5.
        """
        # Pre chunk
        if len(data) == 0 or steps <= 0:
            raise Exception("save_comments_trend: Invalid user input")

        # Edges / Chunks
        min_v, max_v = min(data), max(data)
        bin_width = (max_v - min_v) / steps
        edges = [min_v + i * bin_width for i in range(steps + 1)]

        # Count for each chunk
        counts = [0] * steps
        for x in data:
            idx = min(int((x - min_v) / bin_width), steps - 1)
            counts[idx] += 1

        # Colors, Label for center of each chunk
        cmap = plt.get_cmap("viridis")
        colors = cmap([i / (steps - 1) for i in range(steps)])
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(steps)]

        # Draw
        plt.figure(figsize=(14, 7))
        gca = plt.gca()
        _ = gca.bar(
            range(steps), counts, color=colors, edgecolor="black", linewidth=0.6
        )
        gca.set_xticks(range(steps))
        gca.set_xticklabels([f"{round(c, 2)}" for c in centers])
        gca.set_xlabel("Sentiment Score")
        gca.set_ylabel("Comment Count")
        gca.set_title("Sentiment Trend of Comments")

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches="tight")

    @staticmethod
    def save_tokens_trend(
        data: list[TokenWithScore], mapping: StemToLemMapping, output_path: str
    ) -> None:
        """
        save_tokens_trend saves the tokens sentiment
        trend chart to the given output path.

        Args:
            data (list[TokenWithScore]):
                All tokens of all comments.
            mapping (StemToLemMapping):
                The mapping that allows user get
                lemmatized token by stemmed token.
            output_path (str):
                The path to save the chart.
        """
        if len(data) == 0:
            raise Exception("save_tokens_trend: Invalid user input")

        # Prepare
        data = sorted(data, key=lambda t: t.score)
        x_vals = [t.score for t in data]
        y_vals = [t.count for t in data]
        labels = [mapping.get_lem_token(t.token, "(negative)") for t in data]

        # Colors
        cmap = plt.get_cmap("rainbow")
        colors = cmap([i / (len(data) - 1) for i in range(len(data))])

        # Normal
        plt.figure(figsize=(14, 7))
        plt.plot(x_vals, y_vals, color="lightblue", linewidth=1.5, zorder=1)
        plt.scatter(x_vals, y_vals, c=colors, edgecolor="black", zorder=3)

        # Label
        texts = []
        for x, y, txt in zip(x_vals, y_vals, labels):
            texts.append(plt.text(x, y, txt, ha="center", va="bottom", fontsize=9))
        adjust_text(
            texts,
            x=x_vals,
            y=y_vals,
            expand_points=(1.4, 1.4),
            expand_text=(1.2, 1.2),
            force_text=0.8,
            force_points=0.8,
        )

        # Normal
        plt.xlabel("Sentiment Score")
        plt.ylabel("Token Appear Count")
        plt.title("Sentiment Trend of Tokens")

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches="tight")


class CSVDumper:
    """
    CSVDumper is the dumper that used to dump
    our sentiment analysis results to CSV file.
    """

    @staticmethod
    def dump_comments_trend(file_path: str, data: list[CommentWithScore]) -> None:
        """
        dump_comments_trend dumps the comments sentiment trend to a CSV file.
        Note that if `file_path` is not exists, then it will be created.

        Args:
            file_path (str): The output CSV file path.
            data (list[CommentWithScore]): The comments sentiment trend data.
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w+", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["comment", "score"])
            for i in data:
                writer.writerow([i.comment.origin_text, i.score])

    @staticmethod
    def dump_tokens_trend(
        file_path: str, data: list[TokenWithScore], mapping: StemToLemMapping
    ) -> None:
        """dump_tokens_trend dumps the tokens sentiment trend to a CSV file.
        Note that if `file_path` is not exists, then it will be created.

        Args:
            file_path (str):
                The output CSV file path.
            data (list[TokenWithScore]):
                The tokens sentiment trend data.
            mapping (StemToLemMapping):
                The mapping that allows user get
                lemmatized token by stemmed token.
        """

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w+", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["stem_token", "lem_token", "appear_count", "score"])
            for i in data:
                writer.writerow(
                    [
                        i.token,
                        mapping.get_lem_token(i.token, "(negative)"),
                        i.count,
                        i.score,
                    ]
                )
