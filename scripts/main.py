import os
import argparse
from dataclasses import dataclass
from ingest import IngestReader
from preprocess import (
    FilterSentences,
    StopWordCleanner,
    InverseCompacter,
    StemToLemMapping,
)
from analyse import CommentWithScore, TokenProcesser, TokenWithScore
from visualise import Visualizer, CSVDumper


@dataclass
class Config:
    """Config is the config of this program."""

    input_stream: str
    output_csv_dir: str
    output_plot_dir: str
    visual_comments_chunks: int
    visual_tokens_percent: float


def parse_args() -> Config:
    """
    parse_args parses and returns
    arguments from the command line.

    Returns:
        Config: The parsed config.
    """
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Pipeline",
    )
    parser.add_argument(
        "--input-stream",
        type=str,
        help="the input stream file path, must be a `.jsonl` file",
        required=True,
    )
    parser.add_argument(
        "--output-csv-dir",
        type=str,
        help="the output dir of the sentiment analysis CSV files",
        required=True,
    )
    parser.add_argument(
        "--output-plot-dir",
        type=str,
        help="the output dir of the image for the visualization analysis result",
        required=True,
    )
    parser.add_argument(
        "--visual-comments-chunks",
        type=int,
        help="chunk numbers or steps that used for draw the comments sentiment trend",
        default=20,
    )
    parser.add_argument(
        "-visual-tokens-percent",
        type=float,
        help="to show the top N tokens by percentage in the ouput visualization for tokens sentiment trend",
        default=1.0,
    )

    args = parser.parse_args()
    return Config(
        input_stream=args.input_stream,
        output_csv_dir=args.output_csv_dir,
        output_plot_dir=args.output_plot_dir,
        visual_comments_chunks=args.visual_comments_chunks,
        visual_tokens_percent=args.visual_tokens_percent,
    )


def get_filter_sentences(path: str) -> list[FilterSentences]:
    """
    get_filter_sentences reads the input stream
    file as multiple filter sentences.

    Args:
        path (str):
            The input path of the stream file.

    Returns:
        list[FilterSentences]:
            The filter sentences that read from the stream file.
    """
    reader = IngestReader(path)  # type: IngestReader
    result = []  # type: list[FilterSentences]

    while True:
        comment = reader.read_next()
        if comment is None:
            break

        text = comment["text"]
        timestamp = comment["timestamp"]

        result.append(FilterSentences(text, timestamp))

    return result


def process_filter_sentences(
    sentences: list[FilterSentences],
) -> tuple[list[FilterSentences], StemToLemMapping]:
    """
    process_filter_sentences processes the given filter sentences,
    clean most of the stop words, and then compact the negative
    tokens and affected tokens into single tokens, and last make
    the stemmed token to lemmatized token map.

    Args:
        sentences (list[FilterSentences]):
            All the comments.

    Returns:
        tuple[list[FilterSentences], StemToLemMapping]:
            The process results.
    """
    sentences = [StopWordCleanner.clean(i) for i in sentences]
    sentences = [InverseCompacter.compact_sentences(i) for i in sentences]
    return sentences, StemToLemMapping().build_mapping(sentences)


def analysis_filter_sentences(
    sentences: list[FilterSentences],
    mapping: StemToLemMapping,
    percent: float = 0.2,
) -> tuple[list[CommentWithScore], list[TokenWithScore]]:
    """
    analysis_filter_sentences does the sentiment analysis
    by the given comments (sentences), mapping and percent.

    Args:
        sentences (list[FilterSentences]):
            All the comments.
        mapping (StemToLemMapping):
            A map that allows user to get
            lemmatized token by stemmed token.
        percent (float, optional):
            Only count the top `percent` percent tokens.
            Defaults to 0.2.

    Returns:
        tuple[list[CommentWithScore], list[TokenWithScore]]:
            The analysis results.
    """
    comment_scores = [CommentWithScore(i, True) for i in sentences]
    top_tokens = TokenProcesser.get_top_stem_tokens(sentences, percent)
    token_scores = TokenProcesser.get_token_score(top_tokens, mapping)
    return comment_scores, token_scores


def visual_analysis_results(
    comment_scores: list[CommentWithScore],
    token_scores: list[TokenWithScore],
    mapping: StemToLemMapping,
    config: Config,
) -> None:
    """
    visual_analysis_results performs visualization and
    saves the visualization results to the specified path.

    Args:
        comment_scores (list[CommentWithScore]):
            The sentiment scores and original comments of all comments.
        token_scores (list[TokenWithScore]):
            The sentiment scores and token payload of all tokens.
        mapping (StemToLemMapping): _description_
            The mapping used to find lemmatized token by stemmed token.
        config (Config):
            The config of this program.
    """
    Visualizer.save_comments_trend(
        [i.score for i in comment_scores],
        config.output_plot_dir + "/comments_sentiment_trend.png",
        config.visual_comments_chunks,
    )
    Visualizer.save_tokens_trend(
        token_scores,
        mapping,
        config.output_plot_dir + "/tokens_sentiment_trend.png",
    )


def dump_csv_file(
    comment_scores: list[CommentWithScore],
    token_scores: list[TokenWithScore],
    mapping: StemToLemMapping,
    config: Config,
) -> None:
    """
    dump_csv_file dumps the sentiment
    analysis results to CSV files.

    Note that all data will be sorted
    before save to the csv file.

    Args:
        comment_scores (list[CommentWithScore]):
            The sentiment scores and original comments of all comments.
        token_scores (list[TokenWithScore]):
            The sentiment scores and token payload of all tokens.
        mapping (StemToLemMapping):
            The mapping used to find lemmatized token by stemmed token.
        config (Config):
            The config of this program.
    """
    CSVDumper.dump_comments_trend(
        config.output_csv_dir + "/comments_sentiment_trend.csv",
        comment_scores,
    )
    CSVDumper.dump_tokens_trend(
        config.output_csv_dir + "/tokens_sentiment_trend.csv",
        token_scores,
        mapping,
    )


def print_exit_message(config: Config) -> None:
    """
    print_exit_message is executed after all the things are done.
    This function is just used to print some exit messages,
    to make the user know all of those things have been completed.

    Args:
        config (Config): The config of this program.
    """
    print("Sentiment analysis pipeline has completed successfully.")
    print()
    print(
        f"CSV files are saved to: \n\t{os.path.join(os.getcwd(), config.output_csv_dir)}"
    )
    print()
    print(
        f"Visualization plots are saved to: \n\t{os.path.join(os.getcwd(), config.output_plot_dir)}"
    )


def main() -> None:
    """main is the entry point of this program."""
    config = parse_args()
    sentences = get_filter_sentences(config.input_stream)
    sentences, mapping = process_filter_sentences(sentences)
    comment_scores, token_scores = analysis_filter_sentences(
        sentences, mapping, config.visual_tokens_percent
    )
    visual_analysis_results(comment_scores, token_scores, mapping, config)
    dump_csv_file(comment_scores, token_scores, mapping, config)
    print_exit_message(config)


main()
