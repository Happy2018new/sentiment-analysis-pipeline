from dataclasses import dataclass
from preprocess import FilterSentences, StemToLemMapping
from nltk.sentiment.vader import SentimentIntensityAnalyzer

DEFAULT_VADER_ANALYZER = SentimentIntensityAnalyzer()


class CommentWithScore:
    """CommentWithScore stores a comment text and its sentiment score."""

    comment: FilterSentences
    score: float

    def __init__(self, comment: FilterSentences, average_mode: bool = False) -> None:
        """Creates and returns a new CommentWithScore.

        Args:
            comment (FilterSentences): The comment text.
            average_mode (bool, optional):
                When compute the sentiment score of this comment,
                should to use the average sentiment score of the all sentences or not.
                Defaults to False.
        """
        self.comment = comment
        self.score = CommentProcesser.process_comment(comment, average_mode)

    def __repr__(self) -> str:
        """Returns the string representation of this CommentWithScore.

        Returns:
            str: The string representation of this CommentWithScore.
        """
        return f"CommentWithScore(comment={self.comment}, score={self.score})"


class CommentProcesser:
    """CommentProcesser is the processer that compute the sentiment score of comments."""

    @staticmethod
    def process_comment(
        sentences: FilterSentences, average_mode: bool = False
    ) -> float:
        """
        process_comment computes the
        sentiment score of this comment.

        Args:
            sentences (FilterSentences): The FilterSentences to be processed.
            average_mode (bool, optional):
                When compute the sentiment score of this comment,
                should to use the average sentiment score of the all sentences or not.
                Defaults to False.

        Returns:
            float: The sentiment score of this comment.
        """
        if not average_mode:
            result = DEFAULT_VADER_ANALYZER.polarity_scores(sentences.origin_text)
            return result["compound"]

        scores: list[float] = []
        for i in sentences.sent_tokens:
            score = DEFAULT_VADER_ANALYZER.polarity_scores(i)
            scores.append(score["compound"])

        if len(scores) == 0:
            return 0.0
        return sum(scores) / len(scores)


@dataclass
class TokenWithScore:
    """
    TokenWithScore holds a stemmed/lemmatized token,
    and also the appear count and sentiment score of it.
    """

    token: str
    count: int
    score: float = 0.0

    def __repr__(self) -> str:
        """Returns the string representation of this TokenWithScore.

        Returns:
            str: The string representation of this TokenWithScore.
        """
        return f"TokenWithScore(token={self.token}, count={self.count}, score={self.score})"


class TokenProcesser:
    """
    TokenProcesser is the processer that compute the top N percent stem tokens,
    and also used to compute the sentiment score of the given stem tokens.
    """

    @staticmethod
    def get_top_stem_tokens(
        comments: list[FilterSentences], percent: float = 0.2
    ) -> list[TokenWithScore]:
        """
        get_top_stem_tokens gets the top `percent`
        percent stem tokens of the given comments.

        Args:
            comments (list[FilterSentences]): The given comments.
            percent (float, optional):
                The given percent.
                Defaults to 0.2 (20%).

        Returns:
            list[TokenWithScore]: Returns the top `percent` percent stem tokens.
        """
        mapping: dict[str, int] = {}
        for comment in comments:
            for tokens in comment.stem_tokens:
                for token in tokens:
                    if len(token) == 0:
                        continue
                    if token not in mapping:
                        mapping[token] = 0
                    mapping[token] += 1

        token_list = [(value, key) for key, value in mapping.items()]
        token_list.sort(reverse=True)

        result = token_list[: int(len(token_list) * percent)]
        result = [TokenWithScore(i[1], i[0]) for i in result]
        return result

    @staticmethod
    def get_token_score(
        tokens: list[TokenWithScore], mapping: StemToLemMapping
    ) -> list[TokenWithScore]:
        """
        get_token_score computes the sentiment score of the given stem tokens,
        and also convert each stemmed token to lemmatized token by the given mapping.

        Args:
            tokens (list[TokenWithScore]): The given stem tokens.
            mapping (StemToLemMapping):
                The mapping that used to find
                lemmatized token by stemmed token.

        Returns:
            list[TokenWithScore]: The results that corresponding to the given tokens.
        """
        result: list[TokenWithScore] = []

        for i in tokens:
            if not mapping.check_stem_token(i.token):
                continue

            lem_token = mapping.get_lem_token(i.token, "")
            lem_score = DEFAULT_VADER_ANALYZER.polarity_scores(lem_token)
            lem_score = lem_score["compound"]

            if i.token.startswith("NEG_"):
                # By asking AI how to inverse the meaning,
                # it suggests me to times -0.74 and that's
                # why I do it here.
                #       -- Eternal Crystal
                lem_score *= -0.74

            if lem_score != 0.0:
                result.append(TokenWithScore(i.token, i.count, lem_score))

        return result
