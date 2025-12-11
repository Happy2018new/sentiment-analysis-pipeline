import json
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from dataclasses import dataclass, field

DEFAULT_PORTER_STEMMER = PorterStemmer()
DEFAULT_LEMMATIZER = WordNetLemmatizer()

CONST_STOP_WORDS = set(stopwords.words("english"))
CONST_STOP_WORDS |= {",", ".", "!", "?", ";", ":", "'", '"', "`", "``"}
CONST_STOP_WORDS -= {
    "shan't",
    "wouldn't",
    "shouldn't",
    "wasn't",
    "aren't",
    "not",
    "mightn't",
    "no",
    "doesn't",
    "hasn't",
    "won't",
    "isn't",
    "out",
    "don't",
    "didn't",
    "needn't",
    "mustn't",
    "hadn't",
    "couldn't",
    "off",
    "nor",
}
CONST_STOP_WORDS -= {"very", "to"}
CONST_STOP_WORDS -= {"should", "can", "will"}
CONST_STOP_WORDS -= {"if"}
CONST_STOP_WORDS -= {"which", "who", "this", "those", "that", "these", "whom"}
CONST_STOP_WORDS -= {"each", "most", "few", "all", "some", "more", "any"}
CONST_STOP_WORDS -= {"before", "between", "during", "against", "after"}


class LemmerWrapper:
    """LemmerWrapper wrapped the WordNetLemmatizer to provide a simple interface."""

    @staticmethod
    def get_file_key(pos_tag: str) -> str:
        """
        get_file_key returns the file key by the given pos_tag.
        "file key" is the part of speech of a word.

        Args:
            pos_tag (str): The result of method `nltk.pos_tag`.

        Returns:
            str: The file key corresponding to the given pos_tag.
        """
        return {
            "NN": "n",
            "VB": "v",
            "RB": "r",
            "JJ": "a",
        }.get(pos_tag[:2], "n")

    @staticmethod
    def lemmatize(word: str) -> str:
        """
        lemmatize lemmatizes the given word.
        It wrapped `PorterStemmer().lemmatize`.

        Args:
            word (str): The given word.

        Returns:
            str: The lemmatized result.
        """
        tag = nltk.pos_tag([word])[0][1]
        return DEFAULT_LEMMATIZER.lemmatize(word, LemmerWrapper.get_file_key(tag))


class FilterSentences:
    """
    FilterSentences is corresponding to a single comment,
    and stores the sentence tokens, word tokens,
    stemmed tokens and lemmatized tokens of this comment.
    """

    origin_text: str
    timestamp: str
    sent_tokens: list[str]
    word_tokens: list[list[str]]
    stem_tokens: list[list[str]]
    lem_tokens: list[list[str]]

    def __init__(self, text: str, timestamp: str) -> None:
        """Creates and returns a new FilterSentences.

        Args:
            text (str): The comment text of this FilterSentences.
            timestamp (str): The submit time of this comment.
        """
        self.origin_text = text
        self.timestamp = timestamp
        self.sent_tokens = nltk.sent_tokenize(self.origin_text)
        self.word_tokens = [nltk.word_tokenize(i) for i in self.sent_tokens]
        self.stem_tokens = [
            [DEFAULT_PORTER_STEMMER.stem(j) for j in i] for i in self.word_tokens
        ]
        self.lem_tokens = [
            [LemmerWrapper.lemmatize(j).lower() for j in i] for i in self.word_tokens
        ]

    def __repr__(self) -> str:
        """Returns the string representation of this FilterSentences.

        Returns:
            str: The string representation of this FilterSentences.
        """
        result = "FilterSentences("
        result += f"origin_text={json.dumps(self.origin_text,ensure_ascii=False)}, "
        result += f"timestamp={self.timestamp}, "
        result += f"sent_tokens={self.sent_tokens}, "
        result += f"word_tokens={self.word_tokens}, "
        result += f"stem_tokens={self.stem_tokens}, "
        result += f"lem_tokens={self.lem_tokens}"
        return result + ")"


class StopWordCleanner:
    """StopWordCleanner is used to clean the stop words from the given FilterSentences."""

    @staticmethod
    def clean(sent: FilterSentences) -> FilterSentences:
        """
        clean cleans the stop words
        of the given sent.

        Note that only the following
        field of sent will be cleaned:
            - stem_tokens
            - lem_tokens

        Args:
            sent (FilterSentences): The given FilterSentences.

        Returns:
            FilterSentences: The cleaned FilterSentences.
        """
        sent.stem_tokens = [
            [j if j not in CONST_STOP_WORDS else "" for j in i]
            for i in sent.stem_tokens
        ]
        sent.lem_tokens = [
            [j if j not in CONST_STOP_WORDS else "" for j in i] for i in sent.lem_tokens
        ]
        return sent


class InverseCompacter:
    """
    InverseCompacter is a compacter that compacts inverse tokens.
    It gathers negative words into one word.
    """

    WINDOW_SIZE = 3
    NEG_TOKENS = {
        "won't",
        "n't",
        "out",
        "without",
        "no",
        "don't",
        "mightn't",
        "isn't",
        "doesn't",
        "shouldn't",
        "can't",
        "wouldn't",
        "hadn't",
        "nor",
        "off",
        "cannot",
        "needn't",
        "never",
        "shan't",
        "didn't",
        "couldn't",
        "mustn't",
        "not",
        "aren't",
        "hasn't",
        "wasn't",
    }

    @staticmethod
    def compact_tokens(tokens: list[str]) -> list[str]:
        """
        compact_tokens compacts the given tokens.
        It collect all the negative tokens and those
        affected tokens, make each of them in one token.

        Args:
            tokens (list[str]): The given tokens.

        Returns:
            list[str]: The compact result.
        """
        result = []
        count = 0

        for token in tokens:
            token = token.lower()
            if token in InverseCompacter.NEG_TOKENS:
                count = InverseCompacter.WINDOW_SIZE
                result.append(token)
                continue
            if count > 0:
                result.append(f"NEG_{token}")
                count -= 1
            else:
                result.append(token)

        return result

    @staticmethod
    def compact_sentences(sent: FilterSentences) -> FilterSentences:
        """
        compact_sentences compacts `sent.stem_tokens` by calling `compact_tokens`.

        Args:
            sent (FilterSentences): The given FilterSentences.

        Returns:
            FilterSentences: The compacted FilterSentences.
        """
        sent.stem_tokens = [
            InverseCompacter.compact_tokens(i) for i in sent.stem_tokens
        ]
        return sent


@dataclass
class StemToLemMapping:
    """StemToLemMapping is the map that allows user get lemmatized token by stemmed token."""

    mapping: dict[str, str] = field(default_factory=lambda: {})

    def build_mapping(self, sentences: list[FilterSentences]) -> StemToLemMapping:
        """
        build_mapping initializes current
        mapping by the given sentences.

        Args:
            sentences (list[FilterSentences]): The given FilterSentences.

        Returns:
            StemToLemMapping: Returns `StemToLemMapping` itself.
        """
        for sentence in sentences:
            for i, tokens in enumerate(sentence.stem_tokens):
                for j, token in enumerate(tokens):
                    key, value = token, sentence.lem_tokens[i][j]
                    if len(token) == 0 or len(value) == 0:
                        continue
                    if key in self.mapping:
                        continue
                    self.mapping[key] = value
        return self

    def check_stem_token(self, stem_token: str) -> bool:
        """check_stem_token checks whether the given stem_token exists in the mapping.

        Args:
            stem_token (str): The given stem_token.

        Returns:
            bool: If `stem_token` in current mapping, then return True;
                  otherwise, return False.
        """
        return stem_token in self.mapping

    def get_lem_token(self, stem_token: str, negative_prefix: str = "not") -> str:
        """
        get_lem_token finds the lemmatized
        token by the given stem_token.

        Args:
            stem_token (str): The given stem_token.
            negative_prefix (str, optional):
                The prefix of the results.
                Only used for these negative tokens.
                Defaults to "not".

        Returns:
            str: The lemmatized token corresponding to the given stem_token.
        """
        result = self.mapping.get(stem_token, stem_token)
        if stem_token.startswith("NEG_"):
            return negative_prefix + " " + result
        return result
