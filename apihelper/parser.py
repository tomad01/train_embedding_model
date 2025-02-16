import re, logging
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import html_text
from apihelper.config import ParsingParameters

logger = logging.getLogger(__name__)

import string


class Parser:
    def __init__(
        self,
        languages: list = ["english"],
        custom_stopwords: list = [],
        settings: ParsingParameters = ParsingParameters(),
    ):
        self.languages = languages
        self.settings = settings

        if self.settings.remove_stopwords:
            try:
                import nltk
                nltk.download("stopwords")
                nltk.download("punkt")
            except Exception as e:
                logger.error(f"nltk not installed: {e}")
            self.custom_stopwords = custom_stopwords
            # download language specific stopwords
            for lang in self.languages:
                try:
                    self.custom_stopwords.extend(stopwords.words(lang))
                    logger.info(
                        f"Stopwords for {lang} added. Total stopwords: {len(self.custom_stopwords)}"
                    )
                except Exception as e:
                    logger.error(f"nltk not installed: {e}")
            self.stopword_list = set(self.custom_stopwords)
            self.tknzr = TweetTokenizer()
        self.email_regex = re.compile(r"\S*@\S*\s?")
        self.domain_regex1 = re.compile(r"[^\s]*\.(com|org|net|fi)\S*")
        self.domain_regex2 = re.compile(r"https?://\S+|www\.\S+")
        self.brackets_regex1 = re.compile(r"\[.*?\]")
        self.brackets_regex2 = re.compile("<.*?>+")
        self.punctuation_regex = re.compile("[%s]" % re.escape(string.punctuation))
        self.numbers_regex = re.compile(r"\w*\d\w*")


    def parse_text(self, text: str):
        if self.settings.remove_html_tags:
            text = self.remove_html_tags(text)
        if self.settings.limit_length:
            text = self.limit_length(text)
        if self.settings.lowercase:
            text = self.lowercase_text(text)
        text = self.remove_emails(text)
        text = self.remove_domains(text)
        text = self.remove_brackets(text)
        text = self.remove_punctuation(text)
        if self.settings.remove_stopwords:
            text = self.remove_stopwords(text)
        if self.settings.remove_punctuation:
            text = self.remove_numbers(text)
        return text

    def remove_html_tags(self, text):
        return html_text.extract_text(text)

    def lowercase_text(self, text):
        """This function makes text lowercase"""
        return text.lower()

    def remove_emails(self, text):
        """This function removes emails from text"""
        return self.email_regex.sub("", text)

    def remove_domains(self, text):
        """This function removes .com .org .net .fi domains
        and strings with https www from text"""
        text = self.domain_regex1.sub("", text)
        text = self.domain_regex2.sub("", text)
        return text

    def remove_stopwords(self, text):
        text_tokens = self.tknzr.tokenize(text)
        filtered_tokens = [
            word for word in text_tokens if word not in self.stopword_list
        ]
        filtered_text = " ".join(filtered_tokens)
        return filtered_text

    def remove_brackets(self, text):
        """This function removes content within []-brackets and <>-marks from text"""
        text = self.brackets_regex1.sub("", text)
        text = self.brackets_regex2.sub("", text)
        return text

    def remove_punctuation(self, text):
        """This function removes punctuation from text"""
        return self.punctuation_regex.sub(" ", text)

    def remove_numbers(self, text):
        """This function removes numbers, number series,
        and words that have numbers in them from text"""
        return self.numbers_regex.sub("", text)

    def limit_length(self, text):
        words = str(text).split()[: self.settings.max_words]
        words_iterator = map(lambda w: w[: self.settings.max_word_length], words)
        text = " ".join(
            word for word in words_iterator if len(word) > self.settings.min_word_length
        )
        return text


if __name__ == "__main__":
    parser = Parser(
        languages=["english", "finnish", "french", "german"],
        custom_stopwords=["test", "stopword"],
    )
    text = "This is a test text with an email address and a domain www.google.com and a number 1234567890 and a bracket [test] and a punctuation mark ! and a stopword the and what else is there to test"
    print(parser.parse_text(text))
    print(len(parser.stopword_list))
