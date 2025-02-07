import os
import re
import emot
import spacy
import spacy.tokens
import fasttext

from lingua import LanguageDetectorBuilder
from langdetect import detect

from typing import List, Dict, Tuple, Optional

# Directory setup
directory = os.getcwd().split(os.path.sep)
root_index = directory.index("V_HLT")
root_path = os.path.sep.join(directory[:root_index+1])



class Cleaner:
    """
    Class to clean text data using various methods.

    Attributes
    ----------
    `model` : emot.core.Emot
        The emot model used for processing emojis and emoticons.
    `patterns` :  dict
        Dictionary containing patterns for contracted forms and text removal.
        The dictionary should have the following structure:
        
        -- contracted : list of tuples
            List of tuples where each tuple contains:
            
            -- pattern (str): The regex pattern to be replaced.
            -- replacement (str): The string to replace the pattern with.

            .. code-block:: python
            Example:
            {
                "contracted": [
                    (r"\\bcan't\\b", "cannot"),
                    (r"\\bwon't\\b", "will not"),
                    ...
                ]
            }
    
    Methods
    -------
    * `_mentions_remover(text)`:
        Remove mentions from text.
    
    * `_from_text_remover(text, values)`:
        Remove specific values from text.
    
    * `_newlines_remover(text)`:
        Replace newlines and carriage returns with space and convert text to lowercase.
    
    * `_non_ascii_remover(text)`:
        Remove non-ASCII characters from text.
    
    * `_repeated_letters_remover(text)`:
        Remove repeated letters in words.
    
    * `_tags_html_remover(text)`:
        Remove HTML tags from text.
    
    * `_contracted_form(text, patterns)`:
        Expand contracted forms in text based on given patterns.
    
    * `_links_remover(text)`:
        Remove links from text.
    
    * `_hashs_remover(text)`:
        Remove hashtags from text.
    
    * `emoji_remover(text)`:
        Remove emojis from text using the emot library.
    
    * `emoticons_remover(text)`:
        Remove emoticons from text using the emot library.
    
    * `text_cleaner(text)`:
        Clean text using various cleaning methods.
    """

    def __init__(self, patterns: Dict[str, List[Tuple[str, str]]]) -> None:
        """
        Initialize the `Cleaner` class with given patterns.
        
        :param `patterns`: Dictionary containing patterns for contracted forms and text removal.
        """
        self.model = emot.core.emot()
        self.patterns = patterns

    @staticmethod
    def _mentions_remover(text: str) -> str:
        """Remove mentions from text."""
        return re.sub(r"@[\w]+(?:_[\w]+)?", "", text)

    @staticmethod
    def _from_text_remover(text: str, values: List[str]) -> str:
        """Remove specific values from text."""
        response = "|".join(map(re.escape, values))
        return re.sub(response, "", text)
    
    @staticmethod
    def _newlines_remover(text: str) -> str:
        """Replace newlines and carriage returns with space and convert text to lowercase."""
        return text.replace("\r", "").replace("\n", " ").lower()
    
    @staticmethod
    def _non_ascii_remover(text: str) -> str:
        """Remove non-ASCII characters from text."""
        return re.sub(r"[^\x00-\x7F]+", "", text)
    
    @staticmethod
    def _repeated_letters_remover(text: str) -> str:
        """Remove repeated letters in words."""
        return re.sub(r"(\w)(\1{2,})\b", r"\1\1", text)
    
    @staticmethod
    def _tags_html_remover(text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r"<[^>]*>", "", text)

    @staticmethod
    def _contracted_form(text: str, patterns: Dict[str, List[Tuple[str, str]]]) -> str:
        """Expand contracted forms in text based on given patterns."""
        for pattern, replacement in patterns["contracted"]:
            text = re.sub(pattern, replacement, text)
        return text

    @staticmethod
    def _links_remover(text: str) -> str:
        """Remove links from text."""
        return re.sub(
            r"\b(?:https?|ftp|file):\/\/\S+|www\.\S+|\b(?:fb|t|pic)\.me\/\S+|\b(?:pic\.|twitter\.)com\/\S+",
            "",
            text
        )

    @staticmethod
    def _hashs_remover(text: str) -> str:
        """Remove hashtags from text."""
        return re.sub(r"[#$]", "", text)


    def emoji_remover(self, text: str) -> str:
        """
        Remove emojis from text using the emot library.
        
        :param `text`: Text from which emojis are to be removed.
        :return: Text with emojis removed.
        """
        response = self.model.emoji(text)["value"]
        return self._from_text_remover(text, response)
    
    def emoticons_remover(self, text: str) -> str:
        """
        Remove emoticons from text using the emot library.
        
        :param `text`: Text from which emoticons are to be removed.
        :return: Text with emoticons removed.
        """
        response = self.model.emoticons(text)["value"]
        return self._from_text_remover(text, response)
    
    def hypen_remover(self, text: str) -> str:
        try:
            non_hypened = list()
            splitted_text = text.split()
            for split in splitted_text:
                unhypen = split.replace("-", " ")
                non_hypened.append(unhypen)
            return " ".join(non_hypened)
        except Exception as e:
            raise ValueError(f"Hypen Remover Error: {e}")

    def text_cleaner(self, text: str) -> str:
        """
        Clean text using various cleaning methods.

        :param `text`: Text to be cleaned.
        :return: Cleaned text.
        """
        try:
            cln_text = text
            cln_text = self._mentions_remover(cln_text)
            cln_text = self.emoji_remover(cln_text)
            cln_text = self.emoticons_remover(cln_text)
            cln_text = self._non_ascii_remover(cln_text)
            cln_text = self._repeated_letters_remover(cln_text)
            cln_text = self._tags_html_remover(cln_text)
            cln_text = self._links_remover(cln_text)
            cln_text = self._hashs_remover(cln_text)
            cln_text = self._contracted_form(cln_text, self.patterns)
            cln_text = self.hypen_remover(cln_text)
            cln_text = self._newlines_remover(cln_text)
            return cln_text
        
        except Exception as e:
            raise ValueError(f"Cleaning error: {e}")


class LanguageDetector:
    """
    Class to detect the language of a given text using various language detection models.

    Attributes
    ----------
    `path` : str
        Path to the FastText language detection model file.
    `model` : fasttext.FastText._FastText
        Loaded FastText language detection model.
    `detector` : lingua.LanguageDetector
        Language detector from the Lingua library.

    Methods
    -------
    * `detector_langdetect(text)`:
        Detect language using the langdetect library.
    
    * `detector_fasttext(text)`:
        Detect language using the FastText model.
    
    * `detector_language(text)`:
        Detect language using the Lingua library.
    
    * `language_detector(text, metrics)`:
        Detect language using the specified language detection model.

    """
    def __init__(self) -> None:
        """
        Initialize the `LanguageDetector` class by loading the necessary language detection models.
        """
        self.path = os.path.join(root_path, "external_models", "lid.176.bin")
        self.model = fasttext.load_model(self.path)
        self.detector = LanguageDetectorBuilder.from_all_languages().build()

    def detector_langdetect(self, text: str) -> str:
        """
        Detect language using `langdetect` library.

        :param `text`: Text whose language is to be detected.
        :return: Detected language code.
        """
        return detect(text)
    
    def detector_fasttext(self, text: str) -> str:
        """
        Detect language using `fasttext` library.

        :param `text`: Text whose language is to be detected.
        :return: Detected language code.
        """
        pred = self.model.predict(text, k=1)
        return pred[0][0].split("__label__")[1]
    
    def detector_language(self, text: str) -> str:
        """
        Detect language using `lingua` library.

        :param `text`: Text whose language is to be detected.
        :return: Detected language code.
        """
        result = self.detector.detect_language_of(text)
        return result.iso_code_639_1.name.lower()
    
    def language_detector(self, text: str, metrics: str) -> Optional[str]:
        """
        Detect language using `langdetect` library.

        :param `text`: Text whose language is to be detected.
        :param `metrics`: The language detection model to use (`langdetect`, `fasttext`, or `lingua`).
        :return: Detected language code, or None if language is not recognized.
        """
        try:
            if metrics == "langdetect":
                return self.detector_langdetect(text)
            elif metrics == "fasttext":
                return self.detector_fasttext(text)
            elif metrics == "lingua":
                return self.detector_language(text)
            else:
                raise ValueError(f"{metrics} are not supported.")
        except Exception as e:
            return None
        