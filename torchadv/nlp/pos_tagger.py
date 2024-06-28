import nltk
import stanza


class POSTagger:
    def __init__(self, library="nltk"):
        """
        Initialize the POS tagger with the specified library.

        Args:
            library (str): The library to use for POS tagging. Options are "nltk", "stanza".
        """
        self.library = library.lower()

        if self.library == "nltk":
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)

        elif self.library == "stanza":
            stanza.download('en', verbose=False)
            self._nlp = stanza.Pipeline('en', processors='tokenize,pos')

        else:
            raise ValueError("Unsupported library. Choose from 'nltk', 'stanza'.")

    def tag(self, text):
        """
        Perform POS tagging on the provided text.

        Args:
            text (str): The input text to be POS tagged.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing the word and its POS tag.
        """
        if self.library == "nltk":
            return self._tag_nltk(text)

        elif self.library == "stanza":
            return self._tag_stanza(text)

        else:
            raise ValueError("Unsupported library. Choose from 'nltk', 'flair', 'stanza'.")

    def _tag_nltk(self, text):
        """
        POS tagging using NLTK.

        Args:
            text (str): The input text to be POS tagged.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing the word and its POS tag.
        """
        tokens = nltk.word_tokenize(text)
        nltk_tags = nltk.pos_tag(tokens)
        return [(word, self._map_to_universal_tag(tag)) for word, tag in nltk_tags]

    def _tag_stanza(self, text):
        """
        POS tagging using Stanza.

        Args:
            text (str): The input text to be POS tagged.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing the word and its POS tag.
        """
        doc = self._nlp(text)
        return [(word.text, word.upos) for sentence in doc.sentences for word in sentence.words]

    def _map_to_universal_tag(self, tag):
        """
        Map NLTK POS tag to Universal POS tag.

        Args:
            tag (str): The NLTK POS tag.

        Returns:
            str: The corresponding Universal POS tag.
        """
        tag_mapping = {
            'CC': 'CONJ',
            'CD': 'NUM',
            'DT': 'DET',
            'EX': 'ADV',
            'FW': 'X',
            'IN': 'ADP',
            'JJ': 'ADJ',
            'JJR': 'ADJ',
            'JJS': 'ADJ',
            'LS': 'X',
            'MD': 'VERB',
            'NN': 'NOUN',
            'NNS': 'NOUN',
            'NNP': 'PROPN',
            'NNPS': 'PROPN',
            'PDT': 'DET',
            'POS': 'PRT',
            'PRP': 'PRON',
            'PRP$': 'PRON',
            'RB': 'ADV',
            'RBR': 'ADV',
            'RBS': 'ADV',
            'RP': 'PRT',
            'SYM': 'X',
            'TO': 'PRT',
            'UH': 'INTJ',
            'VB': 'VERB',
            'VBD': 'VERB',
            'VBG': 'VERB',
            'VBN': 'VERB',
            'VBP': 'VERB',
            'VBZ': 'VERB',
            'WDT': 'DET',
            'WP': 'PRON',
            'WP$': 'PRON',
            'WRB': 'ADV',
            ',': 'PUNCT',
            '.': 'PUNCT',
            ':': 'PUNCT',
            '(': 'PUNCT',
            ')': 'PUNCT',
            '$': 'PUNCT',
            '#': 'PUNCT',
            '``': 'PUNCT',
            "''": 'PUNCT',
        }
        return tag_mapping.get(tag, 'X')  # Default to 'X' if tag is not found
