import pytest

from torchadv.nlp import POSTagger


@pytest.fixture
def input_text():
    return "The quick brown fox jumps over the lazy dog."

def test_nltk(input_text):
    tagger = POSTagger()
    tags = tagger.tag(input_text)
    assert tags == [('The', 'DET'), ('quick', 'ADJ'), ('brown', 'NOUN'), ('fox', 'NOUN'), ('jumps', 'VERB'), ('over', 'ADP'), ('the', 'DET'), ('lazy', 'ADJ'), ('dog', 'NOUN'), ('.', 'PUNCT')]

def test_stanza(input_text):
    tagger = POSTagger(library="stanza")
    tags = tagger.tag(input_text)
    assert tags == [('The', 'DET'), ('quick', 'ADJ'), ('brown', 'ADJ'), ('fox', 'NOUN'), ('jumps', 'VERB'), ('over', 'ADP'), ('the', 'DET'), ('lazy', 'ADJ'), ('dog', 'NOUN'), ('.', 'PUNCT')]
