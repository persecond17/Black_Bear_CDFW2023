import pandas as pd
import string
from collections import defaultdict
import re
import spacy
from spacy.matcher import Matcher
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy.cli
spacy.cli.download("en_core_web_lg")


def clear_text(df):
    """
    Preprocess the content of tweets by removing emojis,
    punctuation marks, and other special characters, and
    normalize the text. Add the preprocessed content to a list.
    """
    contents = df.content.values
    clear_text = []
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002500-\U00002BEF"  # chinese char
                                        "]+", flags=re.UNICODE)

    for line in contents:
        line = [c for c in line if c not in string.punctuation]
        line = ''.join(line).lower().replace("’", "'").replace('\n', ' ')
        line = regrex_pattern.sub(r' ', line)
        clear_text.append(line)
    return clear_text


def stemmer(words):
    """
    Stem each word using the Porter stemming algorithm.
    Return a list of tuples with the original word and
    its stemmed version.
    """
    stemmer = PorterStemmer()
    stemmed = [[w, stemmer.stem(w)] for w in words]
    return stemmed


def count_words(stemmed):
    """
    Take a list of stemmed words, remove stop words, count
    the frequency of each word. Return a DataFrame with columns
    for the number of words, root words, and sets of words that
    have the same root.
    """
    freq = defaultdict(list)
    for w in stemmed:
        k, v = w
        if k not in ENGLISH_STOP_WORDS:
            freq[v].append(k)
    temp = sorted([(k, set(v), len(v)) for k,v in freq.items()], key=lambda x:-x[2])
    temp_df = pd.DataFrame(temp, columns=['root_word', 'words', 'count', ])
    return temp_df


def pos_tagging(contents):
    """
    Extract nouns and verbs, and the corresponding indexes
    of verbs containing any form of the key word 'bear'.
    """
    nlp = spacy.load('en_core_web_lg')
    verb_matcher = Matcher(nlp.vocab)
    verb_matcher.add("verb", [[{"POS": "VERB"}]])
    noun_list, verb_list = [], []

    for doc in nlp.pipe(contents, disable=["ner"]): # reduce execution time
        noun_list += [n.root.text for n in doc.noun_chunks if len(n.root.text)>1]
        verb_matches = verb_matcher(doc)
        verb_phrase = [doc[start:end].text for _, start, end in verb_matches]
        verb_list += [verb for verb in verb_phrase if verb!='bear' and len(verb) > 1]

    return noun_list, verb_list
