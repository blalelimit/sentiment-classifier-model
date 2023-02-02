from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *
from collections import defaultdict


# convert list of lists to list
def flatten(lst):
    return [item for sublist in lst for item in sublist]


# convert list to list of lists
def extract_digits(lst):
    return [[el] for el in lst]


# remove whitespace between @ and name
def fix_mention(entry):
    pattern = re.compile(r'@ ')
    return re.sub(pattern, '@', entry)


# remove hyperlinks with whitespace in between
def fix_hyperlink(entry):
    pattern = re.compile(r'https?(\s|)([!:])(\s|)/(\s|)(/|)(\s|)\S+(\s|).(\s|)\S+(\s|)(/|)(\s|)\S+')
    return re.sub(pattern, '', entry)


# Tagging words if it is noun, adjective, verb, or adverb
def tagging_map():
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    return tag_map


class PreProcess:
    def __init__(self, dataframe, hashtags):
        self.dataframe = dataframe
        self.hashtags = hashtags

    def get_sentiment(self):
        return self.dataframe

    # Change all the text to lower case
    def lower(self):
        self.dataframe['text_lower'] = self.dataframe['text'][0].lower()
        return self.dataframe

    # Remove whitespace between mentions and hyperlinks with whitespace in between
    def cleaning_a(self):
        self.dataframe['text_cleaned'] = fix_mention(self.dataframe['text_lower'][0])
        self.dataframe['text_cleaned'] = fix_hyperlink(self.dataframe['text_cleaned'][0])
        return self.dataframe

    # Remove mentions, hyperlinks, and symbols
    def cleaning_b(self):
        self.dataframe['text_cleaned'] = re.sub('@[A-Za-z0-9_]+', '', self.dataframe['text_cleaned'][0])
        self.dataframe['text_cleaned'] = re.sub('https?([!:])//\\S+', '', self.dataframe['text_cleaned'][0])
        self.dataframe['text_cleaned'] = re.sub('[^A-Za-z0-9_ \\r\\n]+', '', self.dataframe['text_cleaned'][0])
        return self.dataframe

    # Tokenization process
    def tokenization(self):
        self.dataframe['text_tokenized'] = [word_tokenize(entry) for entry in self.dataframe['text_cleaned']]
        return self.dataframe

    # Lemmatization using WordNetLemmatizer()
    def lemmatization(self):
        tag_map = tagging_map()
        lemmatized = []
        word_lemmatized = WordNetLemmatizer()

        # Removal of stop words and unnecessary text, then lemmatization
        for word, tag in pos_tag(self.dataframe['text_tokenized'][0]):
            if word not in stopwords.words('english') and (word.isalpha() or word in self.hashtags):
                words = word_lemmatized.lemmatize(word, tag_map[tag[0]])
                lemmatized.append(words)
        self.dataframe.loc[0, 'text_preprocessed'] = str(lemmatized)
        return self.dataframe
