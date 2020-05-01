from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def get_wordnet_pos(nltk_tag):
    return {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }.get(nltk_tag[0].upper(), None)


class Category:

    def __init__(self, category_title, total_rows):
        self.category_title = category_title
        self.total_rows = total_rows
        self.words = Counter()

    def add_text(self, text):
        for sent in nltk.sent_tokenize(text):
            self.add_sentence(sent)

    def add_sentence(self, sentence):
        tagged_sentence = [
            (word, get_wordnet_pos(tag))
            for (word, tag) in nltk.pos_tag(nltk.regexp_tokenize(sentence.lower(), r'\w+'))
            if len(word) > 1 and word not in stop_words
        ]

        for word, tag in tagged_sentence:
            if tag:
                word = lemmatizer.lemmatize(word, tag)
            else:
                word = lemmatizer.lemmatize(word)

            self.words[word] += 1


class Classifier:

    def __init__(self, data_file_name, classification_cols, category_col):
        self._category_col = category_col

        df = pd.read_csv(data_file_name,
                         usecols=[*classification_cols, category_col])

        df.dropna(inplace=True)

        self.total_rows, _ = df.shape
        category_titles = df[category_col].unique()

        df = df.groupby(category_col)

        self.data_categories = {}
        for category_title in category_titles:
            category_df = df.get_group(category_title)
            category_total_rows, _ = category_df.shape
            category = Category(category_title, category_total_rows)

            for index, row in category_df.iterrows():
                for col in classification_cols:
                    category.add_text(row[col])

            self.data_categories[category_title] = category
