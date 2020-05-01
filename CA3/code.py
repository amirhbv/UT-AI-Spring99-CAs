import math
from collections import Counter
from random import random

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


def clean_text(text):
    result = []
    for sentence in nltk.sent_tokenize(text):
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

            result.append(word)

    return result


class Category:

    def __init__(self, category_title, category_total_rows, data_total_rows):
        self.category_title = category_title
        self.category_probability = math.log(
            category_total_rows / data_total_rows, 10)

        self.words = Counter()
        self.word_count = 0

    def add_text(self, text):
        for word in clean_text(text):
            self.words[word] += 1
            self.word_count += 1

    def calculate_probability(self, text):
        p = self.category_probability
        for word in clean_text(text):
            p += math.log(self.words[word] / self.word_count or 1e-6, 10)

        return p


class Classifier:

    def __init__(self, data_file_name, classification_cols, category_col):
        self._category_col = category_col

        df = pd.read_csv(data_file_name,
                         usecols=[*classification_cols, category_col])

        df.dropna(inplace=True)

        self.total_rows, _ = df.shape
        category_titles = df[category_col].unique()

        df = df.groupby(category_col)

        self.categories = {}
        self.test_data = {}
        for category_title in category_titles:
            category_df = df.get_group(category_title)
            category_total_rows, _ = category_df.shape
            category = Category(
                category_title, category_total_rows, self.total_rows)

            self.test_data[category_title] = []

            for index, row in category_df.iterrows():
                if random() < 0.8:
                    for col in classification_cols:
                        category.add_text(row[col])
                else:
                    self.test_data[category_title].append(row)

            self.categories[category_title] = category

    def _find_category(self, text):
        _, result_category = max([
            (category.calculate_probability(text), category.category_title)
            for category in self.categories.values()
        ])

        return result_category

    def classify(self, test_file_name, classification_col):
        df = pd.read_csv(test_file_name, usecols=[
                         'index', classification_col]).dropna()

        df[self._category_col] = df[classification_col].apply(
            self._find_category)

        return df
