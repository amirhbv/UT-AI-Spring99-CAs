import math
from collections import Counter
from random import random

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from prettytable import PrettyTable

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
        self.category_probability = category_total_rows / data_total_rows

        self.words = Counter()
        self.word_count = 0

    def add_text(self, text):
        for word in clean_text(text):
            self.words[word] += 1
            self.word_count += 1

    def calculate_probability(self, text):
        p = math.log10(self.category_probability)
        for word in clean_text(text):
            p += math.log10((self.words[word] or 0.1) / self.word_count)

        return p


class Classifier:

    def __init__(self, data_file_name, classification_cols, category_col, oversample=False):
        self._category_col = category_col
        self._classification_cols = classification_cols

        df = pd.read_csv(data_file_name,
                         usecols=[*classification_cols, category_col])

        df.dropna(inplace=True)

        self.total_rows, _ = df.shape
        category_titles = df[category_col].unique()

        df = df.groupby(category_col)
        max_category_data_rows = max([
            categpry_data_rows
            for categpry_data_rows, _ in [
                category.shape for _, category in df
            ]
        ])
        print(max_category_data_rows)

        self.categories = {}
        self.test_data = {}
        for category_title in category_titles:
            category_df = df.get_group(category_title)
            category_total_rows, _ = category_df.shape

            if oversample:
                category_df = category_df.append(
                    category_df.sample(
                        n=max_category_data_rows - category_total_rows,
                        replace=True,
                    )
                )

            category_total_rows, _ = category_df.shape
            category = Category(
                category_title, category_total_rows, self.total_rows)

            test_data = []
            for _, row in category_df.iterrows():
                if random() < 0.8:
                    for col in classification_cols:
                        category.add_text(row[col])
                else:
                    test_data.append(row)

            self.test_data[category_title] = pd.DataFrame(
                columns=category_df.columns,
                data=test_data,
            )

            self.categories[category_title] = category

    def _find_category(self, text, include_categories=None):
        _, result_category = max([
            (category.calculate_probability(text), category.category_title)
            for category in self.categories.values()
            if not include_categories or category.category_title in include_categories
        ])

        return result_category

    def evaluate(self, classification_col, categories=None):
        valid_categories = list(self.categories.keys())
        if not categories:
            categories = valid_categories
        else:
            categories = [
                category for category in categories if category in valid_categories]

        actual_categories, predicted_categories = zip(*[
            (category, self._find_category(
                row[classification_col], categories))
            for category in categories
            for _, row in self.test_data[category].iterrows()
        ])

        category_indices = {
            category: index for (index, category) in enumerate(categories)
        }

        confustion_matrix = [[0 for _ in categories] for _ in categories]
        for actual_category, predicted_category in zip(actual_categories, predicted_categories):
            confustion_matrix[
                category_indices[actual_category]
            ][
                category_indices[predicted_category]
            ] += 1

        confustion_matrix_table = PrettyTable(
            field_names=[
                'Confusion Matrix',
                *categories
            ],
        )
        for i, row in enumerate(confustion_matrix):
            confustion_matrix_table.add_row([
                categories[i],
                *row,
            ])

        print(confustion_matrix_table)

        metrics_table = PrettyTable(
            field_names=[
                '',
                'Accuracy',
                'Precision',
                'Recall',
            ],
        )

        all_true_positive_cases = sum(
            confustion_matrix[i][i] for i in range(len(confustion_matrix)))
        all_cases = len(actual_categories)
        accuracy = all_true_positive_cases / all_cases

        for i, category in enumerate(categories):
            true_positive_cases = confustion_matrix[i][i]
            actual_positive_cases = sum(
                confustion_matrix[i][j] for j in range(len(confustion_matrix)))
            predicted_positive_cases = sum(
                confustion_matrix[j][i] for j in range(len(confustion_matrix)))

            precision = true_positive_cases / predicted_positive_cases
            recall = true_positive_cases / actual_positive_cases

            metrics_table.add_row([
                category,
                accuracy,
                precision,
                recall,
            ])

        print(metrics_table)

        return confustion_matrix

    def classify(self, test_file_name, classification_col):
        df = pd.read_csv(test_file_name, usecols=[
                         'index', classification_col]).dropna()

        df[self._category_col] = df[classification_col].apply(
            self._find_category)

        return df
