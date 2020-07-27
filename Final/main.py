from collections import Counter
from math import sqrt

import hazm
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
from nltk.corpus import wordnet
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)
from sklearn.svm import SVR
from unidecode import unidecode

normalizer = hazm.Normalizer(persian_numbers=False)
post_tagger = hazm.POSTagger(model='./resources-0.5/postagger.model')
lemmatizer = hazm.InformalLemmatizer()
stopwords = hazm.stopwords_list()


def clean_text(text):
    text = text.replace('\n', ' ').replace('/', ' ').replace('آ', 'ا').lower()
    result = []
    for sentence in hazm.sent_tokenize(normalizer.normalize(text)):
        tagged_sentence = [
            (word, tag)
            for (word, tag) in post_tagger.tag(hazm.word_tokenize(sentence))
            if len(word) > 1 and word not in stopwords
        ]

        for word, tag in tagged_sentence:
            if tag:
                word = lemmatizer.lemmatize(word, tag)
            else:
                word = lemmatizer.lemmatize(word)

            result.append(unidecode(word))

    return result


def vectorize_text_column(df, column_name):
    df[column_name] = df[column_name].apply(clean_text)
    word_counter = Counter()
    for _, words in df[column_name].items():
        word_counter.update(words)

    new_columns = []
    most_common_words = [word for word, _ in word_counter.most_common(200)]
    for word in most_common_words:
        new_column_name = f'has_{word}_in_{column_name}'
        new_columns.append(new_column_name)
        df[new_column_name] = df[column_name].apply(
            lambda x: 1 if x.count(word) > 0 else 0)

    return new_columns


COLUMN_TARGET = 'price'
COLUMN_BRAND = 'brand'
COLUMN_CITY = 'city'
COLUMN_TITLE = 'title'
COLUMN_DESCRIPTION = 'desc'
COLUMN_IMAGE_COUNT = 'image_count'
INVALID_PRICE_THRESHOLD = 50000

df = pd.read_csv('./mobile_phone_dataset.csv')

print(
    f'There are {len(df[df[COLUMN_TARGET] <= INVALID_PRICE_THRESHOLD])} rows with invalid price.')

df = df[df[COLUMN_TARGET] > 0]

df[COLUMN_BRAND] = df[COLUMN_BRAND].apply(lambda brand: brand.split('::')[0])

mean_price_by_brand = df.groupby([COLUMN_BRAND]).mean()[COLUMN_TARGET]
df[COLUMN_TARGET] = df.apply(
    lambda row:
        row[COLUMN_TARGET] if row[COLUMN_TARGET] > INVALID_PRICE_THRESHOLD else mean_price_by_brand[row[COLUMN_BRAND]],
    axis=1,
)
target = df[COLUMN_TARGET]

title_columns = vectorize_text_column(df, COLUMN_TITLE)
description_columns = vectorize_text_column(df, COLUMN_DESCRIPTION)

BRANDS = sorted(df[COLUMN_BRAND].unique())
CITIES = sorted(df[COLUMN_CITY].unique())

features_df = pd.DataFrame(
    data=ColumnTransformer(
        transformers=[
            (
                'cleaner',
                'drop',
                [
                    'Unnamed: 0',
                    'created_at',
                    COLUMN_IMAGE_COUNT,
                    COLUMN_TITLE,
                    COLUMN_DESCRIPTION,
                    COLUMN_TARGET,
                ]
            ),
            (
                'scaler',
                MinMaxScaler(),
                [
                    COLUMN_IMAGE_COUNT,
                ]
            )
            (
                'encoder',
                OneHotEncoder(),
                [
                    COLUMN_BRAND,
                    COLUMN_CITY,
                ]
            ),
        ],
        remainder='passthrough',
        sparse_threshold=0,
    ).fit_transform(df),
    columns=[
        COLUMN_IMAGE_COUNT,
        *BRANDS,
        *CITIES,
        *title_columns,
        *description_columns,
    ],
).infer_objects()

x_train, x_test, y_train, y_test = train_test_split(features_df, target)


def calculate_metrics(y_true, y_pred):
    return {
        'mae': metrics.mean_absolute_error(y_true, y_pred),
        'rmse': sqrt(metrics.mean_squared_error(y_true, y_pred)),
    }


def evaluate_model(Model, **kwargs):
    model = Model(**kwargs).fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    return Model.__name__, {
        'train': calculate_metrics(y_train, y_train_pred),
        'test': calculate_metrics(y_test, y_test_pred),
    }


def find_best_hyper_param_index(res):
    _, best_param_index = min(
        [(v, i) for i, v in enumerate(list(map(lambda x: x[1]['test']['rmse'], res)))]
    )
    return best_param_index


print(evaluate_model(LinearRegression))

ridge_res = [
    evaluate_model(Ridge, solver="sag", random_state=42, alpha=alpha)
    for alpha in [1, 2, 3, 3.5, 4, 4.5, 5, 6, 7]
]
print(ridge_res[find_best_hyper_param_index(ridge_res)])

# print(evaluate_model(SVR, C=0.3, max_iter=2000))
print(evaluate_model(LGBMRegressor, subsample=0.9))

params = {
    'learning_rate': uniform(0, 1),
    'n_estimators': sp_randint(200, 1500),
    'num_leaves': sp_randint(20, 200),
    'max_depth': sp_randint(2, 15),
    'min_child_weight': uniform(0, 2),
    'colsample_bytree': uniform(0, 1),
}
best_params = RandomizedSearchCV(
    estimator=LGBMRegressor(subsample=0.9), param_distributions=params, n_iter=10, cv=3, random_state=42,
    scoring='neg_root_mean_squared_error', verbose=10, return_train_score=True, n_jobs=-1
).fit(x_train, y_train).best_params_

print(best_params)
print(evaluate_model(LGBMRegressor, **best_params,
                     subsample=0.9, random_state=42, n_jobs=-1))
