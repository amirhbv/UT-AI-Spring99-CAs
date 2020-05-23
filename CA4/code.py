import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

COLUMN_TARGET = 'Is Back'
COLUMN_COUNTRY = 'Country'
COLUMN_QUANTITY = 'Total Quantity'
COLUMN_PRICE = 'Total Price'
COLUMN_PURCHASE_COUNT = 'Purchase Count'
COLUMN_ID = 'Customer ID'
COLUMN_DATE = 'Date'
COLUMN_MONTH = 'Month'
COLUMN_DOW = 'DoW'

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday',
                'Thursday', 'Friday', 'Saturday', 'Sunday']

df = pd.read_csv('data.csv')
COUNTRIES = df[COLUMN_COUNTRY].unique()

target = df[COLUMN_TARGET] == 'Yes'

df[COLUMN_DATE] = pd.to_datetime(df[COLUMN_DATE])
df[COLUMN_MONTH] = df[COLUMN_DATE].dt.month
df[COLUMN_DOW] = df[COLUMN_DATE].dt.dayofweek

features_df = pd.DataFrame(
    data=ColumnTransformer(
        transformers=[
            (
                'cleaner',
                'drop',
                [
                    COLUMN_ID,
                    COLUMN_DATE,
                    COLUMN_TARGET
                ]
            ),
            (
                'encoder',
                OneHotEncoder(),
                [
                    COLUMN_COUNTRY,
                    COLUMN_MONTH,
                    COLUMN_DOW
                ]
            ),
        ],
        remainder='passthrough',
        sparse_threshold=0,
    ).fit_transform(df),
    columns=[
        *COUNTRIES, *MONTHS, *DAYS_OF_WEEK,
        COLUMN_QUANTITY, COLUMN_PRICE, COLUMN_PURCHASE_COUNT
    ],
).infer_objects()

mutual_info = dict(
    zip(
        list(features_df.columns.values),
        mutual_info_classif(features_df, target)
    )
)

X_train, X_test, y_train, y_test = train_test_split(features_df, target)


def test_classifier(Classifier, **kwargs):
    clf = Classifier(**kwargs).fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    return {
        'train': {
            'accuracy': metrics.accuracy_score(y_train, y_train_pred),
            'recall': metrics.recall_score(y_train, y_train_pred),
            'precision': metrics.precision_score(y_train, y_train_pred),
        },
        'test': {
            'accuracy': metrics.accuracy_score(y_test, y_test_pred),
            'recall': metrics.recall_score(y_test, y_test_pred),
            'precision': metrics.precision_score(y_test, y_test_pred),
        },
    }


dt_res = []
for i in range(1, 100):
    dt_res.append(test_classifier(DecisionTreeClassifier, max_depth=i))

plt.plot(list(map(lambda x: x['train']['accuracy'], dt_res)))
plt.plot(list(map(lambda x: x['test']['accuracy'], dt_res)))

plt.show()

plt.plot(list(map(lambda x: x['train']['recall'], dt_res)))
plt.plot(list(map(lambda x: x['test']['recall'], dt_res)))

plt.show()

plt.plot(list(map(lambda x: x['train']['precision'], dt_res)))
plt.plot(list(map(lambda x: x['test']['precision'], dt_res)))

plt.show()


for i in range(1, 100):
    test_classifier(KNeighborsClassifier, n_neighbors=i)

print(test_classifier(LogisticRegression, max_iter=1000))
