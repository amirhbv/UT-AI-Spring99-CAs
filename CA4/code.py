import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

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
target = df[COLUMN_TARGET] == 'Yes'

COUNTRIES = df[COLUMN_COUNTRY].unique()

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

plt.figure(figsize=(50, 5))
plt.plot(
    list(mutual_info.keys()),
    list(mutual_info.values()),
)

for col, gain in mutual_info.items():
    print(col, gain)


X_train, X_test, y_train, y_test = train_test_split(features_df, target)


def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'recall': metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def test_classifier(Classifier, **kwargs):
    clf = Classifier(**kwargs).fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    return {
        'train': calculate_metrics(y_train, y_train_pred),
        'test': calculate_metrics(y_test, y_test_pred),
    }


def plot_metrics(title, param_title, res):
    fig, (accuracy_ax, recall_ax, precision_ax) = plt.subplots(nrows=1, ncols=3)

    fig.suptitle(title)
    fig.set_figheight(5)
    fig.set_figwidth(25)

    accuracy_ax.plot(
        range(1, 50),
        list(map(lambda x: x['train']['accuracy'], res)),
        list(map(lambda x: x['test']['accuracy'], res)),
    )

    recall_ax.plot(
        range(1, 50),
        list(map(lambda x: x['train']['recall'], res)),
        list(map(lambda x: x['test']['recall'], res)),
    )

    precision_ax.plot(
        range(1, 50),
        list(map(lambda x: x['train']['precision'], res)),
        list(map(lambda x: x['test']['precision'], res)),
    )

    accuracy_ax.legend(['Train', 'Trest'])
    accuracy_ax.set_xlabel(param_title)
    accuracy_ax.set_ylabel('Accuracy')
    recall_ax.set_ylabel('Recall')
    precision_ax.set_ylabel('Precision')


def test_ensemble_classifier(Classifier, **kwargs):
    clf = Classifier(**kwargs).fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print('train', metrics.classification_report(y_train, y_train_pred))
    print('test', metrics.classification_report(y_test, y_test_pred))


def find_most_accurate_hyper_param(res):
    _, best_param = max(
        [(v, i + 1) for i, v in enumerate(list(map(lambda x: x['test']['accuracy'], res)))])

    return best_param


dt_res = [
    test_classifier(DecisionTreeClassifier, max_depth=i)
    for i in range(1, 50)
]
plot_metrics(
    DecisionTreeClassifier.__name__,
    'max_depth',
    dt_res,
)

most_accurate_depth = find_most_accurate_hyper_param(dt_res)
print("Most accurate Decision Tree Depth:", most_accurate_depth)
test_ensemble_classifier(DecisionTreeClassifier, max_depth=most_accurate_depth)


knn_res = [
    test_classifier(KNeighborsClassifier, n_neighbors=i)
    for i in range(1, 50)
]
plot_metrics(
    KNeighborsClassifier.__name__,
    'n_neighbors',
    knn_res,
)

most_accurate_n_neighbors = find_most_accurate_hyper_param(knn_res)
print("Most accurate Decision Tree Depth:", most_accurate_n_neighbors)
test_ensemble_classifier(KNeighborsClassifier,
                         n_neighbors=most_accurate_n_neighbors)


test_ensemble_classifier(LogisticRegression, max_iter=1000)


test_ensemble_classifier(
    BaggingClassifier,
    base_estimator=KNeighborsClassifier(n_neighbors=most_accurate_n_neighbors),
    max_samples=0.5,
    max_features=0.5,
    n_estimators=5,
)


test_ensemble_classifier(
    BaggingClassifier,
    base_estimator=DecisionTreeClassifier(max_depth=most_accurate_depth),
    max_samples=0.5,
    max_features=0.5,
    n_estimators=5,
)


test_ensemble_classifier(
    RandomForestClassifier,
    max_depth=most_accurate_depth,
    n_estimators=10,
    max_features=5,
)

plot_metrics(
    RandomForestClassifier.__name__,
    'max_depth',
    [
        test_classifier(
            RandomForestClassifier,
            max_depth=i,
            n_estimators=10,
            max_features=5,
        )
        for i in range(1, 50)
    ]
)


plot_metrics(
    RandomForestClassifier.__name__,
    'n_estimators',
    [
        test_classifier(
            RandomForestClassifier,
            max_depth=most_accurate_depth,
            n_estimators=i,
            max_features=5,
        )
        for i in range(1, 50)
    ]
)


test_ensemble_classifier(
    VotingClassifier,
    estimators=[
        (
            DecisionTreeClassifier.__name__,
            DecisionTreeClassifier(max_depth=most_accurate_depth),
        ),
        (
            KNeighborsClassifier.__name__,
            KNeighborsClassifier(n_neighbors=most_accurate_n_neighbors),
        ),
        (
            LogisticRegression.__name__,
            LogisticRegression(max_iter=1000),
        ),
    ],
    voting='hard',
)

best_dt_res = DecisionTreeClassifier(
    max_depth=most_accurate_depth).fit(X_train, y_train).predict(X_test)

best_knn_res = KNeighborsClassifier(
    n_neighbors=most_accurate_n_neighbors).fit(X_train, y_train).predict(X_test)

best_lr_res = LogisticRegression(
    max_iter=1000).fit(X_train, y_train).predict(X_test)

dt_knn = 0
dt_lr = 0
knn_lr = 0
dt_knn_lr = 0

for i in range(len(best_dt_res)):
    dt_knn += (best_dt_res[i] == best_knn_res[i])
    dt_lr += (best_dt_res[i] == best_lr_res[i])
    knn_lr += (best_knn_res[i] == best_lr_res[i])
    dt_knn_lr += (best_dt_res[i] == best_knn_res[i]
                  and best_knn_res[i] == best_lr_res[i])

print('len', len(best_dt_res))
print('dt_knn', dt_knn)
print('dt_lr', dt_lr)
print('knn_lr', knn_lr)
print('dt_knn_lr', dt_knn_lr)
