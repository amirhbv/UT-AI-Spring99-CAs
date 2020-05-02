from code import Classifier
from time import time
import pandas as pd

start = time()
classifier = Classifier(
    data_file_name='./data.csv',
    classification_cols=['short_description', 'headline'],
    category_col='category',
    oversample=True,
)
print("Elapsed Time (train):", time() - start)

start = time()
confustion_matrix = classifier.evaluate(
    classification_col='short_description',
    categories=["TRAVEL", "BUSINESS"],
)
print("Elapsed Time (evaluate phase1):", time() - start)

start = time()
confustion_matrix = classifier.evaluate(
    classification_col='short_description',
)
print("Elapsed Time (evaluate phase2):", time() - start)

start = time()
res = classifier.classify(
    test_file_name='./test.csv',
    classification_col='short_description',
)
print("Elapsed Time (classify):", time() - start)

res[['index', 'category']].to_csv('output.csv', index=False)
pd1 = pd.read_csv('output.csv')
pd2 = pd.read_csv('ans.csv')
joined = pd.merge(pd1, pd2, on='index')
joined = joined[joined['category_x'] != joined['category_y']]
print(joined.shape)
