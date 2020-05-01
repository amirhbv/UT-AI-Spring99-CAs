from code import Classifier

classifier = Classifier(
    data_file_name='./data.csv',
    classification_cols=['short_description', 'headline'],
    category_col='category',
)
