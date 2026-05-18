import sklearn

data_X, data_y = sklearn.datasets.make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=2,
    random_state=42,
    n_clusters_per_class=1
)

scaler = sklearn.preprocessing.StandardScaler()

data_X = scaler.fit_transform(data_X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    data_X, data_y,
    test_size=0.3,
    stratify=data_y,
    random_state=42
)

