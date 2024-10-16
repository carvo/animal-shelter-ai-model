from animal_shelter.data_loader import load_data
from animal_shelter.features import add_features

test_data = load_data("../data/test.csv")
with_features = add_features(test_data)
X_test = with_features[cat_features + num_features]
clf_model.predict(X_test)
