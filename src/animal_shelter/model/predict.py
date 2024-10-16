from animal_shelter.data_loader import load_data
from animal_shelter.features import add_features
from animal_shelter.model.default_features import DefaultFeatures

test_data = load_data("../data/test.csv")
with_features = add_features(test_data)
X_test = with_features[DefaultFeatures.cat_features + DefaultFeatures.num_features]
clf_model.predict(X_test)
