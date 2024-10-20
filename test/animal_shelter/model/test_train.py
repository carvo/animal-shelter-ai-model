from animal_shelter.model import train
from animal_shelter.paths import DefaultPaths


#TODO real unit test
def test_train_model():
    model = train.train("data/train.csv", DefaultPaths.OUTPUT_PATH / "test_animal_model.gz")

    assert model is not None
