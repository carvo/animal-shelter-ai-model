import logging
from pathlib import Path
from animal_shelter.data_loader import load_data
from animal_shelter.features import add_features
from animal_shelter.DataEnhancerOLD import add_features as add_features_old

project_root_path = Path(__file__).parent.parent.parent
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)-15s] %(name)s - %(levelname)s - %(message)s",
)

def main():
    print("----------- Started ----------- ")

    csv_file = project_root_path / "data/train.csv"
    animal_outcomes = load_data(csv_file.__str__())

    new = add_features(animal_outcomes)
    print(new.head().to_string())
    # print(new)

    old = add_features_old(animal_outcomes)
    print(old.head().to_string())
    # print(old)

    print("----------- Finished -----------")


if __name__ == "__main__":
    main()
