from pathlib import Path

from animal_shelter.data_loader import load_data
from animal_shelter.features import add_features

project_root_path = Path(__file__).parent.parent.parent

def main():
    print("----------- Started ----------- ")

    csv_file = project_root_path / "data/train.csv"
    animal_outcomes = load_data(csv_file.__str__())

    new = add_features(animal_outcomes)
    print(new.head().to_string())
    # print(new)

    print("----------- Finished -----------")


if __name__ == "__main__":
    main()
