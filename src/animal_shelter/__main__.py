from animal_shelter.feature.enhancer import add_features
from animal_shelter.helper.data_loader import load_data
from animal_shelter.paths import DefaultPaths


def main():
    print("----------- Started ----------- ")

    csv_file = DefaultPaths.DATA_PATH / "train.csv"
    animal_outcomes = load_data(csv_file)

    new = add_features(animal_outcomes)
    print(new.head().to_string())
    # print(new)

    print("----------- Finished -----------")


if __name__ == "__main__":
    main()
