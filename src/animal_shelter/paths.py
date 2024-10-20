from pathlib import Path


class DefaultPaths:
    PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent
    DATA_PATH = PROJECT_ROOT_PATH / "data"
    OUTPUT_PATH = PROJECT_ROOT_PATH / "output"
    ANIMAL_MODEL_PATH = OUTPUT_PATH / "animal_model.gz"
