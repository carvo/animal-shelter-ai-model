from fastapi import FastAPI
from pathlib import Path
from animal_shelter.data_loader import load_data
import logging

app = FastAPI()
project_root_path = Path(__file__).parent.parent.parent
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)-15s] %(name)s - %(levelname)s - %(message)s",
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/animals")
async def animals_file_head(limit: int = 5):
    csv_file = project_root_path / "data/train.csv"
    return load_data(csv_file.__str__()).head(limit)
