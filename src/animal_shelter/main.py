from fastapi import FastAPI
from pathlib import Path
from animal_shelter.FileLoader import load_data

app = FastAPI()
project_root_path = Path(__file__).parent.parent.parent

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/animals")
async def animals_file_head(limit: int = 5):
    csv_file = project_root_path / 'data/train.csv'
    return load_data(csv_file.__str__()).head(limit)
