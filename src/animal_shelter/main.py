import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile

from animal_shelter.helper.data_loader import load_data
from animal_shelter.model.domain import ListAnimalPrediction, AnimalPrediction
from animal_shelter.model.predict import predict_file as pf, predict_json as pj, predict_json_list as pjl
from animal_shelter.model.train import train
from animal_shelter.paths import DefaultPaths


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ~Path.exists(DefaultPaths.ANIMAL_MODEL_PATH):
        train(DefaultPaths.DATA_PATH / "train.csv", DefaultPaths.ANIMAL_MODEL_PATH)
        LOG.info("model trained and saved")  # not working :P

    yield
    LOG.info("do nothing")


app = FastAPI(lifespan=lifespan)
LOG = logging.getLogger(__name__)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/animals/train-data")
async def animals_file_head(limit: int = 5):
    LOG.info("calling /animals/train-data")
    csv_file = DefaultPaths.DATA_PATH / "train.csv"
    return load_data(csv_file).head(limit)


@app.post("/predictions/file")
async def create_upload_file(file: UploadFile):
    data = await file.read()
    return pf(data, DefaultPaths.ANIMAL_MODEL_PATH).to_dict(orient="records")


@app.post("/predictions/json")
async def predict_json(pred_data: AnimalPrediction):
    return pj(pred_data, DefaultPaths.ANIMAL_MODEL_PATH).to_dict(orient="records")


@app.post("/predictions/json-list")
async def predict_json_list(pred_data: ListAnimalPrediction):
    return pjl(pred_data, DefaultPaths.ANIMAL_MODEL_PATH).to_dict(orient="records")
