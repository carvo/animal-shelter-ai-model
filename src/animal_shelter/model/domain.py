from pydantic import BaseModel
from datetime import datetime


class AnimalPrediction(BaseModel):
    id: int
    name: str | None = None
    date_time: datetime
    animal_type: str
    sex_upon_outcome: str
    age_upon_outcome: str
    breed: str
    color: str


class ListAnimalPrediction(BaseModel):
    predictions: list[AnimalPrediction]
