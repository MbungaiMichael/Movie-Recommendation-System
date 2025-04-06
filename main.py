from pydantic import BaseModel
from fastapi import FastAPI
from app.model.source_code import *


# Load the trained pipeline
file_directory = r"C:\Users\UltraBook 3.1\PycharmProjects\Movie_rec2\cluster_pipeline_v1.0.0.pkl"
pipeline = joblib.load(file_directory)

app = FastAPI()


# Input model for prediction
class MovieData(BaseModel):
    tag: str
    genres: str
    rating: float
    mean: float
    count: int


@app.get("/")
def home():
    return {"Health Check": "OK", "model_version": version}


@app.post("/predict_cluster")
def predict_cluster(data: MovieData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Transform and predict using the pipeline
    transformed = pipeline.transform(input_df)
    cluster = transformed['cluster'].iloc[0]

    return {
        "cluster": int(cluster),
        "status": "success"
    }
