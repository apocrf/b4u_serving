import os
from ast import literal_eval
import numpy as np
import uvicorn

from fastapi import FastAPI
from dotenv import load_dotenv
from utils import loader


load_dotenv()
MODEL_KEY = os.environ.get("MODEL_KEY")

model = loader.loader("model", MODEL_KEY)  # type: ignore

# fill redis
loader.data_s3_redis("id_title_mapping_data", "id_title_mapping_data.parquet")
loader.data_s3_redis("full_id_mapping", "full_id_mapping.parquet")
loader.data_s3_redis("vectorised_data", "description_vectorized.parquet")

app = FastAPI()


@app.get("/")
async def root():
    """Root test"""
    return {"message": "Welcome to book recommendation system"}


@app.get("/v1/recommend")
async def recommend(index: int) -> str:
    """Recommend endpoint

    Returns
    -------
    str
        book vector
    """
    recommendation = []
    vector = np.array(
        literal_eval(loader.finder(index, "vectorised_data")), dtype=np.float16
    )

    nn_prediction = model.kneighbors(vector.reshape(1, -1), n_neighbors=6)
    indices = nn_prediction[1]
    indices = indices.flatten()
    book_indices_list = indices[-5:].tolist()

    for book_index in book_indices_list:
        recommendation.append(loader.finder(book_index, "id_title_mapping_data"))
    recommendation = [literal_eval(book)[0] for book in recommendation]
    return str(recommendation)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
