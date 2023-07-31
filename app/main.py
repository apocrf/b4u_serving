import os
from ast import literal_eval
import numpy as np

import uvicorn
from pydantic import BaseModel  # pylint: disable=E0611
from fastapi import FastAPI
from dotenv import load_dotenv

from utils import loader

app = FastAPI()


load_dotenv()
MODEL_KEY = os.environ.get("MODEL_KEY")

model = loader.loader("model", MODEL_KEY)  # type: ignore

# fill redis
loader.data_s3_redis("id_title_mapping_data", "id_title_mapping_data.parquet")
loader.data_s3_redis("full_id_mapping", "full_id_mapping.parquet")
loader.data_s3_redis("vectorised_data", "description_vectorized.parquet")


class Book(BaseModel):
    author: str
    title: str


@app.get("/")
async def root():
    """Root test"""
    return {"message": "Welcome to book recommendation system"}


@app.get("/v1/recommend", response_model=list[Book])
async def recommend(index: int) -> list[Book]:
    """Recommend endpoint

    Returns
    -------
    str
        book vector
    """
    books_list = []
    vector = np.array(
        literal_eval(loader.finder(index, "vectorised_data")), dtype=np.float16
    )

    nn_prediction = model.kneighbors(vector.reshape(1, -1), n_neighbors=6)
    indices = nn_prediction[1]
    indices = indices.flatten()
    book_indices_list = indices[-5:].tolist()

    for book_index in book_indices_list:
        books_list.append(loader.finder(book_index, "full_id_mapping"))
    books_t = {literal_eval(book)[0]: literal_eval(book)[1] for book in books_list}
    books = [Book(author=author, title=title) for author, title in books_t.items()]
    return books


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
