import os
from ast import literal_eval
from uuid import UUID, uuid4
import json
import numpy as np

import uvicorn
from pydantic import BaseModel, Field  # pylint: disable=E0611
from fastapi import FastAPI
from dotenv import load_dotenv
from utils import loader
from db import store_recs

load_dotenv()
MODEL_KEY = os.environ.get("MODEL_KEY")
model = loader.loader("model", MODEL_KEY)  # type: ignore

# fill redis
# loader.data_s3_redis("id_title_mapping_data", "id_title_mapping_data.parquet")
# loader.data_s3_redis("full_id_mapping", "full_id_mapping.parquet")
# loader.data_s3_redis("vectorised_data", "description_vectorized.parquet")
# loader.data_s3_redis("title_id_mapping_data", "title_id_mapping_data.parquet")


app = FastAPI()


class Book(BaseModel):
    author: str
    title: str


class Response(BaseModel):
    tg_id: int
    uid: str = Field(default_factory=lambda: str(uuid4()))
    books: list[Book]


@app.get("/")
async def root():
    """Root test"""
    return {"message": "Welcome to book recommendation system"}


@app.get("/v1/recommend", response_model=Response)
async def recommend(tg_id: int, liked_book: str | int) -> Response:
    """Recommend endpoint

    Parameters
    ----------
    tg_id : int
        Telegram id
    liked_book : str | int
        Recommendation base book title

    Returns
    -------
    Response
    """

    books_list = []

    try:
        int(liked_book)
        index = liked_book
    except ValueError:
        index = loader.finder(book_index=liked_book, redis_key="title_id_mapping_data")
        index = int(index[1:-1])  # type: ignore

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
    response = Response(tg_id=tg_id, books=books)

    response_data = json.loads(response.json())
    print(response_data)

    store_recs.db_interaction(
        db=store_recs.SessionLocal(),
        tg_id=response_data["tg_id"],
        liked_book=str(liked_book),
        books=response_data["books"][0],
        uid=response_data["uid"],
    )

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
