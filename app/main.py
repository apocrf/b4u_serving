from typing import Any
import uvicorn
import ast
import numpy as np
from utils import loader
from fastapi import FastAPI


model = loader.loader("model")

app = FastAPI()


@app.get("/")
async def root():
    """Root test"""
    return {"message": "Welcome to book recommendation system"}


@app.get("/v1/recommend")
async def recommend(index: int) -> Any:
    """Recommend endpoint

    Returns
    -------
    str
        book vector
    """

    vector = np.array(ast.literal_eval(loader.finder(index)), dtype=np.float16)

    nn_prediction = model.kneighbors(vector.reshape(1, -1), n_neighbors=6)
    return str(nn_prediction)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
