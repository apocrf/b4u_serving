import os
import boto3  # type: ignore
import uvicorn
import tempfile
import joblib
import numpy as np

from fastapi import FastAPI
from dotenv import load_dotenv


load_dotenv()
S3_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")
MODEL_KEY = os.environ.get("MODEL_KEY")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_REGION = os.environ.get("AWS_REGION")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL")

app = FastAPI()


@app.get("/")
async def root():
    """_summary_

    Returns:
        _type_: _description_
    """
    return {"message": "Welcome to book recommendation system"}


@app.get("/v1/recommend")
def recommend() -> str:
    """Recommend endpoint

    Returns
    -------
    str
        book vector
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    samples = np.ones(768)
    samples = samples.reshape(1, -1)

    # Download the model from MinIO
    with tempfile.TemporaryFile() as f:
        s3.download_fileobj(
            Fileobj=f,
            Bucket=S3_BUCKET_NAME,
            Key=MODEL_KEY,
        )
        f.seek(0)
        model = joblib.load(f)

    nn_prediction = model.kneighbors(samples, n_neighbors=6)
    return str(nn_prediction)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
