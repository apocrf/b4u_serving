import os
import boto3
import uvicorn
import sklearn
from fastapi import FastAPI
from dotenv import load_dotenv


load_dotenv()
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
MODEL_KEY = os.environ.get("MODEL_KEY")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_REGION = os.environ.get("AWS_REGION")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL")

app = FastAPI()


@app.get("/recommend")
def recommend(input_books: list[str]) -> list[str]:
    """Recommend endpoint

    Parameters
    ----------
    input_books : list[str]
        books input

    Returns
    -------
    list[str]
        recommend books
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
        logged_model=MODEL_KEY,
    )

    # Download the model from MinIO
    obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=MODEL_KEY)
    model_data = obj["Body"].read()

    # Load model
    nn_loaded_model = sklearn.load_model(model_data)
    nn_prediction = nn_loaded_model.kneighbors(input_books, n_neighbors=6)
    nn_prediction = list(nn_prediction[1][0][1:])

    return {"recommend": nn_prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
