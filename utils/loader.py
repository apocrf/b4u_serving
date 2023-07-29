import os
import tempfile
import redis  # type: ignore
import boto3  # type: ignore
import joblib  # type: ignore
from typing import Literal
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
S3_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")
MODEL_KEY = os.environ.get("MODEL_KEY")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_REGION = os.environ.get("AWS_REGION")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
DOCKER_MLFLOW_S3_ENDPOINT_URL = os.environ.get("DOCKER_MLFLOW_S3_ENDPOINT_URL")
S3_DATA_BUCKET_NAME = os.environ.get("S3_DATA_BUCKET_NAME")
DATA_KEY = os.environ.get("DATA_KEY")
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT")
REDIS_DB = os.environ.get("REDIS_DB")
REDIS_KEY = "vectorised_data"

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
print(redis_client)


def loader(load_type: Literal["model", "data"]):
    """Universal s3 loader

    Parameters
    ----------
    load_type : Literal[&quot;model&quot;, &quot;data&quot;]
        model or data

    Returns
    -------
    Any
        object from s3
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=DOCKER_MLFLOW_S3_ENDPOINT_URL,  # ,MLFLOW_S3_ENDPOINT_URL
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    # Download data from MinIO
    match load_type:
        case "model":
            with tempfile.TemporaryFile() as f:
                s3.download_fileobj(
                    Fileobj=f,
                    Bucket=S3_BUCKET_NAME,
                    Key=MODEL_KEY,
                )
                f.seek(0)
                data = joblib.load(f)
        case "data":
            with tempfile.TemporaryFile() as f:
                s3.download_fileobj(
                    Fileobj=f,
                    Bucket=S3_DATA_BUCKET_NAME,
                    Key=DATA_KEY,
                )
                f.seek(0)
                data = pd.read_parquet(f)
        case _:
            raise TypeError("Wrong type")
    return data


def finder(id: int):
    vector = redis_client.hget(REDIS_KEY, id)
    return vector.decode()
