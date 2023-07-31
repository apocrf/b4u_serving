import os
import tempfile
import json
from typing import Any, Literal
import redis  # type: ignore
import boto3  # type: ignore
import joblib  # type: ignore
import pandas as pd
from urllib.parse import unquote
from dotenv import load_dotenv

load_dotenv()
S3_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")
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


redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
print(redis_client)


def loader(load_type: Literal["model", "data"], data_path: str) -> Any:
    """Universal s3 loader

    Parameters
    ----------
    load_type : Literal[model, data]
        model or data

    Returns
    -------
    Any
        data object from s3
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
                    Key=data_path,
                )
                f.seek(0)
                data = joblib.load(f)
        case "data":
            with tempfile.TemporaryFile() as f:
                s3.download_fileobj(
                    Fileobj=f,
                    Bucket=S3_DATA_BUCKET_NAME,
                    Key=data_path,
                )
                f.seek(0)
                data = pd.read_parquet(f)
        case _:
            raise TypeError("Wrong type")
    return data


def data_s3_redis(redis_key: str, data_path: str):
    """Store the dataframe in Redis Hashes

    Parameters
    ----------
    redis_key : str
        key
    data_path : str
        s3 data path
    """
    data = loader("data", data_path)
    try:
        for index, row in data.iterrows():
            row_as_list = row.to_list()
            row_as_json = json.dumps(row_as_list)
            redis_client.hset(redis_key, index, row_as_json)
    except ValueError:
        print("s3->redis transer failed")


def finder(
    book_index: int | str,
    redis_key: Literal[
        "vectorised_data",
        "id_title_mapping_data",
        "full_id_mapping",
        "title_id_mapping_data",
    ],
):
    if isinstance(book_index, str):
        vector = redis_client.hget(
            redis_key, str(unquote(book_index).strip('"'))
        ).decode()
    else:
        vector = redis_client.hget(redis_key, book_index).decode()
    return vector
