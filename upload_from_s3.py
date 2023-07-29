import os
from dotenv import load_dotenv
import redis
import pandas as pd
import pickle
import json
from utils import loader

load_dotenv()
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT")
REDIS_DB = os.environ.get("REDIS_DB")
REDIS_KEY = "vectorised_data"

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

data = loader.loader("data")

# Store the dataframe in Redis Hashes
for index, row in data.iterrows():
    row_as_list = row.to_list()
    row_as_json = json.dumps(row_as_list)
    redis_client.hset(REDIS_KEY, index, row_as_json)
