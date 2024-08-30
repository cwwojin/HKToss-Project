from pymongo import MongoClient
import pickle
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


def _fetch_data():
    connection_string = os.environ.get("MONGODB_URL")

    if not connection_string:
        raise ValueError("MONGO_CONNECTION_STRING is not set in the environment.")

    client = MongoClient(connection_string)

    try:
        database = client["mongodatabase"]
        collection = database["dataset_final"]

        last_id = None
        batch_size = 10000
        batch_num = 0  # 배치 번호를 추가하여 파일이 덮어씌워지지 않도록

        while True:
            query = {} if last_id is None else {"_id": {"$gt": last_id}}
            batch = list(collection.find(query).sort("_id").limit(batch_size))

            if not batch:
                break  # 더 이상 가져올 데이터가 없으면 종료

            last_id = batch[-1]["_id"]  # 마지막으로 가져온 문서의 _id

            df = pd.DataFrame(batch)
            df = df.drop(columns=["_id"])

            # .cache 디렉토리 생성 (존재하지 않는 경우)
            cache_dir = "/opt/airflow/.cache"
            os.makedirs(cache_dir, exist_ok=True)

            # 배치 데이터를 각각 저장하여 메모리 사용을 줄임
            cache_file_path = os.path.join(
                cache_dir, f"train_data_cache_batch_{batch_num}.pkl"
            )
            df.to_pickle(cache_file_path)
            batch_num += 1

            print(f"Processed {len(batch)} records, saved to {cache_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        client.close()


if __name__ == "__main__":
    _fetch_data()
