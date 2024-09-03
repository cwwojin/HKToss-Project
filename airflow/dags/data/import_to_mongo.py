from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def _import_data(**kwargs):
    connection_string = os.environ.get("MONGODB_URL")

    if not connection_string:
        raise ValueError("MONGO_CONNECTION_STRING is not set in the environment.")

    client = MongoClient(connection_string)

    try:
        database = client["mongodatabase"]
        collection = database["dataset_main"]

        csv_file_path = kwargs["csv_file_path"]
        unique_key = kwargs["unique_key"]
        print(csv_file_path)
        print(unique_key)

        # CSV 파일 읽기
        df = pd.read_csv(csv_file_path)

        # DataFrame에서 중복된 unique_key 제거
        df = df.drop_duplicates(subset=[unique_key])

        # 이미 존재하는 unique_key를 가져오기
        existing_keys = set(
            doc[unique_key] for doc in collection.find({unique_key: {"$in": df[unique_key].tolist()}}, {unique_key: 1})
        )

        # MongoDB에 없는 unique_key를 가진 문서만 선택
        new_documents = df[~df[unique_key].isin(existing_keys)]

        # DataFrame을 딕셔너리 목록으로 변환
        documents_to_insert = new_documents.to_dict(orient='records')

        if documents_to_insert:
            # Bulk insert 문서들
            result = collection.insert_many(documents_to_insert)
            print(f"Inserted {len(result.inserted_ids)} documents.")
        else:
            print("No new documents to insert.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        client.close()