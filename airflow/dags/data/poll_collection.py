import time
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

def poll_mongo_collection(**kwargs):
    connection_string = os.environ.get("MONGODB_URL")
    if not connection_string:
        raise ValueError("MONGODB_URL이 환경 변수에 설정되지 않았습니다.")
    
    client = MongoClient(connection_string)
    database = client["mongodatabase"]
    collection = database["dataset_main"]

    last_checked_id = kwargs[""]
    # polling_interval = 10  # 폴링 간격 (초)

    try:
        # last_checked_id 이후의 문서만 조회
        query = {} if last_checked_id is None else {"_id": {"$gt": last_checked_id}}
        new_documents = list(collection.find(query).sort("_id", 1))

        # 새로운 문서가 있을 경우
        if new_documents:
            for doc in new_documents:
                print(f"새로운 문서가 감지되었습니다: {doc}")
                last_checked_id = doc["_id"]  # 마지막으로 확인된 _id 업데이트
        else:
            print("새로운 문서가 없습니다.")

        return last_checked_id  # 마지막으로 확인된 _id 반환

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return last_checked_id  # 오류 발생 시에도 마지막 _id 반환

    finally:
        client.close()
        
if __name__ == "__main__":
    poll_mongo_collection()