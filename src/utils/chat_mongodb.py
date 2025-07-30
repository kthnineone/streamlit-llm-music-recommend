import pymongo
from pymongo import DESCENDING
import json
import time
from datetime import datetime

# --- 2. MongoDB 설정 ---
# MongoDB 서버 정보 (필요에 따라 수정)
MONGO_URI = 'mongodb://localhost:27017/'
MONGO_DB_NAME = 'llm_responses_db'
MONGO_COLLECTION_NAME = 'llm_messages'

'''
pymongo.MongoClient: MongoDB 서버에 연결하기 위한 클라이언트 객체를 생성합니다. MONGO_URI를 서버 URI에 맞게 변경하세요.
mongo_client[MONGO_DB_NAME]: 특정 데이터베이스에 접근합니다.
mongo_db[MONGO_COLLECTION_NAME]: 특정 컬렉션에 접근합니다.

'''

class MongoDBClient():
    def __init__(self):
        try:
            mongo_client = pymongo.MongoClient(MONGO_URI)
            self.mongo_db = mongo_client[MONGO_DB_NAME]
            #mongo_collection = mongo_db[MONGO_COLLECTION_NAME]
            print("MongoDB 서버에 성공적으로 연결되었습니다.")
            #return mongo_db
            self.raw_messages_collection = self.init_collection(mongo_collection_name='raw_messages')
            self.messages_for_user_collection = self.init_collection(mongo_collection_name='messages_for_user')
        except pymongo.errors.ConnectionFailure as e:
            print(f"MongoDB 서버 연결 실패: {e}")
            print("MongoDB 서버가 실행 중인지 확인해 주세요.")
            exit()


    def init_collection(self, mongo_collection_name=MONGO_COLLECTION_NAME):
        try:
            mongo_collection = self.mongo_db[mongo_collection_name]
            print(f"MongoDB의 Collection: {mongo_collection_name}에 성공적으로 연결되었습니다.")
            return  mongo_collection
        except pymongo.errors.ConnectionFailure as e:
            print(f"MongoDB 서버의 컬렉션 {mongo_collection_name} 연결 실패: {e}")
            print("MongoDB 서버가 실행 중인지 혹은 컬렉션이 정확한지 확인해 주세요.")
            exit()

    def insert_one_document(self, collection, document):
        """지정된 컬렉션에 여러 문서를 배치로 삽입합니다."""
        if not document:
            print("삽입할 문서가 없습니다.")
            return

        try:
            # insertMany() 메서드를 사용하여 여러 문서를 한 번에 삽입합니다.
            result = collection.insert_one(document)
            print(f"\nLLM 응답이 MongoDB에 영구 저장되었습니다. (Document ID: {result.inserted_id})")

            # MongoDB에서 데이터 확인 (선택 사항)
            print("\nMongoDB에서 가져온 데이터:")
            for doc in collection.find().limit(1): # 최근 저장된 1개 문서 확인
                print(doc)
            return result.inserted_id
        
        except pymongo.errors.BulkWriteError as e:
            print(f"문서 삽입 중 오류가 발생했습니다: {e.details}")
            # 오류 발생 시 어떤 문서가 삽입되었는지 확인할 수 있습니다.
            if e.details and 'writeErrors' in e.details:
                for error in e.details['writeErrors']:
                    print(f"  오류가 발생한 인덱스: {error['index']}, 메시지: {error['errmsg']}")
            return None
    
    def insert_many_documents(self, collection, documents):
        """지정된 컬렉션에 여러 문서를 배치로 삽입합니다."""
        if not documents:
            print("삽입할 문서가 없습니다.")
            return

        try:
            # insertMany() 메서드를 사용하여 여러 문서를 한 번에 삽입합니다.
            result = collection.insert_many(documents)
            print(f"총 {len(result.inserted_ids)}개의 문서가 성공적으로 삽입되었습니다.")
            print(f"삽입된 문서 ID: {result.inserted_ids}")
            return result.inserted_ids
        except pymongo.errors.BulkWriteError as e:
            print(f"문서 삽입 중 오류가 발생했습니다: {e.details}")
            # 오류 발생 시 어떤 문서가 삽입되었는지 확인할 수 있습니다.
            if e.details and 'writeErrors' in e.details:
                for error in e.details['writeErrors']:
                    print(f"  오류가 발생한 인덱스: {error['index']}, 메시지: {error['errmsg']}")
            return None
        
    def insert_raw_message(self, document):
        print('---- Raw Messages 저장 ----')
        print(f'document: {document}')
        self.insert_one_document(self.raw_messages_collection, document)

    def insert_message_for_user(self, document):
        print('---- Messages For Users, Chat History 저장 ----')
        self.insert_one_document(self.messages_for_user_collection, document)
        
    def insert_raw_messages(self, documents):
        print('---- Raw Messages 저장 ----')
        self.insert_many_documents(self.raw_messages_collection, documents)

    def insert_messages_for_user(self, documents):
        print('---- Messages For Users, Chat History 저장 ----')
        self.insert_many_documents(self.messages_for_user_collection, documents)

    
    def insert_update_messages(self, collection, chat_data: dict, is_for_user=False):
        try:
            redis_key = chat_data['redis_key']

            # 기존 채팅 데이터 조회
            existing_chat = collection.find_one({"redis_key": redis_key})

            # 기존 채팅이 있을 경우
            if existing_chat:
                print(f"Chat with redis_key '{redis_key}' already exists.")
                """start_time, end_time, latency는 싱글턴 대화에서의 내역
                멀티턴 대화의 경우 start_time, end_time, latency를 리스트로 저장"""
                print("Messages have changed. Updating 'messages', 'num_chat_turn', 'conversation_start_time', 'conversation_end_time', 'conversation_latency', and 'updated_at' fields.")

                updated_messages = chat_data.get('messages', []) # 'messages'가 없으면 빈 리스트 
                if is_for_user: # messages_for_user 컬렉션의 경우 
                    if updated_messages:
                        updated_messages = updated_messages
                    else:
                        updated_messages = existing_chat['messages']
                else: # raw_messages 컬렉션의 경우 
                    if updated_messages:
                        updated_messages = existing_chat['messages'] + updated_messages
                    else:
                        updated_messages = existing_chat['messages']
                updated_num_chat_turn = chat_data.get('num_chat_turn', existing_chat['num_chat_turn'])

                updated_start_time = existing_chat['conversation_start_time'] + chat_data.get('conversation_start_time', [])
                updated_end_time = existing_chat['conversation_end_time'] + chat_data.get('conversation_end_time', [])
                updated_latency = existing_chat['conversation_latency'] + chat_data.get('conversation_latency', [])

                update_fields = {
                    "messages": updated_messages,
                    "num_chat_turn": updated_num_chat_turn,
                    'conversation_start_time': updated_start_time,
                    'conversation_end_time': updated_end_time,
                    'conversation_latency': updated_latency,
                    "updated_at": time.time() #datetime.now()
                }
                result = collection.update_one(
                    {"redis_key": redis_key},
                    {"$set": update_fields}
                )
                if result.matched_count > 0:
                    print(f"Successfully updated redis_key: {redis_key}")
                else:
                    print(f"Failed to update redis_key: {redis_key}")
            # 새로운 채팅일 경우
            else:
                print(f"New chat with redis_key '{redis_key}'. Inserting new document.")
                # 'created_at'이 없으면 현재 시간으로 설정
                if 'created_at' not in chat_data:
                    chat_data['created_at'] = time.time() #datetime.now()
                # 'updated_at'도 없으면 'created_at'과 동일하게 설정
                if 'updated_at' not in chat_data:
                    chat_data['updated_at'] = chat_data['created_at']

                result = collection.insert_one(chat_data)
                print(f"Successfully inserted new chat with id: {result.inserted_id}")

        except Exception as e:
            print(f"An error occurred: {e}")
        #finally:
        #    if 'client' in locals() and client:
        #        client.close()

    def insert_update_raw_messages(self, chat_data: dict):
        print('---- Raw Messages 저장 in MongoDB----')
        self.insert_update_messages(self.raw_messages_collection, chat_data)

    def insert_update_messages_for_user(self, chat_data: dict):
        print('---- Messages for User 저장 in MongoDB ----')
        self.insert_update_messages(self.messages_for_user_collection, chat_data, is_for_user=True)


    def count_document(self, collection, target_user_id):
        # 쿼리 조건 정의
        query = {"user_id": target_user_id}

        print(f"Searching for documents with user_id: '{target_user_id}'...")

        # 1. 특정 user_id를 가진 문서 가져오기
        # find() 메서드는 커서(cursor) 객체를 반환합니다.
        # 실제 문서는 커서를 반복할 때 가져와집니다.
        #matching_documents_cursor = collection.find(query)

        # 커서에서 모든 문서를 리스트로 변환하여 저장
        # (주의: 문서가 매우 많을 경우 메모리 문제가 발생할 수 있으므로,
        # 필요한 경우 커서를 직접 반복하여 처리하는 것이 좋습니다.)
        #matching_documents = list(matching_documents_cursor)

        # 2. 특정 user_id를 가진 문서의 개수 파악
        # count_documents() 메서드를 사용하는 것이 가장 효율적입니다.
        # 이는 서버에서 직접 개수를 세어 반환합니다.
        document_count = collection.count_documents(query)
        return document_count
    
    def count_document_by_user(self, target_user_id):
        return self.count_document(self.raw_messages_collection, target_user_id)
    
    def load_recent_data_by_user(self, collection, target_user_id, limit=5):
        # 3. 특정 user_id를 만족하는 문서를 찾고, 'updated_at' 필드를 기준으로
        #    내림차순 정렬하며, 상위 limit 개수만큼 가져오기
        # find({"user_id": user_id}) 부분에서 특정 user_id를 조건으로 필터링합니다.
        query = {"user_id": target_user_id}
        
        recent_documents = list(
            collection.find(query)
            .sort("updated_at", DESCENDING)
            .limit(limit)
        )
        return recent_documents
    
    def load_recent_data(self, target_user_id, limit=5):
        return self.load_recent_data_by_user(self.messages_for_user_collection, target_user_id, limit)
    
    def get_redis_keys_from_recent_data(self, recent_data):
        redis_key_list = []
        for data in recent_data:
            redis_key = data['redis_key']
            redis_key_list.append(redis_key)
        return redis_key_list




