import redis
import json
import time
from datetime import datetime

# --- 1. Redis 설정 ---
# Redis 서버 정보 (필요에 따라 수정)
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

'''
redis.StrictRedis: Redis 서버에 연결하기 위한 클라이언트 객체를 생성합니다. host, port, db를 서버 설정에 맞게 변경하세요.
decode_responses=True: Redis에서 가져온 데이터를 문자열로 디코딩하도록 설정합니다.
ping(): Redis 서버에 연결되었는지 확인합니다.
'''

class RedisClient:

    def __init__(self):
        try:
            redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
            redis_client.ping()
            print("Redis 서버에 성공적으로 연결되었습니다.")
            self.db = redis_client
        except redis.exceptions.ConnectionError as e:
            print(f"Redis 서버 연결 실패: {e}")
            print("Redis 서버가 실행 중인지 확인해 주세요.")
            exit()
        '''
        finally:
            # --- 연결 종료 ---
            if 'redis_client' in locals() and redis_client:
                redis_client.close()
                print("Redis 클라이언트 연결이 성공적으로 종료되었습니다.")
        '''

    def save_into_redis(self, json_data, redis_key=None):
        # --- Redis에 임시 저장 ---
        # 데이터를 JSON 문자열로 변환하여 저장
        # Redis 키는 고유하게 식별할 수 있는 값
        if redis_key:
            pass
        else:
            # redis_key가 없으면 타임스탬프로 설정
            redis_key = f"llm_response:{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        try:
            key_already_in = self.db.keys('redis_key')
            if key_already_in:
                # Redis에서 데이터 로딩 
                retrieved_data = self.db.get(redis_key)
                created_at = json.loads(retrieved_data)['created_at']
                json_data['created_at'] = created_at
                json_data['updated_at'] = time.time()
            else:
                json_data['created_at'] = time.time()
                json_data['updated_at'] = time.time()

                self.db.set(redis_key, json.dumps(json_data))
            print(f"LLM 응답이 Redis에 임시 저장되었습니다. (키: {redis_key})")

            # Redis에서 데이터 확인 (선택 사항)
            #retrieved_data = self.db.get(redis_key)
            #if retrieved_data:
            #    print("\nRedis에서 가져온 데이터:")
            #    print(json.loads(retrieved_data))
        except Exception as e:
            print(f"Redis 저장 중 오류 발생: {e}")

    def load_from_redis(self, redis_key=None):
        if not redis_key:
            return None
        else:
            try:
                # Redis에서 데이터 확인
                retrieved_data = self.db.get(redis_key)
                if retrieved_data:
                    print(f"\nRedis에서 가져온 데이터: (키: {redis_key})")
                    retrieved_data = json.loads(retrieved_data)
                    print(retrieved_data)
                    return retrieved_data
            except Exception as e:
                print(f"Redis 저장 중 오류 발생: {e}")

    def load_previous_chats(self, keys=None):
        if not keys:
            all_keys = self.db.keys('*')
        else:
            all_keys = keys
        previous_chats = {}
        try:
            for redis_key in all_keys:
                # Redis에서 데이터 확인
                retrieved_data = self.db.get(redis_key)
                if retrieved_data:
                    print(f"\nRedis에서 가져온 데이터: (키: {redis_key})")
                    retrieved_data = json.loads(retrieved_data)
                    #print(retrieved_data)
                    previous_chats[redis_key] = retrieved_data
            return previous_chats
        except Exception as e:
            print(f"Redis에서 이전 채팅을 불러오는 중 오류 발생: {e}")





def convert_to_redis_json(USER_ID,
                          USER_NAME,
                          thread_id,
                          chat_name,
                          num_chat_turn,
                          user_preference_reflected,
                          re_recommendation_allowed,
                          raw_data):
    # content (message) and corresponding metadata in json
    #print(f'raw_data: \n{raw_data}')
    #json_shaped_data = raw_data.to_json()['kwargs']
    raw_data = raw_data['supervisor_agent']['messages']
    json_shaped_data = []
    for message in raw_data:
        json_shaped_data.append(message.to_json()['kwargs'])
    #print(f'json shaped message and metadata:\n {json_shaped_data}')

    redis_json = {'user_id': USER_ID,
                'user_name': USER_NAME,
                'thread_id': thread_id,
                'chat_name': chat_name,
                'num_chat_turn': num_chat_turn,
                'user_preference_reflected': user_preference_reflected,
                're_recommendation_allowed': re_recommendation_allowed,
                'messages': json_shaped_data}    
    return redis_json        

