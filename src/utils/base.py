import os
import json
import yaml

CHAT_HISTORY_DIR = "../logs/chat_logs"


MESSAGE_LOG_DIR = "../logs/message_logs"
if not os.path.exists(MESSAGE_LOG_DIR):
    os.makedirs(MESSAGE_LOG_DIR)

def save_messages_log(messages_to_save, output_file='messages_log.jsonl'):
    """전체 메시지 로그를 JSONL 파일로 저장합니다."""
    filepath = os.path.join(MESSAGE_LOG_DIR, output_file)
    with open(filepath, "w", encoding="utf-8") as f:
        for message in messages_to_save:
            json.dump(message, f, ensure_ascii=False)
            f.write('\n')
    print(f"Messages log saved to {filepath}")


#### Chat Functions ####

# 채팅 기록 저장 
def save_chat_history(history, filename):
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{filename}.jsonl")
    with open(filepath, "w", encoding="utf-8") as f:
        for item in history:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    return filename

def save_chat_history_to_jsonl(history, filename):
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{filename}.jsonl")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False) # ensure_ascii로 한글이 깨지지 않게 저장
        f.write("\n") # json을 쓰는 것과 같지만, 여러 줄을 써주는 것이므로 "\n"을 붙여준다.
    return filename

# 이전 채팅 기록 불러오기 
def load_chat_history(filename):
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{filename}")
    history = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                history.append(json.loads(line))
    except FileNotFoundError:
        pass
    return history


### Convert to JSON ### 
def convert_to_json_for_raw(USER_ID,
                          USER_NAME,
                          thread_id,
                          chat_name,
                          num_chat_turn,
                          user_preference_reflected,
                          re_recommendation_allowed,
                          raw_data,
                          num_previous_chat_history=0):
    # content (message) and corresponding metadata in json
    #print(f'raw_data: \n{raw_data}')
    #json_shaped_data = raw_data.to_json()['kwargs']
    raw_data = raw_data['supervisor_agent']['messages']
    json_shaped_data = []
    for idx, message in enumerate(raw_data):
        # num_previous_chat_history가 5라면, idx 4까지는 패스 
        # idx 5부터 새로운 user_input이다.  
        if idx < num_previous_chat_history:
            continue
        json_shaped_data.append(message.to_json()['kwargs'])
    #print(f'json shaped message and metadata:\n {json_shaped_data}')
    # Add is_final_response to the last message
    json_shaped_data[-1]['is_final_response'] = True

    json_data = {'user_id': USER_ID,
                'user_name': USER_NAME,
                'thread_id': thread_id,
                'chat_name': chat_name,
                'num_chat_turn': num_chat_turn,
                'user_preference_reflected': user_preference_reflected,
                're_recommendation_allowed': re_recommendation_allowed,
                'messages': json_shaped_data}    
    return json_data   



def extract_current_query(raw_data):
    # 맨 첫번째 메시지 묶음의 마지막만 리턴
    messages_cluster = raw_data[0]
    print(f'---- extract latest query process')
    print(f'messages:\n{messages_cluster}')
    messages_cluster['messages'] = messages_cluster['messages'][-1]
    print(f'modified messages:\n{messages_cluster}')
    return [messages_cluster]


def convert_to_json_for_raw_messages(USER_ID,
                          USER_NAME,
                          thread_id,
                          chat_name,
                          num_chat_turn,
                          user_preference_reflected,
                          re_recommendation_allowed,
                          raw_data,
                          num_previous_chat_history=0):
    # content (message) and corresponding metadata in json
    print(f'\noutput_to_save -----\n raw_data: type: {type(raw_data)}, length: {len(raw_data)}')
    print(raw_data)
    #json_shaped_data = raw_data.to_json()['kwargs']
    #raw_data = raw_data['supervisor_agent']['messages']
    #json_shaped_data = raw_data[num_previous_chat_history:]
    #json_shaped_data = raw_data[num_chat_turn-1:]
    #lastest_query = extract_current_query(raw_data)
    #json_shaped_data = raw_data[1:] + lastest_query
    json_shaped_data = raw_data
    #print(f'\nlen of json_shaped_data: {len(json_shaped_data)}\n')
    #print(f'\njson_shaped_data:\n{json_shaped_data}')
    # Add is_final_response to the last message
    json_shaped_data[-1]['is_final_response'] = True

    json_data = {'user_id': USER_ID,
                'user_name': USER_NAME,
                'thread_id': thread_id,
                'chat_name': chat_name,
                'num_chat_turn': num_chat_turn,
                'user_preference_reflected': user_preference_reflected,
                're_recommendation_allowed': re_recommendation_allowed,
                'messages': json_shaped_data}    
    return json_data   



### Models Config ###

def load_models_config(filepath="config/current_models.yaml"):
    #print(f'utils base path: {os.getcwd()}')
    """
    YAML 파일에서 LLM 모델 설정을 로드합니다.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f'config: {config}')
        return config
    except FileNotFoundError:
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
        return []
    except yaml.YAMLError as e:
        print(f"오류: YAML 파일을 파싱하는 중 문제가 발생했습니다: {e}")
        return []


def get_subagent_llm_model(llm_models_config,
                           supervisor_llm_model='gpt-4.1-mini-2025-04-14'):
    for llm_model_config in llm_models_config:
        supervisor_llm_models = llm_model_config.get('supervisor_models')
        subagent_model = llm_model_config.get('subagent_model')
        llm_type = llm_model_config.get('type')
        if supervisor_llm_model in supervisor_llm_models:
            return subagent_model
    return None


