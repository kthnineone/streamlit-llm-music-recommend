import os
import json
import time
import streamlit as st
from controllers.display_chat_list import *
import services.llm_langgraph as llm_langgraph
import services.common as common
from utils.logger import setup_logger, log_structured_data
from utils.chat_redis import *
from utils.chat_mongodb import *
from utils.base import *
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import (
    BaseMessage, 
    ChatMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    ToolMessage
)


# User_info
USER_ID = "user"
USER_NAME = "me"

#### Databases ####

# 파일 저장 경로 설정
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
OUTPUT_FILE = "chat_history.jsonl"

# 파일 저장 경로 설정 (각 채팅 기록을 별도의 파일로 저장)
CHAT_HISTORY_DIR = "../logs/chat_logs"
if not os.path.exists(CHAT_HISTORY_DIR):
    os.makedirs(CHAT_HISTORY_DIR)

redis_client = RedisClient()
mongodb_client = MongoDBClient()


#### Logging ####

logger = setup_logger(name='langgraph_logger', 
                      log_file='../logs/supervisor_stream.log')


### 현재 추천 시스템의 모델 구성 요소 ### 

current_models_config = load_models_config()
llm_models_config = current_models_config.get('llm_models', 'Empty')
embedding_models_config = current_models_config.get('embedding_models', 'Empty')
current_models_config = current_models_config.get('current_models', 'Empty')
st.session_state['llm_models_config'] = llm_models_config
st.session_state['embedding_models_config'] = embedding_models_config
st.session_state['current_models_config'] = current_models_config

# update subagent model

#subagent_llm_model = get_subagent_llm_model(llm_models_config,
#                                            supervisor_llm_model=selected_option)
#current_models_config['subagent_llm_model'] = subagent_llm_model

#### Streamlit session_state Initilization ####

st.session_state['user_id'] = USER_ID
st.session_state['user_name'] = USER_NAME


# 현재 채팅 기록을 저장할 세션 상태 변수
if 'current_chat_history' not in st.session_state:
    st.session_state['current_chat_history'] = []

# 이전 채팅 목록을 저장할 세션 상태 변수 (각 항목은 파일 이름)
# jsonl 파일들
previous_chats = os.listdir(CHAT_HISTORY_DIR)

# redis에서 불러온 이전 채팅 목록
previous_chats = redis_client.load_previous_chats()

# mongodb에서 불러온 이전 채팅 목록
previous_chats = mongodb_client.load_recent_data(USER_ID)
redis_keys_from_previous_chats = mongodb_client.get_redis_keys_from_recent_data(previous_chats)

st.session_state['previous_chats'] = previous_chats

if 'previous_chats' not in st.session_state:
    st.session_state['previous_chats'] = []

# 선택된 채팅 기록을 저장할 세션 상태 변수
if 'selected_chat_data' not in st.session_state:
    st.session_state['selected_chat_data'] = None

# 현재 표시할 페이지 상태를 저장하는 변수
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'main'  # 초기 페이지는 'main'

st.set_page_config(page_title="음악 추천 챗봇 💬", page_icon="💬")

# Name Generator 생성 함수
st.session_state["name_generator"] = llm_langgraph.create_name_generator()


# st.session_state 초기화
# 앱이 처음 로드될 때 변수를 초기화합니다.
if 'user_preference_reflected' not in st.session_state:
    st.session_state.user_preference_reflected = False # 기본값: 기존 선호도 미반영
if 're_recommendation_allowed' not in st.session_state:
    st.session_state.re_recommendation_allowed = True # 기본값: 재추천 허용. No Load Preference를 하기 위해서.

# AI, Human, System, Tool 모든 메시지 저장  
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def change_user_preference_reflection(preference):
    st.session_state['user_preference_reflected'] = preference

def change_re_recommendation_allowance(allowance):
    st.session_state['re_recommendation_allowed'] = allowance

# 현재 유저의 Chat 개수 가져오기 
#num_user_chats = load_num_chats()
num_user_chats = 6
num_user_chats = mongodb_client.count_document_by_user(USER_ID)
print(f'Initial num_user_chats: {num_user_chats}')

st.session_state['num_user_chats'] = num_user_chats

#### Chat Functions ####

# 현재 채팅에 메시지 추가 
def add_to_current_chat(role, content):
    st.session_state['current_chat_history'].append({"role": role, "content": content})

# 전체 메시지 로그에 기록 
def add_to_messages_log(role, messages, model=None):
    """AI, Human, System, Tool 메시지를 전체 메시지 로그에 추가합니다."""
    if model:
        st.session_state["messages"].append({"role": role, "content": messages, "model": model})
        print(f"Added to messages log: {role} (model: {model}) - {messages} ")
    else:
        st.session_state["messages"].append({"role": role, "content": messages})
        print(f"Added to messages log: {role} - {messages} ")



def filter_user_chat_content(raw_content):
    content = raw_content.replace("Allow re-recommendation of previously recommended songs.\n", "")
    content = content.replace("Do NOT allow re-recommendation of previously recommended songs.\n", "")
    content = content.replace("Incorporate the user's existing preferences.\n", "")
    content = content.replace("Do not incorporate the user's existing preferences.\n", "")
    return content

def filter_assistant_chat_content(raw_content):
    content = raw_content.replace("RECOMMENDATION_COMPLETE", "")
    #content = content.replace("temp", "")
    return content


# 이전 채팅 기록 보여주기 
def display_previous_chat_history_jsonl(history):
    print(f'---- history ----: \n{history}')
    for chat in history:
        for message in chat:
            if message["role"].lower() in ("user", "human"):
                chat_content = message["content"]
                chat_content = filter_user_chat_content(chat_content)
                st.chat_message("user").markdown(chat_content)
            elif message["role"].lower() in ("assistant", "ai", 'supervisor_agent'):
                chat_content = message["content"]
                if chat_content in ('Successfully transferred to web_search_agent', 'Successfully transferred to load_preference_agent'):
                    continue
                else:
                    chat_content = filter_assistant_chat_content(chat_content)
                    st.chat_message("assistant").markdown(chat_content)
            else:
                pass 
                #st.chat_message("system").markdown(chat_content)


# 이전 채팅 기록 보여주기 
def display_previous_chat_history_redis(history):
    print(f'---- history ----: \n{history}')
    for chat in history:
        if chat['type'].lower() in ('user', 'human'):
            chat_content = chat["content"]
            chat_content = filter_user_chat_content(chat_content)
            st.chat_message("user").markdown(chat_content)
        elif chat['type'].lower() in ('ai', 'assistant', ''):
            chat_content = chat["content"]
            if chat_content in ('Successfully transferred to web_search_agent', 'Successfully transferred to load_preference_agent', ''):
                continue
            else:
                chat_content = filter_assistant_chat_content(chat_content)
                st.chat_message("assistant").markdown(chat_content)
        else: # tool case
            pass 
            #st.chat_message("system").markdown(chat_content)

# 이전 채팅 기록 보여주기 
def display_previous_chat_history_mongo(history):
    print(f'---- history ----: \n{history}')
    for chat in history:
        role = chat['role'].lower()
        if role in ('user', 'human'):
            chat_content = chat["content"]
            chat_content = filter_user_chat_content(chat_content)
            st.chat_message("user").markdown(chat_content)
        elif role in ('ai', 'assistant', ''):
            chat_content = chat["content"]
            if chat_content in ('Successfully transferred to web_search_agent', 'Successfully transferred to load_preference_agent', ''):
                continue
            else:
                chat_content = filter_assistant_chat_content(chat_content)
                st.chat_message("assistant").markdown(chat_content)
        else: # tool case
            pass 
            #st.chat_message("system").markdown(chat_content)


# 이전 채팅 기록 보여주기 
def display_previous_chat_history(history, db_method='mongo'):
    if db_method.lower() in ('mongo', 'mongodb', 'mongo_db'):
        display_previous_chat_history_mongo(history)
    elif db_method.lower() in ('redis'):
        display_previous_chat_history_redis(history)
    else:
        display_previous_chat_history_jsonl(history)



# 현재 채팅 기록 보여주기 
def display_current_chat_history_(history):
    print(f'---- history ----: \n{history}')
    for chat in history:
        with st.chat_message(chat["role"]):
            if chat["role"] in ("user", "human"):
                chat_content = chat["content"]
                chat_content = filter_user_chat_content(chat_content)
                st.markdown(chat_content)
            elif chat["role"].lower() in ("assistant", "ai", 'supervisor_agent'):
                if chat_content in ('Successfully transferred to web_search_agent', 'Successfully transferred to load_preference_agent'):
                    continue
                else:
                    st.markdown(chat_content)
            else:
                pass 
                #st.markdown(message["content"])


def display_current_chat_history(history):
    print(f'---- history ----: \n{history}')
    for chat in history:
        if chat["role"] in ("user", "human"):
            chat_content = chat["content"]
            chat_content = filter_user_chat_content(chat_content)
            st.chat_message("user").markdown(chat_content)
        elif chat["role"].lower() in ("assistant", "ai", 'supervisor_agent'):
            chat_content = chat["content"]
            if chat_content in ('Successfully transferred to web_search_agent', 'Successfully transferred to load_preference_agent'):
                continue
            else:
                st.chat_message("assistant").markdown(chat_content)
        else:
            pass 
            #st.chat_message("system").markdown(chat["content"])



def display_main_chat_simple_chatbot():
    st.title("나만의 ChatGPT 💬")
    display_current_chat_history(st.session_state['current_chat_history'])
    #display_previous_chat_history(st.session_state['current_chat_history'])
    print(f"current model: {st.session_state['model']}")

    if "graph" not in st.session_state:
        st.session_state["graph"] = llm_langgraph.create_chat_graph(st.session_state["model"])

    # Make chat_history_id as thread_id
    # thread_id 초기화:
    # 1. 'thread_id'가 session_state에 없으면 새로 생성합니다.
    # 2. 'num_user_chats'도 session_state에 없으면 초기화합니다.
    #    (만약 num_user_chats가 전체 사용자 채팅 수를 나타낸다면, 앱 시작 시 0으로 초기화하고,
    #     새로운 채팅 세션이 시작될 때마다 증가시키는 로직이 필요합니다.)
    if 'thread_id' not in st.session_state:
        if 'num_user_chats' not in st.session_state:
            st.session_state['num_user_chats'] = 0 # 앱 시작 시 또는 첫 대화 시작 시 0으로 초기화
        st.session_state['num_user_chats'] += 1 # 새 대화 세션마다 1 증가
        st.session_state['thread_id'] = st.session_state['num_user_chats'] # 새 thread_id 할당
    
    thread_id = st.session_state['thread_id'] 
    config = RunnableConfig(
                recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
                configurable={"thread_id": thread_id},  # 스레드 ID 설정
            )
    init_log = {
        'user_id': USER_ID,
        'thread_id': thread_id
        }
    log_structured_data(logger, init_log, model_name=st.session_state["model"])
    full_response = ""

    if user_input := st.chat_input():
        # chat_history에 추가 
        add_to_current_chat("user", user_input)
        st.chat_message("user").write(user_input)
        print(f'\noriginal user_input: {user_input}')
        # 재추천 허용 여부 결정
        if st.session_state.re_recommendation_allowed == True:
            user_input = "Allow re-recommendation of previously recommended songs.\n" + user_input
        else:
            user_input = "Do NOT allow re-recommendation of previously recommended songs.\n" + user_input
        # 기존 선호도 반영 여부 결정  
        if st.session_state.user_preference_reflected == True:
            user_input = "Incorporate the user's existing preferences.\n" + user_input
        else:
            user_input = "Do not incorporate the user's existing preferences.\n" + user_input

        # 멀티턴 대화에서 유저의 input 순서 
        # display_main_chat이 처음 실행되고
        # 유저의 입력시 2번째 실행되기 때문에 +1이 2번 되어서 2가 나온다. 
        if 'current_num_chat_turn' not in st.session_state:
            st.session_state['current_num_chat_turn'] = 1
        else:
            st.session_state['current_num_chat_turn'] += 1
        print(f'\nmodified user_input: {user_input}')
        print(f'\nchat turn: {st.session_state['current_num_chat_turn']}, thread_id: {thread_id}')
        # 전체 메시지 로그에 사용자 입력 추가
        add_to_messages_log("user", user_input)

        chat_history = st.session_state['current_chat_history']
        num_previous_chat_history = len(chat_history) - 1
        # modified user_input으로 변경 
        chat_history[-1] = {"role": 'user', "content": user_input}
        
        # AI assistant 응답 생성 섹션 
        with st.chat_message("assistant"):
            chat_container = st.empty()
            status_container = st.empty()  # 상태 메시지를 표시할 컨테이너

            # LangGraph 실행 
            status_container.markdown('결과를 생성중...')
            for output in st.session_state["graph"].stream({"messages": chat_history},
                                                                       config=config):
                print(f'\noutput type: {type(output)}')
                output['num_chat_turn'] = st.session_state['current_num_chat_turn']
                print(f'\noutput:\n {output}')
                # 로그에 출력 내용 기록
                log_structured_data(logger, output, model_name=st.session_state["model"])
                if "chatbot" in output: # output.keys()는 모든 키를 포함하는 뷰 객체를 만들기에 오버헤드 발생 
                    #print(f'\noutput of chatbot type: {type(output['chatbot'])}')
                    print(f'\n----- output of chatbot -----: {output["chatbot"]}')
                    if isinstance(output["chatbot"], dict) and "messages" in output["chatbot"]:
                        last_message = output["chatbot"]["messages"][-1]
                        content = last_message.content
                        meta_data = last_message.response_metadata  # 메타데이터 정보가 있다면 가져오기
                        model_name = meta_data.get('model_name', None)
                        full_response += content
                        chat_container.markdown(full_response + "▌") # 스트리밍 효과를 위한 커서 표시
                        # 메시지 로그에 추가
                        add_to_messages_log("assistant", content, model_name)
                        
                        # session state의 'current_chat_history'에 추가 
                        #add_to_current_chat('assistant', content)
                elif "tools" in output:
                    #print(f'\noutput of chatbot type: {type(output['chatbot'])}')
                    print(f'\n----- output of tools -----: {output["tools"]}')
                    status_container.markdown('웹 검색 중...')
                    content = output["tools"]['messages'][-1].content
                    # 메시지 로그에 추가 
                    add_to_messages_log("tool", content)
                    continue
                elif "END" in output:
                    st.markdown("결과 생성 완료")
                    #st.session_state["graph"].reset()  # 그래프 상태 초기화
                    break

                # 상태 메시지 제거
                status_container.empty()

            # 최종 응답 결과 표시
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        # chat_history에 추가
        add_to_current_chat("assistant", full_response)
        print(f'\nchat_history: \n{chat_history}')
        
        # Assistant 세션 끝 
        # 전체 메시지 로그 저장
        save_messages_log(messages_to_save=st.session_state['messages'])
        # Redis에 저장  
        output_dict = dict(output)
        # 채팅 이름 생성 
        if 'chat_name' in st.session_state:
            chat_name = st.session_state["chat_name"]
        else:
            name_generator = st.session_state["name_generator"]
            chat_history_for_name = [user_input, full_response]
            chat_name = name_generator.invoke({"messages": chat_history_for_name})
            st.session_state["chat_name"] = chat_name
        # Tool을 포함한 전체 메시지, 모니터링용 
        raw_messages_json = convert_to_json_for_raw(USER_ID,
                                                USER_NAME,
                                                thread_id,
                                                chat_name,
                                                st.session_state['current_num_chat_turn'],
                                                st.session_state['user_preference_reflected'],
                                                st.session_state['re_recommendation_allowed'],
                                                output_dict)
        # metadata 추가 
        raw_messages_json.update(current_models_config)
        redis_key = f'{USER_ID}:{thread_id}'
        # Redis에 저장 
        redis_client.save_into_redis(raw_messages_json, redis_key)
        raw_messages_json['redis_key'] = redis_key
        # MongoDB에 저장 
        mongodb_client.insert_update_raw_messages(raw_messages_json)
        # 유저에게 보여주기 위한 대화 내역 
        messages_for_user_json = {'user_id': USER_ID,
                                'user_name': USER_NAME,
                                'thread_id': thread_id,
                                'chat_name': chat_name,
                                'redis_key': redis_key,
                                'messages': st.session_state['current_chat_history']}
        # MongoDB에 저장 
        mongodb_client.insert_update_messages_for_user(messages_for_user_json)





def display_main_chat():
    st.title("나만의 ChatGPT 💬")
    display_current_chat_history(st.session_state['current_chat_history'])
    #display_previous_chat_history(st.session_state['current_chat_history'])
    print(f"Display Main Chat, current model: {st.session_state['model']}")

    if "graph" not in st.session_state:
        st.session_state["graph"] = llm_langgraph.create_chat_graph(st.session_state["model"])

    # Make chat_history_id as thread_id
    # thread_id 초기화:
    # 1. 'thread_id'가 session_state에 없으면 새로 생성합니다.
    # 2. 'num_user_chats'도 session_state에 없으면 초기화합니다.
    #    (만약 num_user_chats가 전체 사용자 채팅 수를 나타낸다면, 앱 시작 시 0으로 초기화하고,
    #     새로운 채팅 세션이 시작될 때마다 증가시키는 로직이 필요합니다.)
    if 'thread_id' not in st.session_state:
        if 'num_user_chats' not in st.session_state:
            st.session_state['num_user_chats'] = 0 # 앱 시작 시 또는 첫 대화 시작 시 0으로 초기화
        st.session_state['num_user_chats'] += 1 # 새 대화 세션마다 1 증가
        st.session_state['thread_id'] = st.session_state['num_user_chats']# 새 thread_id 할당
    
    thread_id = st.session_state['thread_id'] 
    config = RunnableConfig(
                recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
                configurable={"thread_id": thread_id},  # 스레드 ID 설정
            )
    init_log = {
        'user_id': USER_ID,
        'thread_id': thread_id
        }
    log_structured_data(logger, init_log, model_name=st.session_state["model"])
    full_response = ""

    if user_input := st.chat_input():
        print(f'If user_input, Current Models Config: {st.session_state['current_models_config']}')
        # chat_history에 추가 
        add_to_current_chat("user", user_input)
        st.chat_message("user").write(user_input)
        print(f'\noriginal user_input: {user_input}')
        # 재추천 허용 여부 결정
        if st.session_state.re_recommendation_allowed == True:
            user_input = "Allow re-recommendation of previously recommended songs.\n" + user_input
        else:
            user_input = "Do NOT allow re-recommendation of previously recommended songs.\n" + user_input
        # 기존 선호도 반영 여부 결정  
        if st.session_state.user_preference_reflected == True:
            user_input = "Incorporate the user's existing preferences.\n" + user_input
        else:
            user_input = "Do not incorporate the user's existing preferences.\n" + user_input
        # 멀티턴 대화에서 유저의 input 순서 
        # display_main_chat이 처음 실행되고
        # 유저의 입력시 2번째 실행되기 때문에 +1이 2번 되어서 2가 나온다. 
        if 'current_num_chat_turn' not in st.session_state:
            st.session_state['current_num_chat_turn'] = 1
        else:
            st.session_state['current_num_chat_turn'] += 1
        print(f'\nmodified user_input: {user_input}')
        print(f'\nchat turn: {st.session_state['current_num_chat_turn']}, thread_id: {thread_id}')
        # 전체 메시지 로그에 사용자 입력 추가
        add_to_messages_log("user", user_input)

        chat_history = st.session_state['current_chat_history']
        num_previous_chat_history = len(chat_history) - 1
        # modified user_input으로 변경 
        chat_history[-1] = {"role": 'user', "content": user_input}
        
        # AI assistant 응답 생성 섹션 
        with st.chat_message("assistant"):
            chat_container = st.empty()
            status_container = st.empty()  # 상태 메시지를 표시할 컨테이너

            # LangGraph 실행 
            status_container.markdown('결과를 생성중...')
            for output in st.session_state["graph"].stream({"messages": chat_history},
                                                                       config=config):
                print(f'\noutput type: {type(output)}')
                output['num_chat_turn'] = st.session_state['current_num_chat_turn']
                print(f'\noutput:\n {output}')
                # 로그에 출력 내용 기록
                log_structured_data(logger, output, model_name=st.session_state["model"])
                # supervisor_agent case 
                if "supervisor_agent" in output:
                    messages = output["supervisor_agent"]["messages"]
                    if messages:
                        last_message = messages[-1]
                        # 메시지 로그에 추가
                        print(f'\n----- supervisor agent step -----')
                        print(f'----- last_message -----: {last_message}')
                        content = last_message.content
                        meta_data = last_message.response_metadata  # 메타데이터 정보가 있다면 가져오기
                        model_name = meta_data.get('model_name', None)
                        add_to_messages_log("supervisor_agent", content, model_name)

                        # session state의 'current_chat_history'에 추가 
                        #add_to_current_chat('assistant', content)
                        
                        # Supervisor가 도구를 호출하는 경우 (create_handoff_tool에 의해 Command가 반환됨)
                        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            tool_name = last_message.tool_calls[0]["name"] # 'name' 속성 사용
                            status_container.markdown(f'{tool_name} 호출 중...')
                            continue # 다음 스트림 출력으로 넘어감
                        
                        # Supervisor가 최종 응답을 생성하는 경우
                        content = last_message.content
                        if "RECOMMENDATION_COMPLETE" in content:
                            full_response += content.replace("RECOMMENDATION_COMPLETE", "").strip()
                            chat_container.markdown(full_response)
                            status_container.markdown("추천 결과 생성 완료")
                            break # 'RECOMMENDATION_COMPLETE' 키워드 확인 후 루프 종료
                        else:
                            # 스트리밍 중간 응답
                            # gemini 모델의 경우 패스 
                            if 'gemini' in st.session_state["model"].lower():
                                continue
                            content = content.replace("Successfully transferred to web_search_agent", "").strip()
                            content = content.replace("Successfully transferred to load_preference_agent", "").strip()
                            full_response += content
                            #chat_container.markdown(full_response + "▌")
                            status_container.markdown('다음 스텝을 검토 중...')
                            continue # 다음 스트림 출력으로 넘어감
                # Agent 상태 메시지 처리 
                elif "web_search_agent" in output:
                    print(f'\n----- output of web_search_agent -----:\n{output["web_search_agent"]['messages']}')
                    status_container.markdown('웹 검색 중...')
                    # 메시지 로그에 추가 
                    content = output["web_search_agent"]["messages"][-1].content
                    add_to_messages_log("web_search_agent", content)
                    # Gemini 모델의 경우 web_search_agent의 메시지를 full_response에 추가해야 함 
                    if 'gemini' in st.session_state["model"].lower():
                        full_response += '\n' + content
                    continue
                elif "load_preference_agent" in output:
                    print(f'\n-----output of load_preference_agent -----:\n {output["load_preference_agent"]['messages']}')
                    status_container.markdown('기존 유저의 선호 음악 정보 불러오는 중...')
                    # 메시지 로그에 추가 
                    content = output["load_preference_agent"]["messages"][-1].content
                    add_to_messages_log("load_preference_agent", content)
                    continue
                elif "END" in output:
                    st.markdown("결과 생성 완료")
                    #st.session_state["graph"].reset()  # 그래프 상태 초기화
                    break

            # 상태 메시지 제거
            status_container.empty()

            # 최종 응답 결과 표시
            if not full_response and "supervisor_agent" in output: # 'RECOMMENDATION_COMPLETE'로 바로 끝나는 경우 대비
                final_messages = output["supervisor_agent"]["messages"]
                if final_messages:
                    final_content = final_messages[-1].content.replace("RECOMMENDATION_COMPLETE", "").strip()
                    full_response = final_content
                    chat_container.markdown(full_response)
            elif full_response:
                 chat_container.markdown(full_response)
            #st.session_state.messages.append({"role": "assistant", "content": full_response})

        # chat_history에 추가
        add_to_current_chat("assistant", full_response)
        print(f'\nchat_history: \n{chat_history}')
        
        # Assistant 세션 끝 
        # 전체 메시지 로그 저장
        save_messages_log(messages_to_save=st.session_state['messages'])
        # Output를 딕셔너리 화  
        output_dict = dict(output)
        # 채팅 이름 생성 
        if 'chat_name' in st.session_state:
            if st.session_state["chat_name"] != '':
                chat_name = st.session_state["chat_name"]
            else:
                name_generator = st.session_state["name_generator"]
                chat_history_for_name = [user_input, full_response]
                chat_name = name_generator.invoke({"messages": chat_history_for_name})
                st.session_state["chat_name"] = chat_name
        else:
            name_generator = st.session_state["name_generator"]
            chat_history_for_name = [user_input, full_response]
            chat_name = name_generator.invoke({"messages": chat_history_for_name})
            st.session_state["chat_name"] = chat_name
        # Tool을 포함한 전체 메시지, 모니터링용
        # 이전의 chat_history 파트 제거를 위해서 num_previous_chat_history 추가
        raw_messages_json = convert_to_json_for_raw(USER_ID,
                                                USER_NAME,
                                                thread_id,
                                                chat_name,
                                                st.session_state['current_num_chat_turn'],
                                                st.session_state['user_preference_reflected'],
                                                st.session_state['re_recommendation_allowed'],
                                                output_dict,
                                                num_previous_chat_history)
        # metadata 추가 
        raw_messages_json.update(st.session_state['current_models_config'])
        print(f'In the Saving Process, Current Models Config: {st.session_state['current_models_config']}')
        redis_key = f'{USER_ID}:{thread_id}'
        # Redis에 저장 
        redis_client.save_into_redis(raw_messages_json, redis_key)
        raw_messages_json['redis_key'] = redis_key
        # MongoDB에 저장 
        mongodb_client.insert_update_raw_messages(raw_messages_json)
        # 유저에게 보여주기 위한 대화 내역 
        messages_for_user_json = {'user_id': USER_ID,
                                'user_name': USER_NAME,
                                'thread_id': thread_id,
                                'num_chat_turn': st.session_state['current_num_chat_turn'],
                                'chat_name': chat_name,
                                'redis_key': redis_key,
                                'messages': st.session_state['current_chat_history']}
        # MongoDB에 저장 
        mongodb_client.insert_update_messages_for_user(messages_for_user_json)




def display_previous_chat_jsonl(filename):
    st.title("이전 채팅")
    history = load_chat_history(filename)
    display_previous_chat_history(history)
    if st.button("돌아가기"):
        st.session_state['current_page'] = 'main'
        st.session_state['selected_chat_data'] = None
        #st.rerun()

def display_previous_chat_redis(previous_messages):
    st.title("이전 채팅")
    history = previous_messages
    display_previous_chat_history(history)
    if st.button("돌아가기"):
        st.session_state['current_page'] = 'main'
        st.session_state['selected_chat_data'] = None
        #st.rerun()

def display_previous_chat(previous_chat_data, is_redis=True):
    if is_redis:
        display_previous_chat_redis(previous_chat_data)
    else:
        display_previous_chat_jsonl(previous_chat_data)

# 새로운 채팅 시작 
def start_new_chat():
    '''
    if st.session_state['current_chat_history']:
        # 현재 채팅 기록을 저장하고 이전 채팅 목록에 추가
        filename = f"chat_{len(st.session_state['previous_chats']) + 1}"
        print(f'\ntype of current_chat_history: {type(st.session_state["current_chat_history"])}')
        print(f'current_chat_history: {st.session_state["current_chat_history"]}\n')
        name_generator = st.session_state["name_generator"]
        name = name_generator.invoke({"messages": st.session_state['current_chat_history']})
        save_chat_history_to_jsonl(st.session_state['current_chat_history'], name)
        st.session_state['previous_chats'].append(name)
        st.session_state['current_chat_history'] = []
        st.session_state['selected_chat_data'] = None
        st.session_state['current_page'] = 'main'
        st.rerun()
    '''
    if 'chat_name' in st.session_state:
        st.session_state['current_chat_history'] = []
        st.session_state['selected_chat_data'] = None
        st.session_state['current_page'] = 'main'
        st.session_state['num_user_chats'] += 1 # 새 대화 세션마다 1 증가
        st.session_state['thread_id'] += 1 # 새 대화 세션마다 1 증가
        st.session_state['current_num_chat_turn'] = 0 # 새 대화마다 초기화 
        st.session_state["chat_name"] = ''

        #st.rerun()
    else:
        pass



with st.sidebar:
    # 모델 선택 UI  
    st.title("모델 선택")

    ## 옵션 정의
    model_options = common.availabe_models
    print(f'Sidebar, model_options: {model_options}')
 
    ## 선택 상자 생성
    selected_option = st.selectbox('모델을 선택하세요:', model_options)

    # 모델이 변경되었는지 확인
    if st.session_state.get("model") != selected_option:
        print('\n---- In Streamlit sidebar session ----')
        print(f'model changed from {st.session_state.get("model")} to {selected_option}')
        st.session_state["model"] = selected_option
        # 모델이 변경되었다면 그래프를 재생성
        if "graph" in st.session_state:
            del st.session_state["graph"]  # 기존 그래프 삭제
        st.session_state["graph"] = llm_langgraph.create_chat_graph(selected_option)
        st.toast(f"모델이 {selected_option}로 설정되었습니다.")
        # metadata인 current_models 딕셔너리의 값도 변경
        current_models_config['supervisor_llm_model'] = selected_option
        subagent_llm_model = get_subagent_llm_model(llm_models_config,
                                                    supervisor_llm_model=selected_option)
        current_models_config['subagent_llm_model'] = subagent_llm_model
        print(f'\nSelected Supervisor Agent Model: {selected_option}')
        print(f'Selected Sub-Agent Model: {subagent_llm_model}')
        st.session_state['current_models_config'] = current_models_config
        print(f'In sidebar, Current Models Config: {st.session_state['current_models_config']}')

    st.markdown("---")  # 수평선 추가

    # 기존 선호 데이터 반영 여부  
    st.title("기존 선호 데이터")
    st.sidebar.header("이전 선호 데이터 반영 여부")
    pref_reflect, pref_not_reflect = st.columns(2)  # 두 개의 컬럼 생성
    pref_reflect.button('기존 선호 반영', on_click=change_user_preference_reflection, args=[True])
    pref_not_reflect.button('기존 선호 미반영', on_click=change_user_preference_reflection, args=[False])
    st.write(f"**현재 상태:** {'✅ 반영됨' if st.session_state.user_preference_reflected else '❌ 미반영'}")
    st.sidebar.header("이전 선호 노래 재추천 여부")
    re_rec, no_re_rec = st.columns(2)  # 두 개의 컬럼 생성
    re_rec.button("재추천 허용", use_container_width=True, on_click=change_re_recommendation_allowance, args=[True])
    no_re_rec.button("재추천 비허용", use_container_width=True, on_click=change_re_recommendation_allowance, args=[False])
    st.write(f"**현재 상태:** {'🔄 재추천 허용' if st.session_state.re_recommendation_allowed else '❌ 재추천 비허용'}")


    st.markdown("---") # 구분선

    col1, col2 = st.columns([1, 1])  # 컬럼 너비 비율 조정 가능
    with col1:
        st.title('채팅 관리')
    with col2:
        st.markdown(
        """
        <style>
            div[data-testid="column"]:nth-child(2) {
                display: flex;
                align-items: center;
                justify-content: flex-end; /* 오른쪽 정렬 유지 */
                height: 100%; /* 컬럼 높이를 부모 요소에 맞춤 */
            }
        </style>
        """,
        unsafe_allow_html=True,
        )
        st.button("새로운 채팅", on_click=start_new_chat)

    st.sidebar.header("이전 채팅 목록")
    if st.session_state['previous_chats']:
        disply_previous_chat_list_in_sidebar()





# 현재 페이지 상태에 따라 콘텐츠 표시
if st.session_state['current_page'] == 'main':
    display_main_chat()
elif st.session_state['current_page'] == 'previous_chat' and st.session_state['selected_chat_data']:
    display_previous_chat(st.session_state['selected_chat_data'])