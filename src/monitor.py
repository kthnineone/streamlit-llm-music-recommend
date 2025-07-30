import os
import json
import time
import pandas as pd
import streamlit as st
import services.common as common
from utils.chat_mongodb import *
from utils.base import *
from utils.monitor.evaluate import *
from utils.monitor.display import *

# User_info
USER_ID = "user"
USER_NAME = "me"

#### Streamlit session_state Initilization ####

st.session_state['user_id'] = USER_ID
st.session_state['user_name'] = USER_NAME

#### Databases ####

mongodb_client = MongoDBClient()

# mongodb에서 불러온 이전 채팅 목록
previous_chats = mongodb_client.load_recent_data(USER_ID)
redis_keys_from_previous_chats = mongodb_client.get_redis_keys_from_recent_data(previous_chats)

st.session_state['previous_chats'] = previous_chats

# --- Streamlit 대시보드 ---
st.set_page_config(page_title="음악 추천 대시보드", page_icon="📊")

collection_raw = mongodb_client.raw_messages_collection

#documents = list(collection_raw.find())
documents = list(collection_raw.find({"redis_key": 'user:10'}))

agent_data = documents[0]

st.sidebar.title("대시보드 선택")
page = st.sidebar.radio(
    "확인할 대시보드를 선택하세요:",
    ["LLM 성능 평가", "제약 조건 준수 평가", "LLM 구성요소 평가", "LangSmith"]
)


if page == "LLM 성능 평가":
    display_llm_performance(agent_data)
elif page == "제약 조건 준수 평가":
    display_constraints_eval_result(agent_data)
elif page == "LLM 구성요소 평가":
    display_llm_component_eval(agent_data)
elif page == "LangSmith":
    disply_langsmith()

