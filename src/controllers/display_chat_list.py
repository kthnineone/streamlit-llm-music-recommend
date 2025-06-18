import streamlit as st

def disply_previous_chats_in_sidebar_mongo():
    # mongodb
    for previous_chat in st.session_state['previous_chats']:
        chat_title = previous_chat['chat_name']
        chat_title = chat_title.title()
        previous_chat_key = previous_chat['thread_id']
        # chat_title은 사용자에게 보여주는 제목,
        # key는 Streamlit 위젯의 고유한 식별자 역할
        if st.sidebar.button(chat_title, key=previous_chat_key):
            st.session_state['current_page'] = 'previous_chat'
            st.session_state['selected_chat_data'] = previous_chat['messages']
            st.rerun()

def disply_previous_chats_in_sidebar_redis():
    for redis_key, previous_chat in st.session_state['previous_chats'].items():
        chat_title = previous_chat['chat_name']
        chat_title = chat_title.title()
        previous_chat_key = previous_chat['thread_id']
        # chat_title은 사용자에게 보여주는 제목,
        # key는 Streamlit 위젯의 고유한 식별자 역할
        if st.sidebar.button(chat_title, key=previous_chat_key):
            st.session_state['current_page'] = 'previous_chat'
            st.session_state['selected_chat_data'] = previous_chat['messages']
            st.rerun()

def disply_previous_chats_in_sidebar_jsonl():
    for filename in st.session_state['previous_chats']:
        # 파일 이름에서 채팅 제목 추출 (예: "chat_1" -> "채팅 1")
        chat_title = filename.replace("_", " ").lower()
        chat_title = chat_title.replace('.jsonl', '').title()
        # chat_title은 사용자에게 보여주는 제목,
        # key는 Streamlit 위젯의 고유한 식별자 역할
        if st.sidebar.button(chat_title, key=filename):
            st.session_state['current_page'] = 'previous_chat'
            st.session_state['selected_chat_data'] = filename
            st.rerun()

def disply_previous_chat_list_in_sidebar(method='mongo'):
    if method.lower() in ('mongo', 'mongodb'):
        disply_previous_chats_in_sidebar_mongo()
    elif method.lower() in ('redis'):
        disply_previous_chats_in_sidebar_redis()
    elif method.lower() in ('json', 'jsonl'):
        disply_previous_chats_in_sidebar_jsonl()
    else:
        print("Not proper method to show previous chats in sidebar")