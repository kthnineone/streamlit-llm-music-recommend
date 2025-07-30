# Music Recommender with Streamlit and LLMs  

## Demo

실제 채팅 화면  

![채팅 화면 움직이는 이미지](https://github.com/kthnineone/streamlit-llm-music-recommend/blob/main/demo/llm_agent_music_rec_250612.gif "Main Chat gif")

<br>

![채팅 화면 이미지](https://github.com/kthnineone/streamlit-llm-music-recommend/blob/main/demo/llm_based_music_recommender_main_3.PNG "Main Chat 3")


**Details**  

블로그에 채팅 이미지, 영상, 대시보드 샘플 업로드 [(🔗 링크)](https://arsetstudium.tistory.com/382)  


**LangGraph Visualization**  

LangGraph 전체  

![전체 LangGraph 이미지](https://github.com/kthnineone/streamlit-llm-music-recommend/blob/main/demo/음악추천챗봇_LangGraph.png "LangGraph - Total")  

Web Search Agent  

![Web Search Agent LangGraph 이미지](https://github.com/kthnineone/streamlit-llm-music-recommend/blob/main/demo/음악추천챗봇_WebSearchAgent.png "LangGraph - Web Search Agent")  


## Description  

+ Streamlit으로 ChatBot 구현  
+ OpenAI, Gemini, Anthropic API 사용  
+ LangGrpah로 Agent와 Supervisor 구현  
+ Tavily API로 웹 검색   
+ Sub-Agents:  
  +  Web Search Agent  
  +  Load User's Preference Agent  
+ MongoDB (Local DB by Docker)  
+ Redis (Local DB by Docker)  

## Requirements  

+ langchain
+ langchain_community
+ langchain_openai
+ langchain_anthropic
+ langchain_google
+ langchain_tavily
+ streamlit
+ pandas
+ google-genai  
