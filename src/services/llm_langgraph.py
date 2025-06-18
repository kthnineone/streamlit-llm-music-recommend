import os
import json
import yaml
import pandas as pd 

from typing import Annotated, Literal, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    BaseMessage, 
    ChatMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    ToolMessage
)
from langchain_core.prompts import (
    PromptTemplate,
    load_prompt,
    MessagesPlaceholder,
    ChatPromptTemplate
)
from langchain_core.tools import tool, InjectedToolCallId

from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState
)
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.prebuilt import (
    ToolNode, 
    tools_condition,
    InjectedState,
    create_react_agent
)
from langgraph.checkpoint.memory import MemorySaver


from .common import get_llm_model, llm_model_configs, subagent_llm_model_configs, availabe_models_dict


########## 1. 상태 정의 ##########
# 상태 정의
class State(TypedDict):
    # 메시지 목록 주석 추가
    messages: Annotated[List[BaseMessage], add_messages]


def create_chat_graph(model_name: str):
    #graph = create_simple_agent_graph(model_name)
    graph = create_supervisor_agent_graph(model_name)
    return graph


def create_simple_agent_graph(model_name: str):
    print(f'Build graph with model: {model_name}')
    # 메모리 저장소 생성
    memory = MemorySaver()
    ########## 2. 도구 정의 및 바인딩 ##########
    # 도구 초기화
    search_tool = TavilySearch(max_results=3)
    tools = [search_tool]

    # LLM 초기화
    llm = get_llm_model(model_name, llm_model_configs)
    # 도구와 LLM 결합
    # 실제 도구의 실행은 ToolNode에서 이루어지며, LLM은 도구의 실행을 요청하는 역할을 합니다.
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = "You are a music recommendation assistant. You will be given a list of songs and you need to recommend a song based on the user's preferences."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{question}"),
        ]
    )


    chain = prompt | llm_with_tools


    ########## 3. 노드 추가 ##########
    # 챗봇 함수 정의
    def chatbot(state: State):
        # 메시지 호출 및 반환
        return {"messages": [chain.invoke({'question': (state["messages"])})]}

    # 상태 그래프 생성
    graph_builder = StateGraph(State)

    # 챗봇 노드 추가
    graph_builder.add_node("chatbot", chatbot)

    # 도구 노드 생성 및 추가
    tool_node = ToolNode(tools=[search_tool])

    # 도구 노드 추가
    graph_builder.add_node("tools", tool_node)

    # 조건부 엣지
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    ########## 4. 엣지 추가 ##########

    # tools > chatbot
    graph_builder.add_edge("tools", "chatbot")

    # START > chatbot
    graph_builder.add_edge(START, "chatbot")

    # chatbot > END
    graph_builder.add_edge("chatbot", END)

    # 그래프 빌더 컴파일
    graph = graph_builder.compile(checkpointer=memory)

    return graph


# 상태 정의
class GraphState(TypedDict):
    # 메시지 목록 주석 추가
    messages: Annotated[List[BaseMessage], add_messages]
    chat_history: Annotated[List[BaseMessage], add_messages]

def load_preference_data():
    curr_dir = os.path.dirname(os.getcwd())
    parent_dir = os.path.dirname(curr_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    data_path = os.path.join(grandparent_dir, "assets", "data")
    data_path = os.path.join(curr_dir, "assets", "data")
    print(f'current directory: {curr_dir}')
    df = pd.read_csv(os.path.join(data_path,'prefered_songs.csv'))
    df = df.rename(columns={'곡': 'song', '아티스트': 'artist', '앨범': 'album'})
    artist_info_by_spotify = pd.read_csv(os.path.join(data_path,'artist_info.csv'))
    df = pd.merge(df, artist_info_by_spotify, left_on='artist', right_on = 'original_name', how='left')
    return df

def convert_to_genre_list(genres_str: any):
    if isinstance(genres_str, str):
        genres_str = genres_str.replace('"', '')
        genres_str = genres_str.replace("[", '')
        genres_str = genres_str.replace("]", '')
        genres_str = genres_str.replace("'", '')
        genres_list = genres_str.split(",")
        genres_list = [genre.strip() for genre in genres_list]
    else:
        genres_list = []
    return genres_list

def is_genre_in_query(genre_list, query):
    if isinstance(genre_list, list):
        for genre in genre_list:
            if genre in query:
                return True
    return False

class PreferenceData:
    def __init__(self):
        self.df = load_preference_data()
        self.df['genres_list'] = self.df['genres'].apply(lambda x: convert_to_genre_list(x))

    def extract_artists_by_genres(self, query_genres: list[str]):
        """Extract artists by genres."""
        has_query_genre = self.df['genres_list'].apply(lambda x: is_genre_in_query(x, query_genres))
        filtered_df = self.df[has_query_genre]
        filtered_artists = filtered_df['artist'].unique().tolist()
        return {'context': filtered_artists}

    def extract_artists_n_songs_by_genres(self, query_genres: list[str]):
        """Extract songs and their artists who are in query_genres genres."""
        has_query_genre = self.df['genres_list'].apply(lambda x: is_genre_in_query(x, query_genres))
        filtered_df = self.df[has_query_genre]
        filtered_songs = filtered_df[['artist', 'song']].to_dict(orient='records')
        return {'context': filtered_songs}


# Handoff = transfer to another agent, 밀치다 

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."
    print(f"Creating handoff tool: {name} with description: {description}")

    # Python decorator to create a tool
    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        '''
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        '''
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,  # ToolMessage에서는 tool_call_id 대신 name을 사용하거나,
                        # LangGraph의 InjectedToolCallId를 사용하는 경우 다르게 처리될 수 있습니다.
                        # 여기서는 `tool_call_id`가 ToolMessage의 일부가 아닌, tool_call_result의 일부입니다.
                        # LangGraph의 Command update context에서는 dict 형태의 메시지가 필요할 수 있으므로,
                        # 이 부분을 LangGraph의 메시지 처리 방식에 맞게 조정해야 합니다.
                        # LangGraph의 StateGraph는 내부적으로 BaseMessage 인스턴스를 기대합니다.
            tool_call_id=tool_call_id # ToolMessage의 constructor에 tool_call_id를 직접 넣을 수도 있습니다.
        )
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )

    return handoff_tool


def get_sub_agent_llm_models(model_name: str):
    """
    Get the LLM model for sub-agents based on the model name.
    """
    if model_name in availabe_models_dict:
        model_group = availabe_models_dict[model_name]
        if model_group == 'openai':
            sub_llm_model = 'gpt-4.1-mini-2025-04-14'
        elif model_group == 'google':
            sub_llm_model = 'gemini-2.0-flash'
        elif model_group == 'anthropic':
            sub_llm_model = 'claude-3-5-haiku-20241022'
        return sub_llm_model
    else:
        raise ValueError(f"Model {model_name} is not available in the configuration.")


def create_supervisor_agent_graph(model_name: str):
    print(f'\nSet up LLMs for each agents with supervisor model: {model_name} in group: {availabe_models_dict[model_name]}')
    supervisor_llm = get_llm_model(model_name, llm_model_configs)
    print(f'Creating supervisor agent with model: {model_name}')
    sub_llm_model_name = get_sub_agent_llm_models(model_name)
    print(f'Sub-agent LLM model: {sub_llm_model_name}\n')
    
    # Web Search Agent 
    web_search = TavilySearch(max_results=3)
    web_llm = get_llm_model(sub_llm_model_name, subagent_llm_model_configs)
    web_search_agent = create_react_agent(
        model=web_llm,
        tools=[web_search],
        prompt=(
            "You are a web search agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with web search-related tasks.\n"
            "- After you're done with your tasks, respond to the supervisor directly.\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text.\n"
            "- Do NOT recommend any music in this step."
        ),
        name="web_search_agent",
    )

    curr_dir = os.path.dirname(os.getcwd())
    parent_dir = os.path.dirname(curr_dir)
    print(f'current directory: {curr_dir}')
    print(f'parent directory: {parent_dir}')
    grandparent_dir = os.path.dirname(parent_dir)
    print(f'grandparent directory: {grandparent_dir}')
    data_path = os.path.join(grandparent_dir, "assets", "data", "genre_set.json")
    data_path = os.path.join(curr_dir, "assets", "data", "genre_set.json")
    with open(data_path, "r") as f:
        genre_set = json.load(f)

    # Preference Data Agent
    preferece_data = PreferenceData()
    load_preference_llm = get_llm_model(sub_llm_model_name, subagent_llm_model_configs)

    load_preference_agent = create_react_agent(
        model=load_preference_llm,
        tools=[
            preferece_data.extract_artists_by_genres, 
            preferece_data.extract_artists_n_songs_by_genres
            ],
        prompt=(
            "You are an agent which loads music preference data.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with loading preference data tasks\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
            "- Exclude era infomation in the query.\n"
            f"- Find most relevant genres among given genre list: {list(genre_set)} and consist the query_genres.\n"
            "- If user's query is about artists, return full context as a list of artists who are in the query_genres.\n"
            "- If user's query is about songs, return full context as a list of songs and their artists who are in the query_genres.\n"
        ),
        name="load_preference_agent",
        )

    # Handoffs
    assign_to_web_search_agent = create_handoff_tool(
        agent_name="web_search_agent",
        description="Assign task to a web searcher agent.",
    )

    assign_to_load_preference_agent = create_handoff_tool(
        agent_name="load_preference_agent",
        description="Assign task to a loading preference agent.",
    )

    # Supervisor Agent
    '''
    supervisor_agent = create_react_agent(
    model=supervisor_llm,
    tools=[assign_to_web_search_agent, assign_to_load_preference_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a web search agent. Assign web search-related tasks to this agent\n"
        "- a preference loading agent. Assign loading music preference-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "After the agent completes its task, it will return to you with the results.\n"
        "You will collect those results and utilize them to recommend music to the user.\n"
        "The recommendation must be produced in supervisor agent.\n"
        "The answer should be written in Korean.\n"
        "After making a recommendation, you will provide the recommendation and then explicitly state 'RECOMMENDATION_COMPLETE' at the very end of your response.\n\n"
    ),
    name="supervisor_agent"
    )
    '''

    # Supervisor Agent Prompt
    supervisor_agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
            "You are a supervisor managing two agents:\n"
            "- a web search agent. Assign web search-related tasks to this agent\n"
            "- a preference loading agent. Assign loading music preference-related tasks to this agent\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "After the agent completes its task, it will return to you with the results.\n"
            "You will collect those results and utilize them to recommend music to the user.\n"
            "The recommendation must be produced in supervisor agent.\n"
            "The answer should be written in Korean.\n"
            "After making a recommendation, you will provide the recommendation and then explicitly state 'RECOMMENDATION_COMPLETE' at the very end of your response.\n\n"
            ),
            MessagesPlaceholder(variable_name="messages") # chat_history를 위한 변수명으로 messages 혹은 chat_history를 추가한다
        ]
    )
    supervisor_agent = create_react_agent(
    model=supervisor_llm,
    tools=[assign_to_web_search_agent, assign_to_load_preference_agent],
    prompt=supervisor_agent_prompt,
    name="supervisor_agent"
    )

    '''
    supervisor_agent = create_react_agent(
    model=supervisor_llm,
    tools=[assign_to_web_search_agent, assign_to_load_preference_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a web search agent. Assign web search-related tasks to this agent\n"
        "- a preference loading agent. Assign loading music preference-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "After the agent completes its task, it will return to you with the results.\n"
        "You will collect those results and utilize them to recommend music to the user.\n"
        "The recommendation must be produced in supervisor agent.\n"
        "The answer should be written in Korean.\n"
        "After making a recommendation, you will provide the recommendation and then explicitly state 'RECOMMENDATION_COMPLETE' at the very end of your response.\n\n"
    ),
    name="supervisor_agent"
    )
    '''

    # Define the multi-agent supervisor graph
    '''
    supervisor = (
        StateGraph(MessagesState)
        # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
        .add_node(supervisor_agent, destinations=("web_search_agent", "load_preference_agent", END))
        .add_node(web_search_agent)
        .add_node(load_preference_agent)
        .add_edge(START, "supervisor_agent")
        # always return back to the supervisor
        .add_edge("web_search_agent", "supervisor_agent")
        .add_edge("load_preference_agent", "supervisor_agent")
        .compile()
    )
    '''
    # Define the multi-agent supervisor graph
    #workflow = StateGraph(MessagesState)
    workflow = StateGraph(GraphState)

    # 노드 추가
    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("web_search_agent", web_search_agent)
    workflow.add_node("load_preference_agent", load_preference_agent)

    # 시작 지점에서 supervisor_agent로 연결
    workflow.add_edge(START, "supervisor_agent")

    # 웹 검색 에이전트와 선호도 로딩 에이전트가 작업 완료 후 항상 supervisor_agent로 복귀
    workflow.add_edge("web_search_agent", "supervisor_agent")
    workflow.add_edge("load_preference_agent", "supervisor_agent")

    # Supervisor 에이전트의 조건부 전이 정의
    # Supervisor가 최종 응답을 생성하고 더 이상 도구를 호출할 필요가 없을 때 END로 전이
    def route_supervisor(state: MessagesState) -> Literal["end", "continue"]:
        # supervisor_agent의 마지막 메시지를 확인
        last_message = state["messages"][-1]
        
        # 만약 마지막 메시지에 tool_calls가 있다면, Supervisor가 다른 도구를 호출할 계획이므로 계속 진행
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # tool_calls가 있으면 해당 툴로 자동 전이될 것이므로 여기서는 'continue'를 반환하여
            # 명시적으로 다른 노드로 가지 않도록 합니다.
            # 하지만 실제 LangGraph ReactAgent는 tool_calls가 있으면 Command를 생성하여 직접 전이합니다.
            # 이 라우팅 함수는 tool_calls가 없는 경우에만 END로 보내는 역할을 하도록 설계합니다.
            return "continue"
        
        # tool_calls가 없고, Supervisor가 최종 답변을 생성했을 때 'RECOMMENDATION_COMPLETE' 키워드를 포함하도록 프롬프트에 지시했으므로 이를 확인
        if "RECOMMENDATION_COMPLETE" in last_message.content:
            return "end" # 'RECOMMENDATION_COMPLETE'가 있으면 END 노드로 전이
        
        # 그 외의 경우 (예: 중간 답변, 아직 최종 단계가 아닌 경우) 다시 Supervisor로 돌아가도록 'continue' 반환
        # 이 'continue'는 실질적으로 ReactAgent가 도구를 호출하지 않고 일반 텍스트를 반환했을 때의 경로가 됩니다.
        return "continue"

    workflow.add_conditional_edges(
        "supervisor_agent", # 이전 노드
        route_supervisor,   # 라우팅 함수
        {
            "end": END,
            "web_search_agent": "web_search_agent", # Supervisor가 web_search_agent를 호출했을 때
            "load_preference_agent": "load_preference_agent", # Supervisor가 load_preference_agent를 호출했을 때
            "continue": "supervisor_agent" # Supervisor가 아직 최종 응답을 생성하지 않고 추가 처리가 필요한 경우
        }
    )
    graph = workflow.compile()
    return graph



def create_name_generator():
    chat_name_generator = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", 
                                    temperature=0.3)

    name_generating_system_prompt = """You are a Chat name generator.  \n
    "You will be given a conversation and you need to generate a name for the conversation.
    Name should be short, catchy, and contains essential information of convsersation. 
    The title should be written in Korean.\n\n

    #Conversation: \n{messages} \n\n
    """
    prompt = PromptTemplate.from_template(name_generating_system_prompt)

    name_generating_chain = prompt | chat_name_generator | StrOutputParser()

    return name_generating_chain
