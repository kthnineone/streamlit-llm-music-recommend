
# RAGAS에서 trace_roots 에서 index 문제가 발생해서 직접 평가하는 방식으로 변경
# RAGAS의 ContextRelevance를 약간 변형하여 사용 
# https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_nv_metrics.py


# OpenAI 사용  
import os
import yaml
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

base_path = os.path.dirname(os.getcwd())
prompt_dir_path = os.path.join(base_path, 'assets', 'templates', 'prompts')

## Load Config ##

## Load Prompts ##

def load_prompt_configs(filepath=None):
    """
    YAML 파일에서 LLM 모델 설정을 로드합니다.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
        return []
    except yaml.YAMLError as e:
        print(f"오류: YAML 파일을 파싱하는 중 문제가 발생했습니다: {e}")
        return []

## Load Current Models Config ##
def load_current_models_config(filepath=None):
    """
    YAML 파일에서 LLM 모델 설정을 로드합니다.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            config = config.get('current_models', [])
        return config
    except FileNotFoundError:
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
        return []
    except yaml.YAMLError as e:
        print(f"오류: YAML 파일을 파싱하는 중 문제가 발생했습니다: {e}")
        return []

## Load Meta data from Current Models Config ## 
config_dir = os.path.join(base_path, 'src', 'config', 'current_models.yaml')
current_config = load_current_models_config(config_dir)
current_config = current_config
print(f'Current Config: {current_config}')

## Set LLM Judges ## 

# OpenAI LLM Wrappers (ChatGPT 또는 GPT-4 모델 사용)
context_relevance_evaluator_llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)

# OpenAI Embeddings Wrapper
rag_evaluator_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)


# Context Relevance Prompt
prompt_path = os.path.join(prompt_dir_path, 'context_relevance.yaml')
context_relevance_template = load_prompt_configs(prompt_path)
context_relevance_template = context_relevance_template['template']

context_relevance_prompt = PromptTemplate.from_template(context_relevance_template)

## Make Chains
context_relevance_evaluator_chain = context_relevance_prompt | context_relevance_evaluator_llm


def web_agent_context_relevance_check(agent_data):
    print('------ Context Relevance Calculation ------')
    messages = agent_data['messages']
    search_results = []
    context_relevance_scores = []
    for message in messages:
        if message['type'] in ('user', 'human'):
            question = message['content']
        elif message['type'] == 'ai' and message['name'] == 'web_search_agent':
            search_result = message['content']
            if search_result != '':
                search_results.append(search_result)
    for search_result in search_results:
        context_relevance_score = context_relevance_evaluator_chain.invoke({'question': question,
                                                                            'context': search_result})
        print(f'Raw score of context relevance score: - type {type(context_relevance_score)} - {context_relevance_score}')
        context_relevance_score = int(context_relevance_score.strip())
        context_relevance_scores.append(context_relevance_score)
    context_relevance_scores = np.array(context_relevance_scores)
    mean_context_relevance_scores = np.mean(context_relevance_scores)
    mean_context_relevance_scores = float(mean_context_relevance_scores)
    return mean_context_relevance_scores

def web_search_context_relevance_check(agent_data):
    print('------ Context Relevance Calculation ------')
    messages = agent_data['messages']
    search_results = []
    context_relevance_scores = []
    for message in messages:
        if message['type'] in ('user', 'human'):
            question = message['content']
        elif message['type'] == 'tool' and message['name'] in ('tavily_search', 'google_search'):
            search_result = message['content']
            if search_result != '':
                search_results.append(search_result)
    for search_result in search_results:
        context_relevance_score = context_relevance_evaluator_chain.invoke({'question': question,
                                                                            'context': search_result})
        print(f'Raw score of context relevance score: - type {type(context_relevance_score)} - {context_relevance_score}')
        context_relevance_score = int(context_relevance_score.content.strip())
        context_relevance_scores.append(context_relevance_score)
    context_relevance_scores = np.array(context_relevance_scores)
    mean_context_relevance_scores = np.mean(context_relevance_scores)
    mean_context_relevance_scores = float(mean_context_relevance_scores)
    return mean_context_relevance_scores



# Initialize with Google AI Studio
hallucination_evaluator_llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)

# Response Groundedness Prompt

prompt_path = os.path.join(prompt_dir_path, 'response_groundedness.yaml')
response_groundedness_template = load_prompt_configs(prompt_path)
response_groundedness_template = response_groundedness_template['template']

response_groundedness_prompt = PromptTemplate.from_template(response_groundedness_template)

groundedness_evaluator_chain = response_groundedness_prompt | hallucination_evaluator_llm


def response_groundedness_check(agent_data):
    print('------ Response Groundedness Calculation ------')
    messages = agent_data['messages']
    search_results = []
    response_groundedness_scores = []
    for message in messages:
        if message['type'] in ('user', 'human'):
            question = message['content']
        elif message['type'] == 'ai' and message['name'] == 'web_search_agent':
            search_result = message['content']
            if search_result != '':
                search_results.append(search_result)
        elif message['type'] == 'ai' and message['name'] == 'supervisor_agent':
            supervisor_result = message['content']
            if supervisor_result != '':
                if "RECOMMENDATION_COMPLETE" in supervisor_result:
                    response = supervisor_result
    for search_result in search_results:
        response_groundedness_score = groundedness_evaluator_chain.invoke({'response': response,
                                                                            'context': search_result})
        print(f'Raw score of context relevance score: - type {type(response_groundedness_score)} - {response_groundedness_score}')
        response_groundedness_score = int(response_groundedness_score.content.strip())
        response_groundedness_scores.append(response_groundedness_score)
    response_groundedness_scores = np.array(response_groundedness_scores)
    mean_response_groundedness_scores = np.mean(response_groundedness_scores)
    mean_response_groundedness_scores = float(mean_response_groundedness_scores)
    return mean_response_groundedness_scores



# OpenAI Embeddings Wrapper
embedding_model = current_config['embedding_model']
embedding_type = current_config['embedding_type']
print(f'Current Embedding type: {embedding_type}, model: {embedding_model}')

response_relevance_evaluator_embeddings = OpenAIEmbeddings(
                                                        model=embedding_model
                                                        )


def calculate_similarity(sequence_0: str, sequence_1: str) -> float:
    seq_vec_0 = np.asarray(response_relevance_evaluator_embeddings.embed_query(sequence_0)).reshape(1, -1)
    seq_vec_1 = np.asarray(response_relevance_evaluator_embeddings.embed_query(sequence_1)).reshape(1, -1)
    norm = np.linalg.norm(seq_vec_0, axis=1) * np.linalg.norm(
            seq_vec_1, axis=1
        )
    return (
            np.dot(seq_vec_0, seq_vec_1.T).reshape(
                -1,
            )
            / norm
        )

def get_response_relevancy_score(question, answer):
    print('------ Response Relevancy Calculation ------')
    response_relevancy_score = calculate_similarity(question, 
                                                    answer)
    print(f'response_relevancy_score: {response_relevancy_score}')
    response_relevancy_score = float(response_relevancy_score[0])
    print(f'response_relevancy_score: {response_relevancy_score}')
    
    return response_relevancy_score


def response_relevancy_check(agent_data):
    print('------ Response Relevancy Calculation ------')
    messages = agent_data['messages']
    for message in messages:
        if message['type'] in ('user', 'human'):
            question = message['content']
        elif message['type'] == 'ai' and message['name'] == 'web_search_agent':
            continue
        elif message['type'] == 'ai' and message['name'] == 'supervisor_agent':
            supervisor_result = message['content']
            if supervisor_result != '':
                if "RECOMMENDATION_COMPLETE" in supervisor_result:
                    response = supervisor_result
    print(f'question: \n{question}\n response: \n{response}\n')
    response_relevancy_score = get_response_relevancy_score(question, response)
    
    return response_relevancy_score


'''
# Faithfullness

faithfulness_evaluator_llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)

prompt_path = os.path.join(prompt_dir_path, 'response_groundedness.yaml')
faithfulness_template = load_prompt_configs(prompt_path)
faithfulness_template = faithfulness_template['template']


faithfulness_score_prompt = PromptTemplate.from_template(faithfulness_template)

faithfulness_evaluator_chain = faithfulness_score_prompt | faithfulness_evaluator_llm 

faithfulness_scores = []

def faithfulness_check(state: GraphState) -> GraphState:
    print('----- Faithfulness Check -----')
    # Create Statements
    statements_data = create_statements(state)
    statements_data = statements_data['sentences']

    context = state['filtered_context']
    n_faithful_statements = 0
    n_statements = 0
    for statement_element in statements_data:
        statements = statement_element['statements']
        for statement in statements:
            print(f'statement: {statement}')
            eval_data = {'answer': statement,
                        'context': context}
            faithfulness_score = faithfulness_evaluator_chain.invoke(eval_data)
            print(faithfulness_score)
            #break
            faithfulness_score = int(faithfulness_score.content)
            n_faithful_statements += faithfulness_score
            n_statements += 1

    faithfulness_score = n_faithful_statements / n_statements
    return GraphState(faithfulness_score=faithfulness_score)
'''

def get_llm_evaluations(agent_data):
    llm_service_type = agent_data['llm_service_type']
    search_tool_name = agent_data['search_tool_name']
    supervisor_model = agent_data['supervisor_llm_model']
    subagent_model = agent_data['subagent_llm_model']
    embedding_model = current_config['embedding_model']
    embedding_type = current_config['embedding_type']
    web_search_context_relevance_score = web_search_context_relevance_check(agent_data)
    response_relevancy_score = response_relevancy_check(agent_data)
    mean_response_groundedness_score = response_groundedness_check(agent_data)

    return {'llm_service_type': llm_service_type,
            'search_tool_name': search_tool_name,
            'supervisor_model': supervisor_model,
            'subagent_model': subagent_model,
            'embedding_model': embedding_model,
            'embedding_type': embedding_type,
            'web_search_context_relevance_score': web_search_context_relevance_score,
            'response_relevancy_score': response_relevancy_score,
            'mean_response_groundedness_score': mean_response_groundedness_score,
            }



