import os
import yaml
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
#from langchain_huggingface import ChatHuggingFace

# 환경변수 로드
load_dotenv()
# OpenAI API 키 설정
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Gemini API 키 설정
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#genai.configure(api_key=GEMINI_API_KEY)

# Credentials 관련 조치
#credentials_path = os.getenv("GEMINI_CREDENTIALS")
#print(credentials_path)
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# 사용 가능한 모델들

def load_llm_configs(filepath="config/llm_models.yaml"):
    """
    YAML 파일에서 LLM 모델 설정을 로드합니다.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config.get('llm_models', [])
    except FileNotFoundError:
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
        return []
    except yaml.YAMLError as e:
        print(f"오류: YAML 파일을 파싱하는 중 문제가 발생했습니다: {e}")
        return []

def get_available_models(llm_configs):
    """
    LLM 모델 설정에서 사용 가능한 모델 이름을 반환합니다.
    """
    availabe_types = ['openai', 'google', 'anthropic']
    available_models = []
    availabe_model_dict = {'openai': [], 'google': [], 'anthropic': [], 'huggingface': []}
    for model in llm_configs:
        if model['type'] in availabe_types:
            available_models.append(model['name'])
            if model['type'] == 'openai':
                availabe_model_dict['openai'].append(model['name'])
                availabe_model_dict[model['name']] = 'openai'
            elif model['type'] == 'google':
                availabe_model_dict['google'].append(model['name'])
                availabe_model_dict[model['name']] = 'google'
            elif model['type'] == 'anthropic':
                availabe_model_dict['anthropic'].append(model['name'])
                availabe_model_dict[model['name']] = 'anthropic'
            else:
                pass # HuggingFace는 현재 주석 처리됨
    return available_models, availabe_model_dict

def get_llm_model(model_name, llm_configs):
    """
    주어진 모델 이름에 해당하는 LLM 모델 설정을 찾고 초기화합니다.
    """

    for config in llm_configs:
        if config.get('name') == model_name:
            model_type = config.get('type')
            if model_type == 'openai':
                return ChatOpenAI(
                        model=config.get('model_name', "gpt-4.1-mini-2025-04-14"),
                        temperature=config.get('temperature', 0.7),
                        max_tokens=config.get('max_tokens', 1000)
                    )
            elif model_type == 'google':
                return ChatGoogleGenerativeAI(
                        model=config.get('model_name', "gemini-2.0-flash"),
                        temperature=config.get('temperature', 0.7),
                        max_tokens=config.get('max_tokens', 1000)
                    )
            elif model_type == 'anthropic':
                return ChatAnthropic(
                        model=config.get('model_name', "claude-3-5-haiku-20241022"),
                        temperature=config.get('temperature', 0.7),
                        max_tokens=config.get('max_tokens', 1000)
                    )
            '''
            elif model_type == 'huggingface':
                # huggingface_api_key = os.getenv("HF_API_KEY", config.get('api_key'))
                return ChatHuggingFace(
                    model_name=config.get('model_name'),
                    api_key=config.get('api_key') # 실제로는 os.getenv() 사용 권장
                )
            '''
    print(f"경고: 알 수 없는 LLM 모델 타입 '{model_type}'입니다.")
    return None

llm_model_configs = load_llm_configs()
subagent_llm_model_configs = load_llm_configs("config/subagent_llm_models.yaml")
availabe_models, availabe_models_dict = get_available_models(llm_model_configs)
