import logging
import sys
import pprint  # pprint를 사용하여 구조화된 데이터를 보기 좋게 포매팅
import time # 스트림 생성 시 지연을 시뮬레이션하기 위해 사용

# 1. 로거 설정
def setup_logger(name='langgraph_logger', log_file='langgraph_stream.log', level=logging.INFO):
    """
    LangGraph 스트림 메시지를 기록할 로거를 설정합니다.

    Args:
        name (str): 로거의 이름.
        log_file (str): 로그 파일의 경로.
        level (int): 로깅 레벨 (예: logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: 설정된 로거 객체.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 핸들러가 중복으로 추가되는 것을 방지
    if not logger.handlers:
        # 파일 핸들러 설정: 로그를 파일에 기록
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 콘솔 핸들러 설정: 로그를 콘솔에 출력
        #console_handler = logging.StreamHandler(sys.stdout)
        #console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        #console_handler.setFormatter(console_formatter)
        #logger.addHandler(console_handler)

    return logger

# 2. pprint를 사용하여 데이터 포매팅 후 로깅
def log_structured_data(logger, data, level=logging.INFO, message_prefix="stream_output", model_name=None):
    """
    구조화된 데이터를 pprint 형식으로 포매팅하여 로그에 기록합니다.

    Args:
        logger (logging.Logger): 사용할 로거 객체.
        level (int): 로깅 레벨 (예: logging.INFO).
        message_prefix (str): 로그 메시지 앞에 붙일 접두사.
        data (any): 기록할 데이터 (딕셔너리, 리스트 등).
    """
    # pprint.pformat()을 사용하여 데이터를 보기 좋게 문자열로 변환
    # indent=4는 들여쓰기 수준을 4칸으로 설정하여 가독성을 높입니다.
    formatted_data = pprint.pformat(data, indent=4)

    # 변환된 문자열을 로그 메시지에 포함하여 기록
    # f-string을 사용하여 접두사와 포매팅된 데이터를 함께 전달
    if level == logging.INFO:
        logger.info(f"{message_prefix} - {model_name}\n{formatted_data}")
    elif level == logging.DEBUG:
        logger.debug(f"{message_prefix} - {model_name}\n{formatted_data}")
    # 다른 레벨에 대해서도 필요에 따라 추가할 수 있습니다.
    else:
        logger.log(level, f"{message_prefix}\n{formatted_data}")