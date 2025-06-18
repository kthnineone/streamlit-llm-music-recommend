import streamlit as st
import pandas as pd


# --- 제약 조건 평가 함수 ---
def evaluate_constraints(data):
    results = {}

    # 1. 사용자 기존 선호도 미반영 (user_preference_reflected: False)
    user_preference_reflected = data.get('user_preference_reflected', None)
    if user_preference_reflected is not None:
        # 'false'로 설정되어 있으면 준수
        results['사용자 선호도 미반영'] = not user_preference_reflected
    else:
        results['사용자 선호도 미반영'] = '정보 없음'

    # 2. 이전에 추천된 곡 재추천 금지 (re_recommendation_allowed: False)
    re_recommendation_allowed = data.get('re_recommendation_allowed', None)
    if re_recommendation_allowed is not None:
        # 'false'로 설정되어 있으면 준수
        results['이전 추천곡 재추천 금지'] = not re_recommendation_allowed
    else:
        results['이전 추천곡 재추천 금지'] = '정보 없음'

    # 3. 사용자 요청 (messages[0]['content']에 명시된 제약 조건 분석)
    # 실제 요청 메시지에서 "Do not incorporate the user's existing preferences."
    # "Do NOT allow re-recommendation of previously recommended songs."
    # 이 두 문구가 존재하는지 확인하여 이중 체크할 수 있습니다.
    initial_user_request = ""
    if data and 'messages' in data and len(data['messages']) > 0 and data['messages'][0]['type'] == 'human':
        initial_user_request = data['messages'][0]['content']

    # 'Do not incorporate the user\'s existing preferences.' 문구 포함 여부
    if "Do not incorporate the user's existing preferences." in initial_user_request:
        results['초기 요청: 선호도 미반영 명시'] = True
    else:
        results['초기 요청: 선호도 미반영 명시'] = False

    # 'Do NOT allow re-recommendation of previously recommended songs.' 문구 포함 여부
    if "Do NOT allow re-recommendation of previously recommended songs." in initial_user_request:
        results['초기 요청: 재추천 금지 명시'] = True
    else:
        results['초기 요청: 재추천 금지 명시'] = False
        
    # 최종 추천 메시지 분석 (RECOMMENDATION_COMPLETE가 있는지, 추천된 곡들이 있는지)
    final_recommendation_message = ""
    recommended_songs = []
    if data and 'messages' in data:
        for message in data['messages']:
            if message.get('type') == 'ai' and "RECOMMENDATION_COMPLETE" in message.get('content', ''):
                final_recommendation_message = message['content']
                # 추천 곡 목록 추출 (간단한 파싱)
                song_list_start = final_recommendation_message.find("추천 곡 목록:")
                if song_list_start != -1:
                    song_list_content = final_recommendation_message[song_list_start:]
                    lines = song_list_content.split('\n')
                    for line in lines:
                        if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')): # 5곡 추천으로 가정
                            recommended_songs.append(line.strip())
                break
    
    results['최종 추천 완료 여부'] = bool(final_recommendation_message)
    results['추천된 곡 수'] = len(recommended_songs)
    
    # 여기서 "이전에 추천하지 않았던 곡들 중에서"라는 문구를 실제 이전 추천 목록과 비교해야 하지만
    # 현재 제공된 데이터만으로는 이전 추천 목록 정보가 없으므로 텍스트 존재 유무로만 판단
    results['추천 메시지: 이전 추천 방지 문구'] = "이전에 추천하지 않았던 곡들 중에서" in final_recommendation_message

    return results


# --- 데이터 파싱 및 지표 추출 함수 ---
def extract_llm_metrics(data):
    metrics_list = []
    
    # 메시지 턴을 순회하며 LLM 관련 메타데이터 추출
    # AI 타입의 메시지에서만 LLM 사용량 정보가 있다고 가정
    for i, message in enumerate(data.get('messages', [])):
        if message.get('type') == 'ai':
            response_metadata = message.get('response_metadata', {})
            usage_metadata = message.get('usage_metadata', {})
            additional_kwargs = message.get('additional_kwargs', {})
            
            # 레이턴시 (응답 시간)
            # 'response_time' 필드가 없으므로, ID의 일부를 사용하여 시간 흐름을 나타냄 (실제 레이턴시는 아님)
            # 실제 레이턴시는 요청 전/후 타임스탬프를 기록하여 계산해야 합니다.
            # 여기서는 예시를 위해 메시지 인덱스를 활용합니다.
            latency = i + 1 # 단순히 메시지 순서를 레이턴시 '척도'로 사용
            
            # 토큰 사용량
            token_usage = response_metadata.get('token_usage', {})
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            total_tokens = token_usage.get('total_tokens', 0)
            
            # 에러 및 안정성
            refusal = additional_kwargs.get('refusal') is not None
            invalid_tool_calls = bool(message.get('invalid_tool_calls'))
            
            # LLM 모델 이름
            model_name = response_metadata.get('model_name', 'N/A')

            # 에이전트 이름 (Supervisor, Web Search 등)
            agent_name = message.get('name', 'Unknown Agent')
            
            metrics_list.append({
                'message_id': message.get('id', f'msg_{i}'),
                'agent_name': agent_name,
                'model_name': model_name,
                'latency_proxy': latency, # 실제 레이턴시 대신 프록시 값
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'refusal_error': refusal,
                'invalid_tool_calls_error': invalid_tool_calls,
                'error_count': (1 if refusal else 0) + (1 if invalid_tool_calls else 0)
            })
    return pd.DataFrame(metrics_list)
    