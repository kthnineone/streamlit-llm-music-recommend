import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
from langsmith import Client
from .evaluate import *
from .evaluate_llm import *
import pymongo

def display_constraints_eval_result(agent_data=None):
    #st.set_page_config(layout="wide")
    st.title("🎵 LLM 기반 음악 추천 서비스 제약 조건 준수 평가")

    st.markdown("""
    이 대시보드는 LLM 기반 음악 추천 서비스의 특정 제약 조건 준수 여부를 평가합니다.
    주어진 에이전트 데이터를 분석하여 '사용자 선호도 미반영' 및 '이전 추천곡 재추천 금지'와 같은
    핵심 제약 조건이 얼마나 잘 지켜졌는지 시각적으로 보여줍니다.
    """)

    st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 24px; /* 값(숫자)의 폰트 크기 */
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 14px; /* 라벨(텍스트)의 폰트 크기 */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    if agent_data:
        st.header("1. 에이전트 데이터 요약")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("사용자 ID", agent_data.get('user_id', 'N/A'))
        with col2:
            st.metric("채팅 이름", agent_data.get('chat_name', 'N/A'))
        with col3:
            st.metric("총 메시지 턴 수", agent_data.get('num_chat_turn', 'N/A'))

        st.subheader("초기 사용자 요청")
        initial_human_message = ""
        if agent_data and 'messages' in agent_data and len(agent_data['messages']) > 0:
            if agent_data['messages'][0].get('type') == 'human':
                initial_human_message = agent_data['messages'][0]['content']
        st.code(initial_human_message, language='text')

        st.header("2. 제약 조건 평가 결과")
        
        constraint_results = evaluate_constraints(agent_data)

        # 평가 결과를 데이터프레임으로 변환하여 시각화
        results_df = pd.DataFrame([constraint_results]).T
        results_df.columns = ['준수 여부']
        
        # 시각화를 위한 색상 매핑
        def get_status_color(value):
            if value is True:
                return ":green[✔ 준수]"
            elif value is False:
                return ":red[✖ 위반]"
            else:
                return ":orange[❓ 정보 없음]"

        st.markdown("### 주요 제약 조건")
        st.table(results_df.map(get_status_color))

        st.markdown("### 상세 평가")
        st.write("각 제약 조건에 대한 상세 평가 결과입니다.")
        
        st.subheader("사용자 선호도 미반영 (user_preference_reflected)")
        if constraint_results['사용자 선호도 미반영'] is True:
            st.success("✅ **준수:** `user_preference_reflected` 값이 `False`로 설정되어 사용자 기존 선호도가 반영되지 않았습니다.")
        elif constraint_results['사용자 선호도 미반영'] is False:
            st.error("❌ **위반:** `user_preference_reflected` 값이 `True`로 설정되어 사용자 기존 선호도가 반영되었을 수 있습니다.")
        else:
            st.warning("❓ **정보 없음:** `user_preference_reflected` 정보가 없습니다.")
        
        if constraint_results['초기 요청: 선호도 미반영 명시']:
            st.info("ℹ️ 초기 사용자 요청에 'Do not incorporate the user's existing preferences.' 문구가 명시되어 있습니다.")
        else:
            st.warning("⚠️ 초기 사용자 요청에 'Do not incorporate the user's existing preferences.' 문구가 명시되지 않았습니다.")

        st.subheader("이전 추천곡 재추천 금지 (re_recommendation_allowed)")
        if constraint_results['이전 추천곡 재추천 금지'] is True:
            st.success("✅ **준수:** `re_recommendation_allowed` 값이 `False`로 설정되어 이전 추천곡 재추천이 허용되지 않았습니다.")
        elif constraint_results['이전 추천곡 재추천 금지'] is False:
            st.error("❌ **위반:** `re_recommendation_allowed` 값이 `True`로 설정되어 이전 추천곡 재추천이 허용되었을 수 있습니다.")
        else:
            st.warning("❓ **정보 없음:** `re_recommendation_allowed` 정보가 없습니다.")

        if constraint_results['초기 요청: 재추천 금지 명시']:
            st.info("ℹ️ 초기 사용자 요청에 'Do NOT allow re-recommendation of previously recommended songs.' 문구가 명시되어 있습니다.")
        else:
            st.warning("⚠️ 초기 사용자 요청에 'Do NOT allow re-recommendation of previously recommended songs.' 문구가 명시되지 않았습니다.")
            
        st.subheader("최종 추천 결과 분석")
        if constraint_results['최종 추천 완료 여부']:
            st.success(f"✅ 최종 추천이 완료되었습니다. 추천된 곡 수: {constraint_results['추천된 곡 수']}곡")
            st.markdown("### 최종 추천 메시지")
            st.write(f"{constraint_results['최종 추천 메시지']}")
            if constraint_results['추천 메시지: 이전 추천 방지 문구']:
                st.info("ℹ️ 최종 추천 메시지에 '이전에 추천하지 않았던 곡들 중에서'라는 문구가 포함되어 있습니다.")
            else:
                st.warning("⚠️ 최종 추천 메시지에 '이전에 추천하지 않았던 곡들 중에서'라는 명시적인 문구가 포함되어 있지 않습니다.")
        else:
            st.error("❌ 최종 추천이 완료되지 않았거나 `RECOMMENDATION_COMPLETE` 마커를 찾을 수 없습니다.")

        st.header("3. 원본 에이전트 데이터")
        st.json(agent_data) # 원본 JSON 데이터 표시

    else:
        st.warning("에이전트 데이터를 로드하는 데 실패했습니다. 코드의 `AGENT_DATA_STR` 변수를 확인해주세요.")


def display_llm_performance(agent_data=None):
    #st.set_page_config(layout="wide")
    st.title("📊 LLM 기반 음악 추천 서비스 성능 평가: 효율성, 안정성, 레이턴시")

    st.markdown("""
    이 대시보드는 LLM 기반 음악 추천 서비스의 핵심 성능 지표인 효율성 (토큰 사용량),
    안정성 (오류 발생 여부), 그리고 응답 속도 (레이턴시)를 분석하고 시각화합니다.
    """)

    st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 18px; /* 값(숫자)의 폰트 크기 */
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 14px; /* 라벨(텍스트)의 폰트 크기 */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    if agent_data and 'messages' in agent_data:
        #df_metrics = extract_llm_metrics(agent_data)
        df_metrics = extract_single_chat_metrics(agent_data)
        print(f'\n---- df_metrics load complete ----\n')
        print(df_metrics)

        if not df_metrics.empty:
            st.header("1. 주요 성능 요약")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 LLM 호출 수", len(df_metrics))
            with col2:
                st.metric("총 프롬프트 토큰", df_metrics['prompt_tokens'].sum())
            with col3:
                st.metric("총 완료 토큰", df_metrics['completion_tokens'].sum())
            with col4:
                st.metric("총 토큰", df_metrics['total_tokens'].sum())

            col5, col6 = st.columns(2)
            with col5:
                st.metric("전체 오류 발생 수", df_metrics['error_count'].sum())
            with col6:
                st.metric("모델별 평균 토큰 사용량", f"{df_metrics.groupby('model_name')['total_tokens'].mean().round(2).to_dict()}", help="모델별 평균 토큰 사용량입니다.")

            st.header("2. 토큰 사용량 분석")
            st.markdown("---")
            st.subheader("에이전트별 토큰 사용량")
            
            # 에이전트별 토큰 사용량 바 차트
            fig_tokens_by_agent = px.bar(
                df_metrics.groupby('agent_name')[['prompt_tokens', 'completion_tokens', 'total_tokens']].sum().reset_index(),
                x='agent_name',
                y=['prompt_tokens', 'completion_tokens', 'total_tokens'],
                title='에이전트별 총 토큰 사용량',
                labels={'value': '토큰 수', 'variable': '토큰 유형', 'agent_name': '에이전트'},
                barmode='group'
            )
            st.plotly_chart(fig_tokens_by_agent, use_container_width=True)

            st.subheader("개별 LLM 호출별 토큰 사용량")
            st.dataframe(df_metrics[['agent_name', 'model_name', 'prompt_tokens', 'completion_tokens', 'total_tokens']].sort_values(by='total_tokens', ascending=False))

            st.subheader("LLM 호출 예상 비용")
            def load_cost_data(agent_data):
                redis_key = agent_data['redis_key']
                print(f'In Loading Cost data, redis_key: {redis_key}')
                MONGO_URI = 'mongodb://localhost:27017/'
                MONGO_DB_NAME = 'llm_monitor_db'
                MONGO_COLLECTION_NAME = 'llm_cost'
                try:
                    mongo_client = pymongo.MongoClient(MONGO_URI)
                    mongo_db_monitor = mongo_client[MONGO_DB_NAME]
                    mongo_collection_cost = mongo_db_monitor[MONGO_COLLECTION_NAME]
                    print("MongoDB 서버에 성공적으로 연결되었습니다.")
                except pymongo.errors.ConnectionFailure as e:
                    print(f"MongoDB 서버 연결 실패: {e}")
                    print("MongoDB 서버가 실행 중인지 확인해 주세요.")
                    exit()
                query = {'redis_key': redis_key}
                cost_result = list(mongo_collection_cost.find(query))
                return cost_result

            cost_data = load_cost_data(agent_data)
            if isinstance(cost_data, list):
                cost_data = cost_data[0]
            print(f'cost_data: {cost_data}')
            col7, col8, col9 = st.columns(3)
            with col7:
                st.metric("LLM 서비스 타입", cost_data['llm_service_type'])
            with col8:
                st.metric("supervisor model", cost_data['supervisor_llm_model'])
            with col9:
                st.metric("subagent model", cost_data['subagent_llm_model'])
            
            col10, col11 = st.columns(2)
            with col10:
                st.metric("채팅 턴 수", cost_data['num_chat_turn'])
            with col11:
                st.metric("총 비용 (USD)", cost_data['total_cost']['total_cost'])
            

            def get_df_cost(cost_data):
                df_data = {'web_search_agent': cost_data['web_search_agent'],
                        'load_preference_agent': cost_data['load_preference_agent'],
                        'supervisor_agent': cost_data['supervisor_agent'],
                        'total_cost': cost_data['total_cost'],
                        }

                df_cost = pd.DataFrame(df_data)
                return df_cost
            
            st.dataframe(get_df_cost(cost_data))


            st.header("3. 응답 속도 (레이턴시) 분석")
            st.markdown("---")
            st.warning("⚠️ **주의:** 현재 레이턴시 값은 추가 분석이 필요합니다.")
            
            # 'latency_proxy'라는 목업 데이터
            st.subheader("LLM 호출 순서별 레이턴시 (초)")
            fig_latency = px.line(
                df_metrics,
                x='message_id',
                y='latency', 
                title='LLM 호출 순서별 레이턴시',
                labels={'message_id': '메시지 ID', 'latency': '레이턴시 (초)'},
                hover_data=['agent_name', 'model_name', 'total_tokens']
            )
            st.plotly_chart(fig_latency, use_container_width=True)

            st.subheader("에이전트별 평균 레이턴시 (초)")
            print(f'df_metrics:\n{df_metrics[['agent_name', 'latency']]}')
            avg_latency_by_agent = df_metrics.groupby('agent_name')['latency'].mean().reset_index()
            print(f'avg_latency_by_agent:\n{avg_latency_by_agent}')
            fig_avg_latency = px.bar(
                avg_latency_by_agent,
                x='agent_name',
                y='latency',
                title='에이전트별 평균 레이턴시 (프록시)',
                labels={'latency': '평균 레이턴시 (초)', 'agent_name': '에이전트'}
            )
            st.plotly_chart(fig_avg_latency, use_container_width=True)


            st.header("4. 안정성 (오류) 분석")
            st.markdown("---")
            
            # 에러 유형별 통계
            error_counts = pd.DataFrame({
                '오류 유형': ['거부 오류 (Refusal)', '유효하지 않은 툴 호출 (Invalid Tool Calls)'],
                '발생 횟수': [df_metrics['refusal_error'].sum(), df_metrics['invalid_tool_calls_error'].sum()]
            })
            st.table(error_counts)

            total_errors = df_metrics['error_count'].sum()
            if total_errors > 0:
                st.error(f"❌ **총 {total_errors} 건의 LLM 관련 오류가 발생했습니다.**")
                st.subheader("오류 발생 LLM 호출 상세")
                st.dataframe(df_metrics[df_metrics['error_count'] > 0][['message_id', 'agent_name', 'model_name', 'refusal_error', 'invalid_tool_calls_error', 'content', 'additional_kwargs']])
            else:
                st.success("✅ **모든 LLM 호출이 성공적으로 완료되었으며, 오류가 감지되지 않았습니다.**")

            st.header("5. 원본 에이전트 데이터 (LLM 응답 부분)")
            # 모든 AI 메시지의 원본 데이터를 보여줌
            def get_llm_responses(raw_data):
                llm_responses = []
                for msg_idx, messages in enumerate(raw_data.get('messages', [])):
                    for message in messages.get('messages', []):
                        message = message.get('kwargs', {})
                        if message.get('type') == 'ai':
                            llm_responses.append(message)
                return llm_responses

            llm_responses = get_llm_responses(agent_data)

            if llm_responses:
                st.json(llm_responses)
            else:
                st.info("LLM 응답 데이터가 없습니다.")

        else:
            st.warning("분석할 LLM 호출 데이터가 없습니다. `messages` 필드를 확인해주세요.")
    else:
        st.warning("에이전트 데이터를 로드하는 데 실패했거나 'messages' 필드가 없습니다. 코드의 `AGENT_DATA_STR` 변수를 확인해주세요.")



def display_llm_component_eval(agent_data=None):
    st.title("📊 LLM 웹 서치와 정답 성능 평가")
    st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 24px; /* 값(숫자)의 폰트 크기 */
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 14px; /* 라벨(텍스트)의 폰트 크기 */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    if agent_data:
        evaluations = get_llm_evaluations(agent_data)
        # --- LLM 서비스 정보 표시 ---
        st.header("⚙️ LLM 에이전트 구성 정보")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("LLM 서비스 유형", evaluations['llm_service_type'])
            st.metric("Supervisor 모델", evaluations['supervisor_model'])
        with col2:
            st.metric("검색 도구 이름", evaluations['search_tool_name'])
            st.metric("Subagent 모델", evaluations['subagent_model'])
        with col3:
            st.metric("임베딩 모델", evaluations['embedding_model'])
            st.metric("임베딩 유형", evaluations['embedding_type'])

        st.markdown("---")

        # --- 평가 지표 요약 표 ---
        st.header("📈 LLM 평가 지표 요약")

        # 데이터프레임으로 변환하여 표로 표시
        eval_df = pd.DataFrame({
            '지표': [
                '웹 검색 맥락 관련성 점수',
                '응답 관련성 점수',
                '응답 근거 점수'
            ],
            '점수': [
                evaluations['web_search_context_relevance_score'],
                evaluations['response_relevancy_score'],
                evaluations['mean_response_groundedness_score']
            ]
        })
        # 점수를 보기 좋게 포맷팅
        eval_df['점수'] = eval_df['점수'].map(lambda x: f"{x:.2f}")

        st.dataframe(eval_df, hide_index=True, use_container_width=True)

        st.markdown("---")

        # --- 평가 지표 시각화 ---
        st.header("📊 세부 평가 지표 시각화")

        # 평가 지표를 게이지 차트로 시각화 (각 점수가 0에서 1 사이임을 가정)
        def create_gauge_chart(title, score):
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"<span style='font-size:1.2em'>{title}</span>"},
                gauge = {
                    'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.5], 'color': 'lightgray'},
                        {'range': [0.5, 0.75], 'color': 'lightblue'},
                        {'range': [0.75, 1], 'color': 'lightgreen'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': score}}))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
            return fig

        col_metric1, col_metric2, col_metric3 = st.columns(3)

        with col_metric1:
            st.plotly_chart(create_gauge_chart("웹 검색 맥락 관련성", evaluations['web_search_context_relevance_score']), use_container_width=True)

        with col_metric2:
            st.plotly_chart(create_gauge_chart("응답 관련성", evaluations['response_relevancy_score']), use_container_width=True)

        with col_metric3:
            st.plotly_chart(create_gauge_chart("응답 근거 점수", evaluations['mean_response_groundedness_score']), use_container_width=True)

        st.markdown("---")

        # --- 추가 정보 (선택 사항) ---
        st.info("이 대시보드는 LLM 에이전트의 주요 평가 지표를 시각화하여 신속한 성능 분석을 돕습니다.")
        st.caption("데이터는 예시이며, 실제 LLM 평가 시스템에 따라 데이터를 연동해야 합니다.")


def disply_langsmith():
    #st.set_page_config(layout="wide")

    # LangSmith 클라이언트 초기화
    # 환경 변수를 통해 API 키를 로드합니다.
    # LANGCHAIN_API_KEY 환경 변수가 설정되어 있어야 합니다.
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true" # Tracing V2 활성화

    try:
        client = Client()
    except Exception as e:
        st.error(f"LangSmith 클라이언트 초기화 실패: {e}")
        st.info("LANGCHAIN_API_KEY 환경 변수가 올바르게 설정되었는지 확인해주세요.")
        st.stop()

    st.title("🚀 LangSmith 모니터링 대시보드")

    st.sidebar.header("필터 설정")

    # 기간 필터
    time_range_options = {
        "지난 1시간": timedelta(hours=1),
        "지난 6시간": timedelta(hours=6),
        "지난 24시간": timedelta(hours=24),
        "지난 7일": timedelta(days=7),
        "모두": None
    }
    selected_time_range_label = st.sidebar.selectbox("기간 선택", list(time_range_options.keys()))
    selected_timedelta = time_range_options[selected_time_range_label]

    # 프로젝트 이름 필터 (선택 사항)
    project_name = st.sidebar.text_input("프로젝트 이름 (선택 사항)", value="")

    # Trace 데이터 가져오기
    @st.cache_data(ttl=600) # 10분마다 캐시 갱신
    def get_langsmith_traces(project_name=None, selected_timedelta=None):
        runs_data = []
        
        # 시간 필터 적용
        if selected_timedelta:
            start_time = datetime.now() - selected_timedelta
        else:
            start_time = None # 모든 기간
            
        try:
            # runs.list()를 사용하여 Run 객체들을 가져옵니다.
            # 이터레이터이므로 반복하여 모든 Run을 가져와야 합니다.
            for run in client.list_runs(project_name=project_name, start_time=start_time):
                runs_data.append({
                    "id": run.id,
                    "name": run.name,
                    "run_type": run.run_type,
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "status": run.status,
                    "feedback_score": run.feedback_score,
                    "error": run.error,
                    "latency_ms": (run.end_time - run.start_time).total_seconds() * 1000 if run.start_time and run.end_time else None
                })
            return pd.DataFrame(runs_data)
        except Exception as e:
            st.error(f"LangSmith 데이터 로드 실패: {e}")
            return pd.DataFrame()

    df = get_langsmith_traces(project_name=project_name if project_name else None, selected_timedelta=selected_timedelta)

    if not df.empty:
        st.subheader("📊 전체 모니터링 개요")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 실행 수", len(df))
        with col2:
            successful_runs = df[df['status'] == 'completed']
            st.metric("성공한 실행", len(successful_runs))
        with col3:
            failed_runs = df[df['status'] == 'failed']
            st.metric("실패한 실행", len(failed_runs))
        with col4:
            avg_latency = df['latency_ms'].mean()
            st.metric("평균 지연 시간 (ms)", f"{avg_latency:.2f}" if pd.notna(avg_latency) else "N/A")

        st.subheader("테이블 형식 데이터")
        st.dataframe(df.sort_values(by="start_time", ascending=False), use_container_width=True)

        st.subheader("📈 실행 타입별 분포")
        run_type_counts = df['run_type'].value_counts()
        st.bar_chart(run_type_counts)

        st.subheader("🚨 실패한 실행 목록")
        if not failed_runs.empty:
            st.dataframe(failed_runs[['name', 'run_type', 'start_time', 'error']], use_container_width=True)
        else:
            st.info("실패한 실행이 없습니다.")

        st.subheader("⏳ 시간 경과에 따른 지연 시간 (상위 20개)")
        # 실행 시간이 있는 경우에만 필터링
        df_with_latency = df.dropna(subset=['latency_ms'])
        if not df_with_latency.empty:
            # 상위 20개 실행만 시각화 (너무 많으면 차트가 복잡해짐)
            top_20_latency = df_with_latency.sort_values(by="latency_ms", ascending=False).head(20)
            st.bar_chart(top_20_latency.set_index('name')['latency_ms'])
        else:
            st.info("지연 시간을 측정할 수 있는 데이터가 없습니다.")

    else:
        st.warning("데이터를 가져오지 못했거나 필터 조건에 맞는 데이터가 없습니다.")
        st.info("LangSmith에 프로젝트가 존재하고 실행 내역이 있는지 확인해주세요.")