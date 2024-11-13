import streamlit as st

from typing import List, Union
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from dotenv import load_dotenv
load_dotenv()

import time
import os

DB_PATH = '/home/user/SE_Project/chorma_db'

st.caption("You are the masterpiece in the making")
# Streamlit UI 디자인
st.title(":red[데이터셋 추천] 및 :red[활용 방안] :sunglasses:")

if "messages" not in st.session_state:
    st.session_state["messages"] = [] # 대화 내용을 저장할 리스트 초기화

# 상수 정의
class MessageRole:
    """
    메시지 역할을 정의하는 클래스입니다.
    """
    USER = "user",
    ASSISTANT = "assistant"

class MessageType:
    """
    메시지 유형을 정의하는 클래스입니다.
    """

    TEXT = "text"
    FIGURE = "figure"
    CODE = "code"
    URL = "url"

# 사이드바 생성

st.subheader("분석하고 싶은 내용을 물어보세요!")

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
#     """
#     새로운 메시지를 저장하는 함수입니다.

#     Args:
#         role (MessageRole): 메시지 역할 (사용자 또는 어시스턴트)
#         content (List[Union[MessageType, str]]): 메시지 내용
#     """
#     messages = st.session_state["messages"]
    
#     if messages and messages[-1][0] == role:
#         messages[-1][1].extend([content])  # 같은 역할의 연속된 메시지는 하나로 합칩니다
#     else:
#         messages.append([role, [content]])  # 새로운 역할의 메시지는 새로 추가합니다

def ask(query, state):
    """
    사용자의 질문을 처리하고 응답을 생성하는 함수입니다.

    Args:
        query (str) : 사용자의 질문
        state (str) : 사용자의 전문성
    """

    if "agent" in st.session_state:
        st.chat_message("user").write(query)
        add_message(MessageRole.USER, [MessageType.TEXT, query])
        
        agent = st.session_state["agent"]
        response = agent.stream({"input": query})

        with st.chat_message("assistant"):
            st.write(response)

# LLM으로부터 추천 데이터셋과 활용 방안을 가져오는 함수임
# def generate_recommendations(user_type):
#     prompts = {
#         "일반인": "일반인을 위한 데이터셋과 활용 방안을 추천해 주세요.",
#         "대학(부)원생": "대학(부)원생이 사용할 수 있는 데이터셋과 연구 방안을 추천해 주세요.",
#         "직장인": "직장인에게 유용한 데이터셋과 분석 방안을 추천해 주세요.",
#         "전문가": "전문가에게 적합한 고급 데이터셋과 활용 방안을 추천해 주세요."
#     }
#     prompt = prompts.get(user_type, "데이터셋과 활용 방안을 추천해 주세요.")

#     # LLM API를 호출하여 데이터셋 추천과 활용 방안 결과를 반환한다고 가정함
#     # dataset_recommendations, usage_suggestions = llm_api_call(prompt)

#     # 예시 
#     dataset_recommendations = f"{user_type}에게 추천하는 데이터셋 목록입니다. (예시)"
#     usage_suggestions = f"{user_type}가 활용할 수 있는 데이터셋 활용 방안입니다. (예시)"

    # return dataset_recommendations, usage_suggestions


# 사용자 전문성 선택영역임
#user_type = st.selectbox("사용자 전문성을 선택하세요:", ["일반인", "대학(부)원생", "직장인", "전문가"])

# 버튼 클릭 시 LLM을 통해 데이터셋 추천 및 활용 방안 생성
# if st.button("추천 받기"):
#     dataset_recommendations, usage_suggestions = generate_recommendations(user_type)

#     # 탭을 사용하여 데이터셋 추천과 활용 방안을 별도의 화면에 표시
#     tab1, tab2 = st.tabs(["추천 데이터셋", "활용 방안"])

#     with tab1:
#         st.subheader("추천 데이터셋")
#         st.write(dataset_recommendations)

#     with tab2:
#         st.subheader("데이터셋 활용 방안")
#         st.write(usage_suggestions)


warning_msg = st.empty()


def process_user_info(user):
    return RunnableLambda(lambda x: f"User Type: {user}")
# 체인 생성
def create_chain(selected_user, model_name="gpt-4o-mini",):
    embedding_model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    persist_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=hf_embeddings,
        collection_name="SE_db",
    )

    retriever = persist_db.as_retriever() # 어떤 방식으로 가져올지 수정 예정
    
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("/home/user/SE_Project/Streamlit/prompts/datasets.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "selected_user" : process_user_info(selected_user), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


print_messages()



# if user_input := st.chat_input():
#     print(user_input)
#     print(selected_user)
#     #chain = create_chain(selected_user)
#     #st.session_state["chain"] = chain
#     # chain 생성
#     #ask(user_input, selected_user)
#     chain = st.session_state["chain"]

#     if chain is not None:
#         # 사용자의 입력
#         st.chat_message("user").write(user_input)
#         # 스트리밍 호출
#         response = chain.stream(user_input)
#         with st.chat_message("assistant"):
#             # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
#             container = st.empty()

#             ai_answer = ""
#             for token in response:
#                 ai_answer += token
#                 container.markdown(ai_answer)

#         # 대화기록을 저장한다.
#         add_message("user", user_input)
#         add_message("assistant", ai_answer)
#         print("대화기록 저장 완료")
#         print(st.session_state)



with st.sidebar:
    # 사이드바에 물어보기
    selected_user = st.selectbox(
        "사용자 전문성", ["일반인", "개발자"], index=0
    )
#    messages = st.container(height=600)

    
    get_temperature = st.slider("[ 답변 자유도 ]", 0, 100)
    #st.caption("You are the masterpiece in the making")
    clear_btn = st.button("대화 초기화")
    # with st.spinner("Loading..."):
        # time.sleep(1)

if "chain" not in st.session_state:
    st.session_state['chain'] = create_chain(selected_user)

if clear_btn:
    st.session_state["messages"] = []  # 대화 내용 초기화

if user_input := st.chat_input():
        print(user_input)
        #chain = create_chain(selected_user)
        #st.session_state["chain"] = chain
        # chain 생성
        #ask(user_input, selected_user)
        chain = st.session_state["chain"]

        if chain is not None:
            # 사용자의 입력
            st.chat_message("user").write(user_input)
            # 스트리밍 호출
            response = chain.stream(user_input)
            with st.chat_message("assistant"):
                # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
                container = st.empty()

                ai_answer = ""
                for token in response:
                    ai_answer += token
                    container.markdown(ai_answer)

            # 대화기록을 저장한다.
            add_message("user", user_input)
            add_message("assistant", ai_answer)
            print("대화기록 저장 완료")
            print(st.session_state)