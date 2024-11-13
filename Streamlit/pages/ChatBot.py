from logging import PlaceHolder
import streamlit as st
from langchain_chroma import Chroma
#from langchain_huggingface import HuggingFaceEmbeddings
from urllib import response
from requests import session
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
 
import requests
from bs4 import BeautifulSoup

def fetch_page_content(url):
    try:
        reponse = requests.get(url)
        reponse.raise_for_status()
        return reponse.text    
    except requests.RequestException as e:
        return f"Error fetching page content : {e}"

def extract_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return ' '.join(s.text.strip() for s in soup.find_all('p'))  # 모든 <p> 태그의 텍스트를 추출


st.caption("You are the masterpiece in the making")
st.title(":red[데이터셋 추천] 및 :red[활용 방안] :sunglasses:")
st.subheader("자세한 내용을 물어보세요!")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}





# 사이드 바
with st.sidebar:
    tab1, tab2 = st.tabs(["데이터 링크", "프리셋"])
    
    dataset_link = tab1.text_area("Datasets URL", placeholder="데이터셋 링크를 입력해주세요.")
    dataset_link_apply_btn = tab1.button("링크 적용", key='apply1')
    # 파일 업로드
    #uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    user_selected_prompt = tab2.selectbox("프리셋 선택", ["데이터셋 분석", "활용방안 추천"])

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )
    get_temperature = st.slider("[ 답변 자유도 ]", 0, 100)
    # 초기화 버튼 생성

    session_id = st.text_input("세션 ID를 입력하세요.", "ibdp")

    clear_btn = st.button("대화 초기화")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]


def create_chain(model_name="gpt-4o"):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),
        ]
    )

    llm = ChatOpenAI(model_name=model_name)

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return chain_with_history

warning_msg = st.empty()

if "chain" not in st.session_state:
    st.session_state['chain'] = create_chain(model_name=selected_model)

if dataset_link_apply_btn and dataset_link:
    html_content = fetch_page_content(dataset_link)
    if html_content.startswith("Error"):
        tab1.markdown(f"⚠️링크 적용에 실패하였습니다.")
    else:
        extracted_text = extract_text(html_content)
        st.session_state["link"] = extracted_text
        chain = st.session_state["chain"]

        if chain is not None:
            question = f"{extracted_text}\n\n위의 내용은 {dataset_link}의 내용을 추출한 내용이야. 잘 알아둬"
            chain.invoke({"question" : question}, config={"configurable" : {"session_id" : session_id}})
        #st.write("Extracted Text:", extracted_text)
            tab1.markdown(f"✅ 링크가 적용되었습니다")
            
        else:
            tab1.markdown(f"⚠️링크 적용에 실패하였습니다.")

user_input = st.chat_input("궁금한 내용을 물어보세요!")

print_messages()

if user_input:
    chain = st.session_state["chain"]
    if chain is not None:
        response = chain.stream(
            {"question" : user_input},
            config={"configurable" : {"session_id" : session_id}},
        )

        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

                # 대화 내용 기록
            add_message("user", user_input)
            add_message("assistant", ai_answer)

    else:
        warning_msg.error("데이터 링크를 적용 해주세요.")
