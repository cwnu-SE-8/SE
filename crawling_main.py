from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from page import process_page
from util import link_page
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
import os
from dotenv import load_dotenv
load_dotenv()

# 경고 무시
warnings.filterwarnings("ignore")

model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

from selenium.webdriver.chrome.options import Options

# headless 모드 설정
chrome_options = Options()
# (optional) 창을 표시하지 않고 실행하려면 추가
chrome_options.add_argument("--headless")
# (optional) GPU 사용 안 함
chrome_options.add_argument("--disable-gpu")
# (optional) 노-샌드박스 모드로 실행
chrome_options.add_argument(f"--no-sandbox")
# (optional) 소프트웨어 래스터라이저 사용 안 함
chrome_options.add_argument(f"--disable-software-rasterizer")
# (optional) dev/shm 사용 안 함
chrome_options.add_argument(f"--disable-dev-shm-usage")
# (optional) 확장 프로그램 비활성화
chrome_options.add_argument(f"--disable-extensions")

# 웹 드라이버 설정
service = Service(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

page = 1


DB_PATH = '/home/user/SE_Project/chorma_db_2'

persist_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=hf_embeddings,
        collection_name="SE_db",
)

while True:
    # 웹사이트 열기
    driver.get(link_page(page))

    while page < 44:
        process_page(link_page(page), driver, persist_db)
        print(f"Finish Page{page}")
        page += 1

        
# 브라우저 닫기
driver.quit()