from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import gspread
import uvicorn
import pandas as pd
from google.oauth2.service_account import Credentials
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка учетных данных Google
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
CREDENTIALS_FILE = "credentials.json"


def load_data_from_google_sheets(spreadsheet_id: str, sheet_name: str) -> pd.DataFrame:
    """
    Загружает данные из Google Sheets в Pandas DataFrame.
    """
    try:
        credentials = Credentials.from_service_account_file(
            CREDENTIALS_FILE, scopes=SCOPES
        )
        gc = gspread.service_account(filename=CREDENTIALS_FILE)
        spreadsheet = gc.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка при загрузке данных: {str(e)}"
        )


@app.get("/data")
async def get_data(spreadsheet_id: str, sheet_name: str) -> List[Dict]:
    """
    Загружает данные из Google Sheets и возвращает их в виде списка словарей.
    """
    try:
        df = load_data_from_google_sheets(spreadsheet_id, sheet_name)
        return df.to_dict(orient="records")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Неизвестная ошибка: {str(e)}")


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def create_agent(df: pd.DataFrame):
    """
    Создает ИИ-агента на основе данных из Pandas DataFrame.
    """
    loader = DataFrameLoader(
        df, page_content_column="название инструмента с сайта realtycalendar.ru"
    )
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)

    return db


@app.get("/ask")
async def ask_question(spreadsheet_id: str, sheet_name: str, question: str) -> str:
    """
    Загружает данные из Google Sheets, создает ИИ-агента и отвечает на вопрос.
    """
    try:
        df = load_data_from_google_sheets(spreadsheet_id, sheet_name)
        db = create_agent(df)

        llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type="stuff")

        docs = db.similarity_search(question)
        answer = chain.run(input_documents=docs, question=question)
        return answer
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка при обработке вопроса: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
