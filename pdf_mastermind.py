import os
from os.path import join, dirname
from dotenv import load_dotenv

import openai
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader


load_dotenv(verbose=True)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


def get_pages(path: str):
  loader = PyPDFLoader(path)
  pages = loader.load_and_split()
  return pages


def get_llm():
  openai.api_key = os.environ.get('OPENAI_API_KEY')
  llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
  return llm


def get_vectorstore(pages: langchain.schema.Document):
  embeddings = OpenAIEmbeddings()
  vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory="vector")
  vectorstore.persist()
  return vectorstore


def pdf_qa(llm, vectorstore):
  pdf_qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

  chat_history = []
  while True:
    query = input('質問を入力してください(\qで終了)：')

    if query == '\q':
      print('プロンプトを終了します')
      break
    else:
      result = pdf_qa({"question": query, "chat_history": chat_history})
      print(result['answer'])

      chat_history.append((query, result['answer']))


def main(path: str):
  pages = get_pages(path=path)
  llm = get_llm()
  vectorstore = get_vectorstore(pages=pages)
  pdf_qa(llm=llm, vectorstore=vectorstore)


if __name__ == '__main__':
  import sys

  try:
    pdf_path = sys.argv[1]
  except:
    sys.exit('USAGE: python pdf_mastermind </path/to/pdf>')
  
  main(path=pdf_path)
  