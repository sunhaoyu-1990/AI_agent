'''
@file: chroma_db.py
@breif: create a database for chroma, and store the data in the database.
@author: Sunhaoyu
@create: 2024.09.09
@update: 2024.09.09
'''

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class ChromaDB:
    def __init__(self):
        # 使用 OpenAI 的嵌入模型
        self.embedding_function = OpenAIEmbeddings()
        
        # 初始化 Chroma 向量数据库
        self.vectorstore = Chroma(
            collection_name="example_collection",
            embedding_function=self.embedding_function,
            persist_directory="./application/RAG/chroma_langchain_db",  # 数据库保存位置
        )

    def create_db(self):
        # 从网页加载数据
        docs = self.load_data()
        for doc in docs:
            self.update_db(doc)

    def load_data(self):
        # 读取document下的所有文档
        docs = []
        file_list = os.listdir("./application/RAG/document")
        for file in file_list:
            with open(f"./application/RAG/document/{file}", 'r', encoding='utf-8') as f:
                doc = f.read()
                docs.append(doc)
        return docs

    def update_db(self, doc):
        """
        更新数据库，将新文档分割并添加到向量数据库中。
        :param doc: 文本文档
        """
        # 使用文本分割器
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=True)
        all_splits = text_splitter.split_text(doc)
        
        # 将分割的文档添加到向量数据库中
        self.vectorstore.add_texts(all_splits)
        self.vectorstore.persist()  # 保存更新后的数据库

    def get_retriever(self, search_type='similarity', search_kwargs={'k': 5}):
        """
        获取向量数据库的检索器，用于检索文档
        :param search_type: 检索类型，例如 'similarity'
        :param search_kwargs: 检索参数，例如 {'k': 5}
        :return: 向量数据库检索器
        """
        retriever = self.vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        return retriever

    def get_retrieved_docs(self, input):
        """
        根据输入文本检索相关文档
        :param input: 用户输入的查询文本
        :return: 检索到的相关文档列表
        """
        retriever = self.get_retriever()
        retrieved_docs = retriever.get_relevant_documents(input)
        return retrieved_docs


if __name__ == '__main__':
    chroma_db = ChromaDB()
    print(chroma_db.get_retrieved_docs("主要负责人有哪些责任？"))