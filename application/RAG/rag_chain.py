'''
@file: rag_chain.py
@breif: This file is the main file for RAG chain.
@author: Sunhaoyu
@create: 2024.09.10
@update: 2024.09.10
'''

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from application.RAG.chroma_db import ChromaDB
from langchain import hub

class CodeGenerateAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        chroma_db = ChromaDB()
        self.retriever = chroma_db.get_retriever()
        self.prompt = hub.pull("rlm/rag-prompt")
        self.custom_prompt = self.create_custom_prompt()

    def create_custom_prompt(self):
        # 自定义提示词模板
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}

        Helpful Answer:"""

        custom_rag_prompt = PromptTemplate.from_template(template)
        return custom_rag_prompt

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_rag_chain(self, prompt):
        return (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def run_hub_prompt(self, input):
        rag_chain = self.get_rag_chain(self.prompt)
        for chunk in rag_chain.stream(input):
            print(chunk, end="", flush=True)
        print('\n')

    def run_custom_prompt(self, input):
        rag_chain = self.get_rag_chain(self.custom_prompt)
        for chunk in rag_chain.stream(input):
            print(chunk, end="", flush=True)
        print('\n')


if __name__ == '__main__':
    issue_list = [
        "主要负责人有哪些责任？",
        "主要负责人有哪些职责？",
        "主要负责人有哪些工作？"
    ]
    agent = CodeGenerateAgent()
    for issue in issue_list:
        print(f"Query: {issue}")
        print("Hub Prompt:")
        agent.run_hub_prompt(issue)
        print("Custom Prompt:")
        agent.run_custom_prompt(issue)
        print("---------------------------------------------------------")
