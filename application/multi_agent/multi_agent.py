'''
@file: multi_agent.py
@breif: This file is the main file for multi-agent chatbot.
@author: Sunhaoyu
@create: 2024.09.09
@update: 2024.09.09
'''

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

class CodeGenerateAgent:
    def __init__(self):
        self.model = 'gpt-4o-mini'

    def run(self, input):
        planner = (
            ChatPromptTemplate.from_template("生成关于以下需求，生成代码的逻辑框架: {input}")
            | ChatOpenAI(model=self.model)
            | StrOutputParser()
            | {'base_response': RunnablePassthrough()}
        )
        python_code = (
            ChatPromptTemplate.from_template("你是Python代码工程师，针对提供的代码框架，通过python语言生成所有代码，代码框架如下：{base_response}")
            | ChatOpenAI(model=self.model)
            | StrOutputParser()
        )
        java_code = (
            ChatPromptTemplate.from_template("你是Java开发工程师，关于{base_response}的需求，请通过java语言生成代码")
            | ChatOpenAI(model=self.model)
            | StrOutputParser()
        )
        end = (
            ChatPromptTemplate.from_messages(
                [
                    ("ai", "实现需求的代码框架为：\n{base_response}"),
                    ("human", "python实现的代码如下:\n{results_1}\n\nJava实现的代码如下:\n{results_2}"),
                    ("system", "请将检查Python和Java代码，没有问题，请分别展示给用户。如果有问题，请修改后，讲修改后的展示给用户。"),
                ]
            )
            | ChatOpenAI(model=self.model)
            | StrOutputParser()
        )

        chain = (
            planner
            | {
                'results_1': python_code,
                'results_2': java_code,
                "base_response": itemgetter("base_response"),
            }
            | end
        )
        print(chain.invoke(input))


if __name__ == '__main__':
    agent = CodeGenerateAgent()
    agent.run("实现一个简单的计算器，支持加减乘除运算。")
