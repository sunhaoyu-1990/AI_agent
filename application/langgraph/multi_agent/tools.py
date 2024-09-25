from typing import Annotated
import os
import pandas as pd
import getpass
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass(f"请输入您的 TAVILY API 密钥：")
# 定义 Tavily 搜索工具，用于搜索最多 5 条结果
tavily_tool = TavilySearchResults(max_results=3)
       
# Python REPL 工具，用于执行 Python 代码
repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

    return f"Successfully executed:\n```python\n{code}\n```\n"

@tool
def table_generator(
    data: Annotated[dict, "The input data to convert to table, should be a dictionary."]
):
    """Converts input data into a table format, prints it out, and saves it to a local file."""
    try:
        # 将数据转换为DataFrame
        df = pd.DataFrame(data)
        
        # 打印表格到控制台
        print("Generated Table:")
        print(df)
        
        # 保存表格到本地文件
        df.to_csv('result.csv', index=False)
        return f"Table successfully saved to result.csv"
    
    except Exception as e:
        return f"Failed to create or save the table. Error: {repr(e)}"
