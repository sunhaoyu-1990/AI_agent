import application.langgraph.multi_agent.workflow as workflow
import application.langgraph.multi_agent.workflow_for_table as workflow_for_table
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    graph = workflow_for_table.create_graph('gpt-4o')

    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Obtain the GDP of the United States from 2000 to 2020, "
                "and then Convert the data to a table format and save the table to the project root directory, naming it `data.csv`. End the task after save the table。"
                )
            ],
        },
        # 设置最大递归限制
        {"recursion_limit": 20},
        stream_mode="values"
    )

    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()  # 打印消息内容
    
