import application.langgraph.multi_agent.workflow as workflow
import application.langgraph.multi_agent.workflow_for_table as workflow_for_table
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    graph = workflow.create_graph('gpt-3.5-turbo-0125')

    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Obtain the GDP of the United States from 2000 to 2020, "
                "and then plot a line chart with Python. End the task after generating the chart。"
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
    
