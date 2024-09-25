import functools
from langgraph.prebuilt import ToolNode
from application.langgraph.multi_agent.agent import Agent
import application.langgraph.multi_agent.node as node
from application.langgraph.multi_agent.tools import tavily_tool, table_generator
from application.langgraph.multi_agent.agent_state import AgentState
import application.langgraph.multi_agent.router as router
from application.langgraph.multi_agent.llm import LLM
from langgraph.graph import END, StateGraph, START

def create_graph(model="gpt-4o-mini"):
    # 为 Agent 配置各自的大模型
    research_llm = LLM.create_openai_llm(model=model, temperature=0.5)
    table_llm = LLM.create_openai_llm(model=model, temperature=0)

    # 研究智能体及其节点
    research_agent = Agent(
        research_llm,  # 使用 research_llm 作为研究智能体的语言模型
        [tavily_tool],  # 研究智能体使用 Tavily 搜索工具
        tool_message=(
        "Before using the search engine, carefully think through and clarify the query."
        " Then, conduct a single search that addresses all aspects of the query in one go",
        ),
        custom_notice=(
            "Notice:\n"
            "Only gather and organize information. Do not generate code or give final conclusions, leave that for other assistants."
        ),
    ).create_agent()

    # 使用 functools.partial 创建研究智能体的节点，指定该节点的名称为 "Researcher"
    research_node = functools.partial(node.agent_node, agent=research_agent, name="Researcher")

    table_agent = Agent(
        table_llm,  # 使用 table_llm 作为图表生成器智能体的语言模型
        [table_generator],
        tool_message="Generate and display data tables based on the provided input, and save them to the specified location.",  # 系统消息，指导智能体如何生成图表
        custom_notice="Notice:\n"
        "Once you have processed the data and generated the table, save the output to the file specified by the user. If everything is completed, respond with FINAL ANSWER.",
    ).create_agent()

    # 使用 functools.partial 创建图表生成器智能体的节点，指定该节点的名称为 "Chart_Generator"
    table_node = functools.partial(node.agent_node, agent=table_agent, name="table_Generator")

    # 定义工具列表，包括 Tavily 搜索工具和 Python REPL 工具
    tools = [tavily_tool, table_generator]

    # 创建工具节点，负责工具的调用
    tool_node = ToolNode(tools)

    # 创建一个状态图 workflow，使用 AgentState 来管理状态
    workflow = StateGraph(AgentState)

    # 将研究智能体节点、图表生成器智能体节点和工具节点添加到状态图中
    workflow.add_node("Researcher", research_node)
    workflow.add_node("table_Generator", table_node)
    workflow.add_node("call_tool", tool_node)

    # 为 "Researcher" 智能体节点添加条件边，根据 router 函数的返回值进行分支
    workflow.add_conditional_edges(
        "Researcher",
        router.router,  # 路由器函数决定下一步
        {
            "continue": "table_Generator",  # 如果 router 返回 "continue"，则传递到 table_Generator
            "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
            "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
        },
    )

    # 为 "table_Generator" 智能体节点添加条件边
    workflow.add_conditional_edges(
        "table_Generator",
        router.router,  # 同样使用 router 函数决定下一步
        {
            "continue": "Researcher",  # 如果 router 返回 "continue"，则回到 Researcher
            "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
            "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
        },
    )

    # 为 "call_tool" 工具节点添加条件边，基于“sender”字段决定下一个节点
    # 工具调用节点不更新 sender 字段，这意味着边将返回给调用工具的智能体
    workflow.add_conditional_edges(
        "call_tool",
        lambda x: x["sender"],  # 根据 sender 字段判断调用工具的是哪个智能体
        {
            "Researcher": "Researcher",  # 如果 sender 是 Researcher，则返回给 Researcher
            "table_Generator": "table_Generator",  # 如果 sender 是 table_Generator，则返回给 table_Generator
        },
    )

    # 添加开始节点，将流程从 START 节点连接到 Researcher 节点
    workflow.add_edge(START, "Researcher")

    # 编译状态图以便后续使用
    graph = workflow.compile()

    return graph
