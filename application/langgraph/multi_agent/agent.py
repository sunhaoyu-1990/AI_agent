from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class Agent:
    def __init__(self, llm, tools, tool_message: str, custom_notice: str=""):
        self.llm = llm
        self.tools = tools
        self.tool_message = tool_message
        self.custom_notice = custom_notice
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    "\n{custom_notice}\n"
                    " You have access to the following tools: {tool_names}.\n{tool_message}\n\n",
                ),
                MessagesPlaceholder(variable_name="messages"),  # 用于替换的消息占位符
            ]
        )

    def create_agent(self):
        # 将系统消息部分和工具名称插入到提示模板中
        prompt = self.prompt.partial(tool_message=self.tool_message, custom_notice=self.custom_notice)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in self.tools]))
        return prompt | self.llm.bind_tools(self.tools)
