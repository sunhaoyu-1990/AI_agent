import application.langgraph.chatbot as chatbot
import application.langgraph.chatbot_tools as chatbot_tools

if __name__ == "__main__":
    print("Running chatbot...")
    chatbot.run("西安明天天气怎么样？")
    print("Running chatbot_tools...")
    chatbot_tools.run("西安明天天气怎么样？")