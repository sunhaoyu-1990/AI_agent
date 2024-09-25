from langchain_openai import ChatOpenAI

class LLM:
    @staticmethod
    def create_openai_llm(model, temperature):
        return ChatOpenAI(model=model, temperature=temperature)