from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def pick_llm(level: str):
    """
    Picks the appropriate Gemini LLM based on question complexity.

    Args:
        level (str): low, intermediate, or high

    Returns:
        Gemini LLM instance
    """

    if level.lower() == "low":
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
        )

    elif level.lower() == "intermediate":
        llm = ChatGoogleGenerativeAI(
            # model="gemini-3.6-flash"
            # model="gemini-3.8-flash"
            model="gemini-3.5-flash-lite",
        )

    elif level.lower() == "high":
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash"
        )

    else:
        raise ValueError(f"Unsupported level: {level}")

    return llm

if __name__ == "__main__":
    llm_obj = pick_llm("low")
    print(llm_obj)
    print(llm_obj.invoke("What is the capital of France?"))
