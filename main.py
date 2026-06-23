from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent  #manager
from langchain.tools import tool           #empoyee
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI #brain

#static tool 
@tool
def search(query: str) -> str:
    """
    Tool search form over internet
    Args:
          query: the search for 
    Return :
           the search result
    """
    print(f"Search for {query}")
    return "chennai weather is rainly"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
tools =[search] # that function name
agent = create_agent(model=llm, tools=tools,) #agent we give llm and function info
def main():
    print("Hello from langchain-agentic-ai!")
    result =agent.invoke({"messages":[HumanMessage(content="what is the weather in chennai")]})
    print(result)

if __name__ == "__main__":
    main()
