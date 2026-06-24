from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent  #manager
from langchain.tools import tool           #empoyee
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI #brain
from tavily import TavilyClient #version 2
from langchain_tavily import TavilySearch  #version 3

tavily=TavilyClient() #auto read env and get api

#static tool 
#@tool
def search(query: str) -> str:
    """
    Tool search form over internet
    Args:
          query: the search for 
    Return :
           the search result
    """
    print(f"Search for {query}")
    #return "write a poem for her"  #V1 static
    return tavily.search(query=query)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
#tools =[search] # that function name 
tools = [TavilySearch()] #version 3
agent = create_agent(model=llm, tools=tools,) #agent we give llm and function info
def main():
    print("Hello from langchain-agentic-ai!")
    result =agent.invoke({"messages":[HumanMessage(content="AI Engineer jobs on LinkedIn  ")]})
    print(result)

if __name__ == "__main__":
    main()
