from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent  #manager
from langchain.tools import tool           #empoyee
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI #brain
from tavily import TavilyClient #version 2
from langchain_tavily import TavilySearch  #version 3
from typing import List                                                    #version 4
from pydantic import BaseModel, Field #own stuture use BaseModel           #version 4

#version 4

class Source(BaseModel):
    """Schema for source used by agent"""
    url:str =Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """schema for agent response with answer and Source"""
    
    answer:str = Field(description="The agents answer to the query")
    source:List[Source]=Field(default_factory=list,description="List of sources uses to geneate the answer")

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
agent = create_agent(model=llm, tools=tools,response_format=AgentResponse) #agent we give llm and function info # version4
def main():
    print("Hello from langchain-agentic-ai!")
    result =agent.invoke({"messages":[HumanMessage(content="AI Engineer jobs on LinkedIn  ")]})
    print(result)

if __name__ == "__main__":
    main()
