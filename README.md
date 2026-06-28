# LangChain--Agentic-AI-Engineering-with-LangChain-LangGraph
Langchain_RAG_Beginner Instructor: Eden Marco

step 1: uv init 

step 2:
        uv add langchain langchain-google-genai langchain-tavily tavily-python python-dotenv black isort

langchain          	Main LangChain framework for chains, prompts, agents, tools
langchain-google-genai	Connect LangChain to Gemini models
langchain-tavily	LangChain wrapper for Tavily search
tavily-python	Direct Tavily API client
python-dotenv	Load API keys from .env file
black	Automatically format Python code
isort	Automatically sort imports

step 3:
      env we alrady have 
           1.genai key 
           2.langsmith key
           3.tavily 
           
step 4:
 
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()            -> Load API keys
ChatGoogleGenerativeAI   -> Gemini model
@tool                    -> Create functions agent can use
create_agent()           -> Create AI agent
HumanMessage             -> User message object


##################################
Agent = Manager

Tool = Employee

Gemini = Brain

#####################################

VERSION 2:
24-jUNE VIDOE 21-

STEP 1 : 
- from tavily import TavilyClinet

step 2:
  tavily=TavilyClinet() 
    
 - this auto search env and get apikey
    
step 3:
   tool function 
     -Static return to real return search capailty
-   return tavily.search(query=query)
step 4:

    finally we give full permission to access internet tavily

Version 3:
   - langcain tavily
   - inbuild tavily search and more 
step 1:
    we use own tavily lanchain
-  from langchain_tavily import tavilySearch

step 2:
    just 
- tools = [tavilySearch ] and RUn the code

    
- version 4
- structure output 

step 1
  from typing import List
  from pydantic import BaseModel, Field #own stuture use BaseModel

step 2: 
  - added new class

class Source(BaseModel):
    """Schema for source used by agent"""
    url:str =Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """schema for agent response with answer and Source"""
    
    answer:str = Field(description="The agents answer to the query")
    source:List[Source]=Field(default_factory=list,description="List of sources uses to geneate the answer")

step 3:
   agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)
    
   # some debug more we try

# Viode numeber 23 

 use stuctutre op 

 1.two type 
         1.toolstrategey 
         2. providerstategey [defautl] langchain past project we done
  

# Agent Understand HOOD layer1 video 24 to 27
- 24 
   explain core all 
   what layer and what we do

- 25
   waht we build : Ecommerce Agent
   Agent about query from agent and get the discount 

- 26
   here we explain ReACT loop
   # Top algorth foundaction in AI
   
    diagram:
    see the diagam
    user query -> THought -> Action -> tool -> Observation -> THought -> answer
  
- 27 stepup

    this project we need ollama

    # uv add langchain langchain-ollama lagchain-google-genai python-dotenv black isort
    # download qwen3:1.7b for toll calling support we use qwen3.5:0.8b

- 28 Write tools
  # Layer [1] for ReAct Loop
  
  Creat a new file name: 1_agent_loop_langchain_tool_calling.py
  # need more attenction for this code in future 

- 29

  how to easy chage model use that import init_chat_model
    Model= "qwen3.5:0.8b"
    Model_gen="gemini-1.5-flash"
    # #llm = init_chat_model(f"ollama:{Model}",temperature=0)
    llm = init_chat_model(f"google_genai:{Model_gen}",temperature=0)
    llm_with_tools = llm.bind_tools(tools)
   
   # main concern is switch model ok but this not enoff

