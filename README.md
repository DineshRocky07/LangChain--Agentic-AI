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