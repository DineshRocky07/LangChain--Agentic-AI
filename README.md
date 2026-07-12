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

- 29 30 31

  how to easy chage model use that import init_chat_model
    Model= "qwen3.5:0.8b"
    Model_gen="gemini-1.5-flash"
    # #llm = init_chat_model(f"ollama:{Model}",temperature=0)
    llm = init_chat_model(f"google_genai:{Model_gen}",temperature=0)
    llm_with_tools = llm.bind_tools(tools)
   
   # main concern is switch model ok but this not enoff

# Section [6]
  # -- [Layer 2] Raw function calling

  - 32 33 34
  # this vidoe we learn Tool raw how they work without langchain
  # this full you will see - 2 agent loop.py 
  # with out lagchain how difficult to apply tools need more study on this
  🧑 USER: "What is the price of a laptop?"
   │
   ▼
 📥 STEP 1: THE MEMORY BANK (messages list)
    We put the user's question into the 'messages' list.
    [ {"role": "user", "content": "What is the price of a laptop?"} ]
   │
   ▼
 🧠 STEP 2: THE AI BRAIN (ollama_chat_trace)
    We hand the memory bank to the AI. The AI thinks, but it CANNOT 
    look up the price itself. So, it asks you to do it.
    It generates a JSON package:
      {
         "name": "get_product_price",
         "arguments": {"product": "laptop"}
      }
   │
   ▼
 ⚙️ STEP 3: THE PYTHON INTERCEPTOR (Your Code)
    Your code catches the AI's JSON package and breaks it apart:
    • tool_name = "get_product_price"
    • tool_args = {"product": "laptop"}
   │
   ▼
 📖 STEP 4: THE DICTIONARY LOOKUP (tools_dict)
    Python needs to translate the text string into real code.
    It looks up "get_product_price" in your tools_dict and finds 
    the actual function sitting in memory.
    • tool_to_use = <function get_product_price>
   │
   ▼
 ⚡ STEP 5: THE EXECUTION (**)
    Python unpacks the bag of arguments directly into the function.
    CODE:  tool_to_use(**tool_args)
    MEANS: get_product_price(product="laptop")
   │
   ▼
 🎯 STEP 6: THE OBSERVATION (The Result)
    The function runs, checks the database, and spits out the answer.
    • observation = 50000
   │
   ▼
 📦 STEP 7: UPDATING THE MEMORY (messages.append)
    We write the result down on a piece of paper and shove it back 
    into the memory bank so the AI can read it.
    [ {"role": "tool", "name": "get_product_price", "content": "50000"} ]
   │
   └───► LOOP RESTARTS ↻ 
         We send the updated memory bank back to the AI (Step 2).
         The AI reads it, sees the price is 50000, and finally has 
         enough info to talk to the user!
         

# section 7 [Layer 3] the foundaction of function calling

- 35 we are bulding function alling
 
  ReAct prompt is most import on in fuction calling
  this use 
  # this wasthe ReAct first prompt 
  Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}

# 36 Generating dynamic Tool Description in python 

    complted but old how llm and ReAct work very tuff

# 40 41 function calling 

   tool calling and Recation is not relaiable 

   # but function calling more relaible that maily focus

   - benifit 
      stucture and relainbale
      effective token cost saving 

      development easy
      structure flow


# 42 video RAG
   - Rag is used for our document -> chunking -> save -> get use llm

# 43 Indroduction RAG implementaction
   - this all topic we will see
     
    #   Embeddings
        use vercote method easy to get checking data
    #  Vector stores (Pinecone)

    #  RetrievalQA Chain

    # LangChain document loaders  
      - load documet use langcain

    #  LangChain text splitters

# 45 start in rag_first.py

from dotenv import load_dotenv
from langchain_community.document_loaders import text_loader  #load data into text
from langchain_text_splitter import character_text_splitter  #split large text into chunks 
from langchain_google_genai import GoogleGenerativeAIEmbeddings # this method is used to convert text into vector representation
from langchain_pinecone import Pineconevectorstore # this method is used to store vector DB into pinecone

# 46 and 47 Data indexing complte you can see in Rag_first.py 
   document loader -> data chunking -> embedding-> vercore DB like pinecone
# 48 we will do data retricial and generation 
  
    user query ->vector embedding -> vector Db -> top-K chucks 
                                                               - llm -> response 
     
   # file name Rag_second without LCEL.py  LangChain Expression Language

# 49 Rag use LCEL
   from langchain_core.output_parsers import StrOutputParser
   from langchain_core.runnables import RunnablePassthrough  # input and op same
   from operator import itemgetter  #

   create new function langchain as chain
 # create_retrieval_chain_with_lcel() need toe lean this fully 
   

