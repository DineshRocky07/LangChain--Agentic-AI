from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import os

load_dotenv()

def main():
    print("Hello from langchain-agentic-aiollama!")
    information = """ who is dinesh Thandapani and he working in valeo chennai india past 1 year check his inforamtion throw online and give me a short summary about him and his work in valeo chennai india"""
    summary_template =""" given then informaction {information} about the person i want you to create :
       1. A short summary    2. two interesting facts about then """
    
    summary_prompt_template =PromptTemplate(
        input_variables=["information"],template=summary_template
    )

    llm= ChatOllama(model="gemma3:270m", temperature=0)

    chain = summary_prompt_template | llm

    response = chain.invoke(input ={"information":information})

    print(response.content)

if __name__ == "__main__":
    main()