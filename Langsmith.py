from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI #lanchain import 

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model ="gemini-2.5-flash"
)
response =llm.invoke("Write a poem about the girl eyes") #involke generate meaning
print(response.content)  #show content only