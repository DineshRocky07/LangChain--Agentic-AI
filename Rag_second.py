import os
from dotenv import __main__, load_dotenv
from lanchain_core.prompt import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()


print("injusting ")
embeddings = GoogleGenerativeAIEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

vectorStore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

retriever = vectorStore.as_retriever(search_kwargs={"K":3})

prompt_template = ChatPromptTemplate.from_template(
        """
        answer the question based only on the following context:

        {context}

        question: {question}
        provied the detials answer                     """
    )


def formate_doc(docs):
    """ format the retrived document into a string"""
    return "\n".join([doc.page_content for doc in docs])

