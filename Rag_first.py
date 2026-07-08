import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader  #load data into text
from langchain_text_splitters import CharacterTextSplitter  #split large text into chunks 
from langchain_google_genai import GoogleGenerativeAIEmbeddings # this method is used to convert text into vector representation
from langchain_pinecone import PineconeVectorStore # this method is used to store vector representation into pinecone

load_dotenv()

if __name__ == "__main__":
    print("injusting ")
    loader = TextLoader(r"C:\Users\tdine\Downloads\mediumblog1.txt", encoding="utf-8")
    documents = loader.load()
    
    print("splitting text into chunks")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text=text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(text)}")

   # embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        task_type="retrieval_document"
    )
    print("injuction")
    PineconeVectorStore.from_documents(text,embeddings, index_name=os.getenv("INDEX_NAME"))