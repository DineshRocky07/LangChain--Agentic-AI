import os
from dotenv import __main__, load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


print("injusting ")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
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

def retrieval_chain_without_lcel(query: str):
    
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # step 2: Format the retrieved documents into a single string
    context = formate_doc(docs)

    # Step 3: Format the prompt with context and question

    messages = prompt_template.format_messages(context=context, question=query)

    # llm invoke the formate message 
    response =llm.invoke(messages)

    return response.content




# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = ( RunnablePassthrough.assign(context=itemgetter("question")| retriever | formate_doc
                        )
                        |prompt_template
                        |llm
                        |StrOutputParser()
    )

    return retrieval_chain 
                       

if __name__ == "__main__":

    query = "what is Pinecone in machine learning?"

    # option :0 Raw invoxtion without Rag

    # print("\n"+"=" * 70)
    # print("Raw invocation without RAG")
    # print("=" * 70)
    # raw_result = llm.invoke ([HumanMessage(content=query)])
    # print("\n ANswer: " + raw_result.content)


    # # option :1 RAG with retrival chain without LCEL LangChain Expression Language

    # print("\n"+"=" * 70)
    # print("RAG with retrival chain without LCEL")
    # print("=" * 70)
    # rag_result = retrieval_chain_without_lcel(query)
    # print("\n ANswer: " + rag_result)

     # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)