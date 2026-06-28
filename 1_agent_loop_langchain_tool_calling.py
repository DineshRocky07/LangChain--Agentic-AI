from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model # easy to use for multi api connection
from langchain.tools import tool           #empoyee
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage #system message menas rules
from langsmith import traceable  # tracing the project flow 


Max_Iterations = 10
Model= "qwen3.5:0.8b"
Model_gen="gemini-2.5-flash"

@tool
def get_product_price(product: str) -> float:  # return value is float
    """
    Look up the price of a product in the catalog."""
    print(f"> Executing get_product_price(product='{product}')")
    prices= {"laptop":50000,"Headphone":1999.95,"Keyboard": 89.50}
    return prices.get(product,0) # that zero default value

@tool
def apply_discount(price:float,discount_tier: str)-> float:
    """ Apply discount_tier to a price and return a final price
    Available tiers: bronze,silver,gold."""
    print(f"> Executing apply_discount(price={price},discount_tier='{discount_tier}')")
    discount_percentage= {"bronze":5,"silver":10,"gold":20}
    discount = discount_percentage.get(discount_tier,0)
    return round(price*(1-discount/100),2)

#---------Agent Loop ----------
@traceable(name="Lagchain Agent Loop")
def run_agent(question:str):
    tools=[get_product_price,apply_discount]

    #print(f"Tools:{tools}")
    

    tools_dict = {t.name: t for t in tools}
    
    #llm = init_chat_model(f"ollama:{Model}",temperature=0)
    llm = init_chat_model(f"google_genai:{Model_gen}",temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"Question: {question}")
    print("=" * 60)

    message=[
        SystemMessage(content=("You are a helpful shopping assistant."
                               "you are access to a product catalog tool"
                               "and a discount tool. \n\n"
                               "Strict Rules- you must follow thers exactly:/n"
                                "1.Never guess or assume any product price."
                                "you must call get_product_price first to get the real price \n"
                                "2. onlt call apply_discount AFTER you have received"
                                "a price form get_product_price.Pass exact price"
                                "returned by get_product_price - do not pass a made up number\n"
                                "3. Never calculate discount yourself you math."
                                "Always use the apply_discount tools.\n"
                                "4. if user not does not dpecify a discount tier"
                                "ask them which tier to sue - do not assume one."
                                )
                ),
                HumanMessage(content=question)

    ]

    for iteration in range(1, Max_Iterations +1):
        print(f"\n-- Iteraction {iteration}  ---")

        ai_message =llm_with_tools.invoke(message)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\n Final Anser:{ai_message.content}\n")
            return ai_message.content
        
        #process only first tool call 
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")   
        print(f"Tool Call: {tool_call} \n Tool name: {tool_name}\n tool_args: {tool_args} \n tool_call_id: {tool_call_id}")


        tool_to_use = tools_dict.get(tool_name)
        # print(f"Tool name {tool_name}")
        # print(f"Tool to use {tool_to_use}")
        if tool_to_use is None:
            raise ValueError(f"Tools {tool_name} not found")
        observation =tool_to_use.invoke(tool_args)
       # print(f"Tool args {tool_args}")
        print(f"[Tool Result] {observation}")

        message.append(ai_message)
        message.append(ToolMessage(content=str(observation),tool_call_id=tool_call_id))

    print("Error : Max itraction reached without a final answer")
    return None



if __name__ == "__main__":
    print("Hello from langchain-agentic-ai!")
    result = run_agent("what is the price of a laptop after applying a gold discounrt")