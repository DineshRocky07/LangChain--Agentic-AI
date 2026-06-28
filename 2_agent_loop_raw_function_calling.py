from dotenv import load_dotenv

load_dotenv()

# from langchain.chat_models import init_chat_model # easy to use for multi api connection
# from langchain.tools import tool           #empoyee
# from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage #system message menas rules

from langsmith import traceable  # tracing the project flow 
import ollama


Max_Iterations = 10
Model= "qwen3.5:0.8b"

#@tool removed
@traceable(name="tool")
def get_product_price(product: str) -> float:  # return value is float
    """
    Look up the price of a product in the catalog."""
    print(f"> Executing get_product_price(product='{product}')")
    prices= {"laptop":50000,"Headphone":1999.95,"Keyboard": 89.50}
    return prices.get(product,0) # that zero default value

#@tool
@traceable(name="tool")
def apply_discount(price:float,discount_tier: str)-> float:
    """ Apply discount_tier to a price and return a final price
    Available tiers: bronze,silver,gold."""
    print(f"> Executing apply_discount(price={price},discount_tier='{discount_tier}')")
    discount_percentage= {"bronze":5,"silver":10,"gold":20}
    discount = discount_percentage.get(discount_tier,0)
    return round(price*(1-discount/100),2)

# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.
@traceable(name="ollama Chat",run_type="llm" )
def ollama_chat_trace(messages):
    return ollama.chat(model=Model,tools=tools_for_llm,messages=messages)

#---------Agent Loop ----------
@traceable(name="ollama Agent Loop")
def run_agent(question:str):
    
    #tools=[get_product_price,apply_discount]
    #print(f"Tools:{tools}")
    #tools_dict = {t.name: t for t in tools}
    
    #llm = init_chat_model(f"ollama:{Model}",temperature=0)
    #llm_with_tools = llm.bind_tools(tools)
    #manulay create dict
    tools_dict ={
        "get_product_price":get_product_price,
        "apply_discount":apply_discount
    }
    
    print(f"Question: {question}")
    print("=" * 60)

    messages=[{
              "role":"system",
              "content":("You are a helpful shopping assistant."
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
             },
                {"role":"user","content":question}

    ]

    for iteration in range(1, Max_Iterations +1):
        print(f"\n-- Iteraction {iteration}  ---")

        #ai_message =llm_with_tools.invoke(message)
        #tool_calls = ai_message.tool_calls

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response=ollama_chat_trace(messages=messages )
        ai_message= response.message
        tool_calls= ai_message.tool_calls

        

        print(ai_message)
        if not tool_calls:
            print(f"\n Final Anser:{ai_message.content}\n")
            return ai_message.content
        
        #process only first tool call 
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        # tool_name = tool_call.get ("name")
        tool_args=tool_call.function.arguments
        # tool_args = tool_call.get("args",{})

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        # print(f"Tool name {tool_name}")
        # print(f"Tool to use {tool_to_use}")
        if tool_to_use is None:
            raise ValueError(f"Tools {tool_name} not found")
        # Difference 7: Direct function call instead of tool.invoke()
        observation=tool_to_use(**tool_args)
        #observation =tool_to_use.invoke(tool_args)
       # print(f"Tool args {tool_args}")
        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
       # messages.append(ToolMessage(content=str(observation),tool_call_id=tool_call_id))
        messages.append({
            "role":"tool",
            "name":tool_name,
            "content":str(observation)})

    

    print("Error : Max itraction reached without a final answer")
    return None



if __name__ == "__main__":
    print("Hello from langchain-agentic-ai!")
    result = run_agent("what is the price of a laptop after applying a gold discounrt")