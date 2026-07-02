from dotenv import load_dotenv

load_dotenv()

# from langchain.chat_models import init_chat_model # easy to use for multi api connection
# from langchain.tools import tool           #empoyee
# from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage #system message menas rules

from langsmith import traceable  # tracing the project flow 
import ollama
import re   # 3 layer
import inspect  # layer

Max_Iterations = 10
MODEL= "qwen3:8b"

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
    price = float(price)
    discount_percentage= {"bronze":5,"silver":10,"gold":20}
    discount = discount_percentage.get(discount_tier,0)
    return round(price*(1-discount/100),2)

# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect.

def get_tool_descriptions(tools_dict): 
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function =getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""



# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent —
# all agency comes from the prompt above and our regex parsing below.

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):  #we will see 
    return ollama.chat(model=model, messages=messages, options=options)

#---------Agent Loop ----------
@traceable(name="ollama Agent Loop")
def run_agent(question:str):
    print(f"Question: {question}")
    print("=" * 60)

     # CHANGE 5: One prompt string replaces the system/user message split

    prompt= react_prompt.format(question=question)
    scratchpad = "" 
    
    for iteration in range(1, Max_Iterations +1):
        print(f"\n-- Iteraction {iteration}  ---")

        full_prompt = prompt + scratchpad   

        #ai_message =llm_with_tools.invoke(message)
        #tool_calls = ai_message.tool_calls
        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response=ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        
        output = response.message.content
        print(f"LLM Output:\n{output}")

        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        
         # CHANGE 6: Parse tool calls from raw text with regex — fragile if LLM doesn't follow format.
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        tool_name = action_match.group(1).strip()
        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")


        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))


        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")