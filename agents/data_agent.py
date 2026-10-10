from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from tools.stock_data import get_stock_history
from state.graph_state import StockGraphState

from tools.fundamental_data import get_stock_fundamentals

# 1. Initialize the Anthropic LLM (using the fast/cheap Haiku model for data gathering)
# Ensure you are using a model version that supports tool calling.
llm = ChatAnthropic(model="claude-haiku-5-5")

# 2. Bind the tool to the LLM. 
# This passes the tool's description and schema to Anthropic.
tools = [get_stock_history, get_stock_fundamentals]
llm_with_tools = llm.bind_tools(tools)

# 3. Define the system instructions for this specific agent
system_prompt = SystemMessage(content="""
You are a Stock Data Gathering Agent. Your responsibility is to fetch data using the provided tools.
- If the user asks for technical analysis, price trends, or charts, use `get_stock_history`.
- If the user asks for fundamental analysis, news, valuation, or P/E ratios, use `get_stock_fundamentals`.
Do not analyze the data yourself. Just fetch it and report back.
""")

def data_gatherer_node(state: StockGraphState):
    """
    This function acts as a node in our LangGraph.
    It reads the current messages, asks the LLM to process them, and returns the LLM's response.
    """
    # Combine the system prompt with the conversation history from the state
    messages = [system_prompt] + state["messages"]
    
    # Invoke the LLM with the bound tools
    response = llm_with_tools.invoke(messages)
    
    # Return the LLM's response to be appended to the state's messages list
    return {"messages": [response]}