from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from state.graph_state import StockGraphState

llm = ChatAnthropic(model="claude-sonnet-4-6")

system_prompt = SystemMessage(content="""
You are an elite Lead Portfolio Manager. You are about to receive BOTH Technical Data 
(price action, RSI, SMA) and Fundamental Data (valuation, news, business context) 
gathered simultaneously by your data agents.

Your job is to synthesize BOTH into a single, cohesive master report.
1. Summarize the fundamental health and news catalysts.
2. Summarize the technical trend and momentum.
3. Provide a final, combined verdict (e.g., "Fundamentally strong, but technically overbought - wait for a pullback").

Do not use tools. Analyze only the data provided in the conversation history.
""")

def comprehensive_analyst_node(state: StockGraphState):
    """
    Reads the combined messages from the parallel data fetchers and generates a master report.
    """
    messages = (
        [system_prompt] 
        + state["messages"] 
        + [HumanMessage(content="Please review all the gathered technical and fundamental data above and provide your comprehensive master report.")]
    )
    
    response = llm.invoke(messages)
    
    return {
        "messages": [response], 
        "final_report": response.content
    }