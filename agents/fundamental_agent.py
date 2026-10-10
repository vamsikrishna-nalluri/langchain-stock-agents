from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from state.graph_state import StockGraphState

llm = ChatAnthropic(model="claude-sonnet-5-5")

system_prompt = SystemMessage(content="""
You are an expert Fundamental Stock Analyst. Your single responsibility is to read 
the fundamental financial data, metrics, and news provided in the conversation history 
and write a concise, professional fundamental analysis report.

Focus on:
1. Valuation (is the P/E ratio high/low?).
2. Business context (sector and industry).
3. Recent news sentiment and potential catalysts.

Do not use tools. Do not ask for more data. Analyze only the data provided to you.
""")

def fundamental_analyst_node(state: StockGraphState):
    """
    Reads the fundamental data gathered by the previous agent and generates a report.
    """
    messages = (
        [system_prompt] 
        + state["messages"] 
        + [HumanMessage(content="Please review the fundamental data and news gathered above and provide your fundamental analysis report.")]
    )
    
    response = llm.invoke(messages)
    
    return {
        "messages": [response], 
        "final_report": response.content
    }