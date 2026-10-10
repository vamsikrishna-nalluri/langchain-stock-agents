from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from state.graph_state import StockGraphState

# Using the reasoning model (no temperature argument)
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

system_prompt = SystemMessage(content="""
You are an expert Stock Technical Analyst. Your single responsibility is to read 
the historical price and volume data provided in the conversation history and 
write a concise, professional technical analysis report.

Focus on:
1. Price action and recent trends.
2. Volume anomalies or patterns.
3. A short-term momentum conclusion.

Do not use tools. Do not ask for more data. Analyze only the data provided to you.
""")

def technical_analyst_node(state: StockGraphState):
    """
    This node reads the raw data gathered by the previous agent and 
    generates a final analytical report.
    """
    # Fix: We append a final HumanMessage to ensure Anthropic's API is happy,
    # and to explicitly command the Analyst to act on the preceding context.
    messages = (
        [system_prompt] 
        + state["messages"] 
        + [HumanMessage(content="Please review the data gathered above and provide your technical analysis report.")]
    )
    
    response = llm.invoke(messages)
    
    return {
        "messages": [response], 
        "final_report": response.content
    }