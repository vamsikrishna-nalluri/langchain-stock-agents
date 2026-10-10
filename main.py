from typing import Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from state.graph_state import StockGraphState
from agents.data_agent import data_gatherer_node
from agents.analyst_agent import technical_analyst_node
from agents.fundamental_agent import fundamental_analyst_node
from agents.comprehensive_agent import comprehensive_analyst_node
from tools.stock_data import get_stock_history
from tools.fundamental_data import get_stock_fundamentals

# 1. Initialize Tools
tools = [get_stock_history, get_stock_fundamentals]
tool_node = ToolNode(tools)

# 2. Update the Supervisor to include a "comprehensive" route
class RouteDecision(BaseModel):
    next_node: Literal["data_gatherer", "fundamental_analyst", "comprehensive"] = Field(
        description="Route to 'data_gatherer' for technicals, 'fundamental_analyst' for news/valuation, or 'comprehensive' to run both in parallel."
    )

def supervisor_node(state: StockGraphState):
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    
    # FIX: Add method="json_schema" to support the newest models
    structured_llm = llm.with_structured_output(RouteDecision, method="json_schema")
    
    system_prompt = SystemMessage(content="""
    You are a Stock Squad Supervisor. Route the user's request:
    - Technicals/Charts only -> 'data_gatherer'
    - News/Valuation only -> 'fundamental_analyst'
    - BOTH (or general "Analyze this stock completely") -> 'comprehensive'
    """)
    
    messages = [system_prompt, state["messages"][0]]
    decision = structured_llm.invoke(messages)
    
    # Fallback just in case the LLM still fails to parse
    if not decision:
        return {"next_node": "data_gatherer"}
        
    return {"next_node": decision.next_node}

# 3. Custom Routers
def route_supervisor(state: StockGraphState) -> list[str]:
    """
    Returns a list of nodes to execute. 
    Returning multiple nodes triggers parallel Fan-Out execution.
    """
    decision = state.get("next_node", "data_gatherer")
    
    if decision == "comprehensive":
        # FAN-OUT: Run both gatherers in parallel
        return ["data_gatherer", "fundamental_analyst"]
    
    # Otherwise, just run the single requested node
    return [decision]

def route_gatherer_output(state: StockGraphState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # If we are in a comprehensive run, skip the standalone technical analyst
    if state.get("next_node") == "comprehensive":
        return "wait_for_comprehensive"
    
    return "technical_analyst"

def route_fundamental_output(state: StockGraphState) -> str:
    # If we are in a comprehensive run, skip the standalone fundamental analyst
    if state.get("next_node") == "comprehensive":
        return "wait_for_comprehensive"
    
    return "END"


# 4. Build the Graph
workflow = StateGraph(StockGraphState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("data_gatherer", data_gatherer_node)
workflow.add_node("tools", tool_node)
workflow.add_node("technical_analyst", technical_analyst_node)
workflow.add_node("fundamental_analyst", fundamental_analyst_node)
workflow.add_node("comprehensive_analyst", comprehensive_analyst_node)

# FIX: Return an empty dict so LangGraph knows there are no state updates to merge
def wait_node(state): return {}
workflow.add_node("wait_for_comprehensive", wait_node)

# --- Edges ---
workflow.add_edge(START, "supervisor")

# Conditional Fan-Out
workflow.add_conditional_edges("supervisor", route_supervisor)

# Technical Branch
workflow.add_conditional_edges("data_gatherer", route_gatherer_output)
workflow.add_edge("tools", "data_gatherer")
workflow.add_edge("technical_analyst", END)

# Fundamental Branch
workflow.add_conditional_edges("fundamental_analyst", route_fundamental_output)

# Fan-In to Comprehensive
# LangGraph natively waits for all incoming edges to complete before firing the target node.
workflow.add_edge(["wait_for_comprehensive", "wait_for_comprehensive"], "comprehensive_analyst") 
# (We define it twice in the list to represent the two paths converging)
workflow.add_edge("comprehensive_analyst", END)

app = workflow.compile()

# --- Test Execution ---
if __name__ == "__main__":
    print("Initializing Parallel Stock Analysis Graph...\n")
    
    # Trigger the comprehensive parallel run
    initial_state = {
        "messages": [HumanMessage(content="Give me a complete, comprehensive analysis (technical and fundamental) for LT.NS.")],
        "ticker": "LT.NS",
        "final_report": ""
    }
    
    final_report_text = "No report generated."
    
    for event in app.stream(initial_state):
        for node_name, node_state in event.items():
            print(f"--- Node Executed: {node_name} ---")
            
            # FIX: Ensure node_state is a dictionary before we try to read from it
            if isinstance(node_state, dict):
                if "next_node" in node_state:
                    print(f"Supervisor Decision: Routing to -> {node_state['next_node']}\n")
                    
                if "final_report" in node_state:
                    final_report_text = node_state["final_report"]
    
    print("\n================ FINAL MASTER REPORT ================\n")
    print(final_report_text)