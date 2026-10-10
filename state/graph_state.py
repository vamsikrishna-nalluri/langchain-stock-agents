from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class StockGraphState(TypedDict):
    """
    Represents the shared state of our multi-agent Stock Analysis system.
    Every node in the graph will receive this state, modify it, and return the updates.
    """
    
    # The `add_messages` reducer is crucial. It tells LangGraph to APPEND new messages 
    # to the existing list, rather than overwriting the entire list.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Domain-specific state variables
    ticker: str
    final_report: str