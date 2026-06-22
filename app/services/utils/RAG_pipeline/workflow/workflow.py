from langgraph.graph import StateGraph, START, END
from .state import ChatState
from .nodes import Nodes

def create_workflow():
    """FIXED: Create workflow with proper parallel execution"""
    node_instance = Nodes()

    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("trim_chat_history", node_instance.trim_chat_history)
    graph.add_node("decide_retrieval_path", node_instance.decide_retrieval_path)
    graph.add_node("file_process", node_instance.file_process)
    graph.add_node("global_retrieval_process", node_instance.global_retrieval_process)
    graph.add_node("llm_call", node_instance.llm_call)

    # Parallel start
    graph.add_edge(START, "trim_chat_history")
    graph.add_edge(START, "decide_retrieval_path")

    # Conditional routing
    graph.add_conditional_edges(
        "decide_retrieval_path",
        lambda state: state["next_node"],
        {
            "file_process": "file_process",
            "global_retrieval_process": "global_retrieval_process",
        }
    )

    graph.add_edge("trim_chat_history", "llm_call")
    graph.add_edge("file_process", "llm_call")
    graph.add_edge("global_retrieval_process", "llm_call")
    graph.add_edge("llm_call", END)

    return graph.compile(), node_instance


async def run_workflow(input_data: dict, config: dict = None):
    workflow, _ = create_workflow()
    result = await workflow.ainvoke(input_data, config=config)
    return result