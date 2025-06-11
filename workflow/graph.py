"""
LangGraph workflow definition for the RAG-based storyboard system.
"""
from langgraph.graph import StateGraph, END
from typing import Dict, Any

from workflow.state import WorkflowState
from workflow.nodes import (
    planner_node, 
    renderer_node, 
    vision_qa_node, 
    memory_update_node,
    initialize_memory_service
)


def create_workflow(config: Dict[str, Any]):
    """Create the LangGraph workflow for storyboard generation."""
    
    # Initialize memory service
    initialize_memory_service(
        db_path=config.get("db_path", "./lancedb_data"),
        data_path=config.get("data_path", "data")
    )
    
    # Create the workflow
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("renderer", renderer_node)
    workflow.add_node("vision_qa", vision_qa_node)
    workflow.add_node("memory_update", memory_update_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Define edges
    workflow.add_edge("planner", "renderer")
    workflow.add_edge("renderer", "vision_qa")
    
    # Conditional routing from planner
    def route_from_planner(state: WorkflowState) -> str:
        if state["status"] == "no_more_scenes":
            return "end"
        else:
            return "renderer"
    
    workflow.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "renderer": "renderer",
            "end": END
        }
    )
    
    # Conditional routing from renderer
    def route_from_renderer(state: WorkflowState) -> str:
        if state["status"] == "error":
            # Skip to memory update on error
            return "memory_update"
        else:
            return "vision_qa"
    
    workflow.add_conditional_edges(
        "renderer",
        route_from_renderer,
        {
            "vision_qa": "vision_qa",
            "memory_update": "memory_update"
        }
    )
    
    # Conditional routing from vision_qa
    def route_from_qa(state: WorkflowState) -> str:
        if state["status"] == "complete":
            return "memory_update"
        else:  # rendering
            return "renderer"
    
    workflow.add_conditional_edges(
        "vision_qa",
        route_from_qa,
        {
            "renderer": "renderer",
            "memory_update": "memory_update"
        }
    )
    
    # Memory update always goes to END
    workflow.add_edge("memory_update", END)
    
    return workflow.compile() 