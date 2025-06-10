"""
Main entry point for the AI storyboard generator.
Orchestrates the workflow and manages parallel processing.
"""
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict
import argparse
import time
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from langgraph.graph import Graph, StateGraph, END
from agents.director import director_node
from agents.artist import artist_node
from agents.critic import critic_node
from agents.shared import build_complete_context
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()

# Define workflow state using TypedDict
class WorkflowState(TypedDict):
    """State passed between workflow nodes."""
    current_shot_id: int
    shot_id: Optional[int]
    shot_text: Optional[str]
    entities: Optional[List[str]]
    attempt: int
    current_variation: int
    status: str
    output_base_path: str
    variations_per_shot: int
    image_path: Optional[str]
    image_paths: List[str]
    critic_notes: Optional[str]
    error: Optional[str]
    context_window: int
    historical_context: List[Dict[str, Any]]  # List of previous generations
    base_context: Dict[str, Any]  # Base documents from data folder

# Configuration
RATE_LIMIT = 3  # Calls per minute
PARALLEL_WORKERS = 1  # Number of parallel workers
DEFAULT_VARIATIONS = 3  # Default number of variations per shot

# Create timestamp-based output folder
def create_output_folder() -> str:
    """Create a timestamped output folder.
    
    Returns:
        Path to the output folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"output/run_{timestamp}")
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)

# Define the workflow
def create_workflow(output_base_path: str, variations_per_shot: int, context_window: int, max_retries: int):
    """Create the LangGraph workflow for storyboard generation.
    
    Args:
        output_base_path: Base path for output files
        variations_per_shot: Number of variations to generate per shot
        context_window: Number of previous images to include in context
        max_retries: Maximum retries per image
    """
    workflow = StateGraph(WorkflowState)
    
    # Create a wrapper to pass max_retries to critic
    async def critic_node_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        # Set max_retries in the state so critic can use it
        state_with_retries = {**state, 'max_retries': max_retries}
        return await critic_node(state_with_retries)

    # Add nodes
    workflow.add_node("director", director_node)
    workflow.add_node("artist", artist_node)
    workflow.add_node("critic", critic_node_wrapper)
    
    # Define edges
    workflow.set_entry_point("director")
    
    # Director decides what to do
    def director_router(state):
        status = state.get("status", "")
        if status == "complete":
            return "end"
        elif status == "skip":
            return "end"  # End workflow after processing this shot
        elif status == "pass" or status == "fail":
            # Shot processing is complete (all variations done)
            return "end"
        else:
            return "artist"
    
    workflow.add_conditional_edges(
        "director",
        director_router,
        {
            "artist": "artist",
            "director": "director",
            "end": END
        }
    )
    
    # Artist generates image
    def artist_router(state):
        status = state.get("status", "")
        if status == "error":
            return "director"  # Skip to next shot on error
        else:
            return "critic"
    
    workflow.add_conditional_edges(
        "artist",
        artist_router,
        {
            "critic": "critic",
            "director": "director"
        }
    )
    
    # Critic evaluates
    def critic_router(state):
        status = state.get("status", "")
        current_variation = state.get("current_variation", 1)
        variations_per_shot = state.get("variations_per_shot", 1)
        
        if status == "pass":
            # Check if we need more variations
            if current_variation < variations_per_shot:
                # Generate another variation
                return "artist"
            else:
                # All variations done, move to next shot
                return "director"
        elif status == "retry":
            return "artist"
        else:
            # On fail or error, check if we should try another variation
            if current_variation < variations_per_shot:
                return "artist"
            else:
                return "director"
    
    workflow.add_conditional_edges(
        "critic",
        critic_router,
        {
            "artist": "artist",
            "director": "director"
        }
    )
    
    return workflow.compile()

async def process_single_shot(
    shot_id: int,
    workflow,
    output_base_path: str,
    variations_per_shot: int,
    context_window: int,
    historical_context: List[Dict[str, Any]] = None,
    max_retries: int = 1,
    context_variations: int = -1
) -> Dict[str, Any]:
    """Process a single shot with all its variations.
    
    Args:
        shot_id: Shot ID to process
        workflow: Compiled workflow
        output_base_path: Base path for output files
        variations_per_shot: Number of variations per shot
        context_window: Number of previous images to include in context
        historical_context: Previous generation context
        max_retries: Maximum retries per image
        context_variations: Number of variations per shot to include in context
        
    Returns:
        Processing result for the shot
    """
    # Initialize state for this shot
    state = {
        "current_shot_id": shot_id,
        "shot_id": None,
        "shot_text": None,
        "entities": None,
        "attempt": 0,
        "current_variation": 1,
        "status": "start",
        "output_base_path": output_base_path,
        "variations_per_shot": variations_per_shot,
        "image_path": None,
        "image_paths": [],
        "critic_notes": None,
        "error": None,
        "context_window": context_window,
        "historical_context": historical_context or [],
        "max_retries": max_retries,
        "base_context": build_complete_context(historical_context)  # Always include base documents
    }
    
    try:
        # Process this shot through the workflow with increased recursion limit
        config = {"recursion_limit": 50}  # Increase limit for safety
        result = await workflow.ainvoke(state, config=config)
        
        # Return the result for this shot
        return {
            "shot_id": result.get("shot_id", shot_id),
            "status": result.get("status", "unknown"),
            "image_paths": result.get("image_paths", []),
            "historical_context": result.get("historical_context", [])
        }
    except Exception as e:
        console.print(f"[red]Error processing shot {shot_id}: {e}[/red]")
        return {
            "shot_id": shot_id,
            "status": "error",
            "image_paths": [],
            "historical_context": []
        }

async def process_shots(
    start_shot: int,
    max_shots: int,
    workflow,
    progress: Progress,
    task_id: int,
    worker_id: int,
    output_base_path: str,
    variations_per_shot: int,
    context_window: int,
    max_retries: int = 1,
    context_variations: int = -1
) -> List[Dict[str, Any]]:
    """Process a batch of shots with rate limiting.
    
    Args:
        start_shot: Starting shot ID
        max_shots: Maximum number of shots to process
        workflow: Compiled workflow
        progress: Rich progress bar
        task_id: Progress task ID
        worker_id: Worker identifier
        output_base_path: Base path for output files
        variations_per_shot: Number of variations per shot
        context_window: Number of previous images to include in context
        max_retries: Maximum retries per image
        context_variations: Number of variations per shot to include in context
        
    Returns:
        List of processing results
    """
    results = []
    historical_context = []
    
    for i in range(max_shots):
        shot_id = start_shot + i
        console.print(f"Worker {worker_id}: Processing Shot {shot_id}")
        
        # Process through workflow
        start_time = time.time()
        
        # Process each shot independently
        result = await process_single_shot(
            shot_id=shot_id,
            workflow=workflow,
            output_base_path=output_base_path,
            variations_per_shot=variations_per_shot,
            context_window=context_window,
            historical_context=historical_context.copy(),
            max_retries=max_retries,
            context_variations=context_variations
        )
        
        # Check if no more shots to process
        if result["status"] == "complete":
            console.print(f"Worker {worker_id}: No more shots to process")
            break
        
        results.append(result)
        
        # Update historical context if shot was successful
        if result["status"] in ["pass", "skip"] and result.get("historical_context"):
            # Add this shot's successful generations to historical context
            new_data = result["historical_context"]
            
            # Filter variations based on context_variations setting
            if context_variations > 0:
                # Group by shot_id and take only the first N variations
                shot_variations = {}
                for item in new_data:
                    shot_id = item["shot_id"]
                    if shot_id not in shot_variations:
                        shot_variations[shot_id] = []
                    shot_variations[shot_id].append(item)
                
                # Take only the specified number of variations per shot
                filtered_data = []
                for shot_id, variations in shot_variations.items():
                    # Sort by variation number to ensure we take the first ones
                    variations.sort(key=lambda x: x.get("variation", 1))
                    filtered_data.extend(variations[:context_variations])
                
                new_data = filtered_data
            
            historical_context.extend(new_data)
            
            # Apply sliding window if not keeping all history
            if context_window > 0:
                # Keep only the last N successful generations
                historical_context = historical_context[-context_window:]
        
        progress.update(task_id, advance=1)
        
        # Rate limiting
        elapsed = time.time() - start_time
        if elapsed < 60 / RATE_LIMIT:
            await asyncio.sleep(60 / RATE_LIMIT - elapsed)
    
    return results

async def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="AI Storyboard Generator")
    parser.add_argument("--shot-id", type=int, default=1, help="Starting shot ID")
    parser.add_argument("--max-shots", type=int, default=60, help="Maximum number of shots to process")
    parser.add_argument("--variations", type=int, default=DEFAULT_VARIATIONS, help="Number of variations per shot")
    parser.add_argument("--context-window", type=int, default=0, 
                       help="Number of previous images to include in context (0=fresh context, -1=all previous)")
    parser.add_argument("--context-variations", type=int, default=-1,
                       help="Number of variations per shot to include in context (-1=all, 1=first only, etc.)")
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum retries per image (default: 1)")
    args = parser.parse_args()
    
    console.print("[bold blue]Storyboard Generator[/bold blue]")
    console.print(f"Configuration: {RATE_LIMIT} calls/min, {PARALLEL_WORKERS} parallel workers")
    console.print(f"Generating {args.variations} variations per shot")
    console.print(f"Maximum retries per image: {args.max_retries}")
    
    # Display context window setting
    if args.context_window == 0:
        console.print("Context mode: Fresh context per shot (no history)")
    elif args.context_window == -1:
        console.print("Context mode: Full history (all previous images)")
    else:
        console.print(f"Context mode: Sliding window ({args.context_window} previous images)")
        if args.context_variations > 0:
            shots_in_context = args.context_window // args.context_variations
            console.print(f"  - Using {args.context_variations} variation(s) per shot")
            console.print(f"  - Context will span approximately {shots_in_context} shots")
    
    # Create output folder
    output_base_path = create_output_folder()
    console.print(f"Output folder: {output_base_path}")
    
    # Create workflow
    console.print("\nBuilding workflow graph...")
    workflow = create_workflow(output_base_path, args.variations, args.context_window, args.max_retries)
    
    # Create progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    )
    
    # Global historical context to share across all shots
    global_historical_context = []
    
    with progress:
        task_id = progress.add_task("Generating storyboards...", total=args.max_shots)
        
        # Since we're processing shots sequentially for context, use only 1 worker
        if args.context_window != 0:
            console.print("\n[yellow]Context mode enabled: Processing shots sequentially[/yellow]")
            # Process shots sequentially to maintain context
            all_results = []
            for i in range(args.max_shots):
                shot_id = args.shot_id + i
                console.print(f"Processing Shot {shot_id}")
                
                # Process single shot with accumulated context
                result = await process_single_shot(
                    shot_id=shot_id,
                    workflow=workflow,
                    output_base_path=output_base_path,
                    variations_per_shot=args.variations,
                    context_window=args.context_window,
                    historical_context=global_historical_context.copy(),
                    max_retries=args.max_retries,
                    context_variations=args.context_variations
                )
                
                # Check if no more shots to process
                if result["status"] == "complete":
                    console.print("No more shots to process")
                    break
                
                all_results.append(result)
                
                # Update global historical context if shot was successful
                if result["status"] in ["pass", "skip"] and "historical_context" in result:
                    # Get new historical data from this shot
                    new_historical_data = result["historical_context"]
                    
                    # Filter variations based on context_variations setting
                    if args.context_variations > 0:
                        # Group by shot_id and take only the first N variations
                        shot_variations = {}
                        for item in new_historical_data:
                            shot_id = item["shot_id"]
                            if shot_id not in shot_variations:
                                shot_variations[shot_id] = []
                            shot_variations[shot_id].append(item)
                        
                        # Take only the specified number of variations per shot
                        filtered_data = []
                        for shot_id, variations in shot_variations.items():
                            # Sort by variation number to ensure we take the first ones
                            variations.sort(key=lambda x: x.get("variation", 1))
                            filtered_data.extend(variations[:args.context_variations])
                        
                        new_historical_data = filtered_data
                    
                    # Only add the new items that aren't already in global context
                    for item in new_historical_data:
                        if not any(h["shot_id"] == item["shot_id"] and h["variation"] == item["variation"] 
                                  for h in global_historical_context):
                            global_historical_context.append(item)
                    
                    # Apply sliding window if not keeping all history
                    if args.context_window > 0:
                        global_historical_context = global_historical_context[-args.context_window:]
                
                progress.update(task_id, advance=1)
                
                # Rate limiting
                await asyncio.sleep(60 / RATE_LIMIT)
        else:
            # Original parallel processing for no-context mode
            tasks = []
            for i in range(PARALLEL_WORKERS):
                shots_per_worker = args.max_shots // PARALLEL_WORKERS
                if i < args.max_shots % PARALLEL_WORKERS:
                    shots_per_worker += 1
                
                if shots_per_worker > 0:
                    start_shot = args.shot_id + i * (args.max_shots // PARALLEL_WORKERS)
                    task = process_shots(
                        start_shot=start_shot,
                        max_shots=shots_per_worker,
                        workflow=workflow,
                        progress=progress,
                        task_id=task_id,
                        worker_id=i + 1,
                        output_base_path=output_base_path,
                        variations_per_shot=args.variations,
                        context_window=args.context_window,
                        max_retries=args.max_retries,
                        context_variations=args.context_variations
                    )
                    tasks.append(task)
            
            # Wait for all workers
            all_results = []
            for task in asyncio.as_completed(tasks):
                results = await task
                all_results.extend(results)
    
    # Summary
    console.print("\n[bold green]Generation Complete![/bold green]")
    console.print("\nSummary:")
    passed = sum(1 for r in all_results if r["status"] == "pass")
    failed = sum(1 for r in all_results if r["status"] == "fail")
    skipped = sum(1 for r in all_results if r["status"] == "skip")
    console.print(f"  Passed: {passed}")
    console.print(f"  Failed: {failed}")  
    console.print(f"  Skipped: {skipped}")
    
    # List generated images
    if passed > 0:
        console.print("\nGenerated images:")
        for result in all_results:
            if result["status"] == "pass" and result.get("image_paths"):
                shot_id = result["shot_id"]
                console.print(f"\n  Shot {shot_id}:")
                for path in result["image_paths"]:
                    console.print(f"    - {path}")

if __name__ == "__main__":
    asyncio.run(main()) 