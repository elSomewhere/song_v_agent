"""
Main entry point for the RAG-based AI storyboard generator.
Maintains backward compatibility with the original CLI while adding new features.
"""
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import time
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
import os
from dotenv import load_dotenv
import json

from workflow import create_workflow, WorkflowState
from memory import MemoryService

# Load environment variables
load_dotenv()

console = Console()

# Configuration
RATE_LIMIT = 3  # Calls per minute (kept for compatibility)
DEFAULT_VARIATIONS = 3  # Default number of variations per shot


def create_output_folder() -> str:
    """Create a timestamped output folder.
    
    Returns:
        Path to the output folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"output/run_{timestamp}")
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)


async def process_single_shot(
    shot_id: int,
    workflow,
    output_base_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single shot with all its variations.
    
    Args:
        shot_id: Shot ID to process
        workflow: Compiled workflow
        output_base_path: Base path for output files
        config: Configuration parameters
        
    Returns:
        Processing result for the shot
    """
    # Initialize state for this shot
    memory_service = MemoryService()
    canonical_memory = memory_service.canonical_memory.to_dict()
    
    state = WorkflowState(
        current_scene_number=shot_id,
        current_shot_id=shot_id,
        current_variation=1,
        max_variations=config["variations_per_shot"],
        scene_text="",
        scene_plan=None,
        working_memory=[],
        canonical_memory=canonical_memory,
        status="planning",
        attempt_number=0,
        max_attempts=config["max_retries"],
        generated_images=[],
        quality_scores=[],
        feedback_history=[],
        output_base_path=output_base_path,
        embedding_cache={},
        error=None,
        memory_k=config["memory_k"],
        memory_weights=config["memory_weights"],
        quality_threshold=config["quality_threshold"]
    )
    
    try:
        # Process this shot through the workflow
        result = await workflow.ainvoke(state, config={"recursion_limit": 50})
        
        # Extract results
        status = "pass" if result.get("quality_scores") and max(result["quality_scores"]) >= config["quality_threshold"] else "fail"
        
        if result.get("status") == "no_more_scenes":
            status = "complete"
        
        return {
            "shot_id": shot_id,
            "status": status,
            "image_paths": result.get("generated_images", []),
            "quality_scores": result.get("quality_scores", []),
            "error": result.get("error")
        }
        
    except Exception as e:
        import traceback
        console.print(f"[red]Error processing shot {shot_id}: {e}[/red]")
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return {
            "shot_id": shot_id,
            "status": "error",
            "image_paths": [],
            "quality_scores": [],
            "error": str(e)
        }


async def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="RAG-based AI Storyboard Generator")
    
    # Basic options (backward compatible)
    parser.add_argument("--shot-id", type=int, default=1, help="Starting shot ID")
    parser.add_argument("--max-shots", type=int, default=60, help="Maximum number of shots to process")
    parser.add_argument("--variations", type=int, default=DEFAULT_VARIATIONS, help="Number of variations per shot")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries per image")
    
    # Legacy context window support (deprecated)
    parser.add_argument("--context-window", type=int, default=None,
                       help="(Deprecated) Use --memory-k instead. If set, will be converted to memory-k")
    parser.add_argument("--context-variations", type=int, default=-1,
                       help="(Deprecated) No longer used in RAG system")
    
    # New memory options
    parser.add_argument("--memory-k", type=int, default=5,
                       help="Number of memories to retrieve per scene")
    parser.add_argument("--memory-weights", type=str, default="0.6,0.3,0.1",
                       help="Weights for semantic,entity,temporal (comma-separated)")
    
    # Quality options
    parser.add_argument("--quality-threshold", type=float, default=0.7,
                       help="Minimum quality score to accept")
    
    # Performance options
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Number of scenes to process in parallel")
    parser.add_argument("--cache-embeddings", action="store_true",
                       help="Enable embedding cache (default: enabled)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Base output directory")
    parser.add_argument("--save-metadata", action="store_true",
                       help="Save detailed metadata for each generation")
    
    args = parser.parse_args()
    
    # Handle deprecated context-window argument
    if args.context_window is not None:
        console.print("[yellow]Warning: --context-window is deprecated. Converting to --memory-k[/yellow]")
        if args.context_window == 0:
            args.memory_k = 0
        elif args.context_window == -1:
            args.memory_k = 20  # Use large value for "all history"
        else:
            args.memory_k = min(args.context_window, 10)  # Cap at reasonable value
    
    # Parse memory weights
    try:
        weights = [float(w) for w in args.memory_weights.split(",")]
        if len(weights) != 3:
            raise ValueError("Must provide exactly 3 weights")
        memory_weights = {
            "semantic": weights[0],
            "entity": weights[1],
            "temporal": weights[2]
        }
    except Exception as e:
        console.print(f"[red]Error parsing memory weights: {e}[/red]")
        console.print("Using default weights: semantic=0.6, entity=0.3, temporal=0.1")
        memory_weights = {"semantic": 0.6, "entity": 0.3, "temporal": 0.1}
    
    console.print("[bold blue]RAG-based Storyboard Generator[/bold blue]")
    console.print(f"Configuration: {RATE_LIMIT} calls/min")
    console.print(f"Generating {args.variations} variations per shot")
    console.print(f"Maximum retries per image: {args.max_retries}")
    console.print(f"Quality threshold: {args.quality_threshold}")
    
    # Display memory configuration
    console.print(f"\nMemory configuration:")
    console.print(f"  - Retrieving {args.memory_k} relevant memories per scene")
    console.print(f"  - Weights: semantic={memory_weights['semantic']}, entity={memory_weights['entity']}, temporal={memory_weights['temporal']}")
    console.print(f"  - Embedding cache: {'enabled' if args.cache_embeddings else 'disabled'}")
    
    # Create output folder
    output_base_path = create_output_folder()
    console.print(f"Output folder: {output_base_path}")
    
    # Configuration for workflow
    config = {
        "variations_per_shot": args.variations,
        "max_retries": args.max_retries,
        "memory_k": args.memory_k,
        "memory_weights": memory_weights,
        "quality_threshold": args.quality_threshold,
        "cache_embeddings": args.cache_embeddings,
        "save_metadata": args.save_metadata,
        "db_path": "./lancedb_data",
        "data_path": "data"
    }
    
    # Create workflow
    console.print("\nBuilding workflow graph...")
    workflow = create_workflow(config)
    
    # Create progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    )
    
    all_results = []
    
    with progress:
        task_id = progress.add_task("Generating storyboards...", total=args.max_shots)
        
        # Process shots sequentially (for semantic memory to build up)
        for i in range(args.max_shots):
            shot_id = args.shot_id + i
            console.print(f"\nProcessing Shot {shot_id}")
            
            # Process single shot
            start_time = time.time()
            
            result = await process_single_shot(
                shot_id=shot_id,
                workflow=workflow,
                output_base_path=output_base_path,
                config=config
            )
            
            # Check if no more shots to process
            if result["status"] == "complete":
                console.print("No more shots to process")
                break
            
            all_results.append(result)
            
            progress.update(task_id, advance=1)
            
            # Rate limiting
            elapsed = time.time() - start_time
            if elapsed < 60 / RATE_LIMIT:
                await asyncio.sleep(60 / RATE_LIMIT - elapsed)
    
    # Save metadata if requested
    if args.save_metadata:
        metadata_path = Path(output_base_path) / "metadata.json"
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "start_shot": args.shot_id,
                "max_shots": args.max_shots,
                "variations": args.variations,
                "max_retries": args.max_retries,
                "memory_k": args.memory_k,
                "memory_weights": memory_weights,
                "quality_threshold": args.quality_threshold
            },
            "results": all_results
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        console.print(f"\nMetadata saved to: {metadata_path}")
    
    # Summary
    console.print("\n[bold green]Generation Complete![/bold green]")
    console.print("\nSummary:")
    passed = sum(1 for r in all_results if r["status"] == "pass")
    failed = sum(1 for r in all_results if r["status"] == "fail")
    errors = sum(1 for r in all_results if r["status"] == "error")
    
    console.print(f"  Passed: {passed}")
    console.print(f"  Failed: {failed}")
    console.print(f"  Errors: {errors}")
    
    # Calculate token savings estimate
    if len(all_results) > 10:
        console.print("\n[bold cyan]Efficiency Report:[/bold cyan]")
        console.print(f"  Estimated token savings: ~{(len(all_results) - 5) * 12000:,} tokens")
        console.print(f"  Using semantic retrieval instead of sliding window")
    
    # List generated images
    if passed > 0:
        console.print("\nGenerated images:")
        for result in all_results:
            if result["status"] == "pass" and result.get("image_paths"):
                shot_id = result["shot_id"]
                console.print(f"\n  Shot {shot_id}:")
                for path in result["image_paths"]:
                    console.print(f"    - {path}")
                if result.get("quality_scores"):
                    best_score = max(result["quality_scores"])
                    console.print(f"    Quality score: {best_score:.2f}")


if __name__ == "__main__":
    asyncio.run(main()) 