"""
LangGraph workflow for the AI Storyboard Generator
Orchestrates the Planner -> Renderer -> Vision QA pipeline
"""

from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from memory import (
    MemoryService,
    AgentState,
    RAGConfig,
    EntityState,
    CameraSpec as MemCameraSpec,
    EnvironmentSpec as MemEnvironmentSpec,
    ImageRef
)
from agents_v2 import PlannerAgent, RendererAgent, VisionQAAgent


class WorkflowState(TypedDict):
    """State passed between workflow nodes"""
    agent_state: AgentState
    memory_service: MemoryService
    config: RAGConfig
    planner: PlannerAgent
    renderer: RendererAgent
    vision_qa: VisionQAAgent


class StoryboardWorkflow:
    """
    Main workflow orchestrating the storyboard generation process
    """
    
    def __init__(self, 
                 memory_service: MemoryService,
                 config: RAGConfig,
                 output_dir: str):
        self.memory = memory_service
        self.config = config
        self.output_dir = Path(output_dir)
        
        # Initialize agents
        self.planner = PlannerAgent(memory_service, config)
        self.renderer = RendererAgent(memory_service, config, output_dir)
        self.vision_qa = VisionQAAgent(memory_service, config)
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create workflow
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("plan", self._plan_scene)
        workflow.add_node("render", self._render_image)
        workflow.add_node("qa", self._qa_image)
        workflow.add_node("write_memory", self._write_memory)
        workflow.add_node("retry_check", self._check_retry)
        workflow.add_node("variation_check", self._check_variation)
        
        # Add edges
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "render")
        workflow.add_edge("render", "qa")
        workflow.add_edge("qa", "retry_check")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "retry_check",
            self._should_retry,
            {
                True: "render",  # Retry rendering
                False: "variation_check"  # Check if more variations needed
            }
        )
        
        workflow.add_conditional_edges(
            "variation_check",
            self._should_continue_variations,
            {
                True: "render",  # Generate next variation
                False: "write_memory"  # All variations done
            }
        )
        
        workflow.add_edge("write_memory", END)
        
        return workflow.compile()
    
    def _plan_scene(self, state: WorkflowState) -> WorkflowState:
        """Plan the scene using Planner agent"""
        agent_state = state["agent_state"]
        
        print(f"\n=== Planning Scene {agent_state.current_scene_id} ===")
        
        # Generate scene plan
        scene_plan = self.planner.plan_scene(
            agent_state.current_scene_id,
            agent_state.current_shot_description
        )
        
        # Update state
        agent_state.scene_plan = scene_plan
        
        print(f"Detected entities: {[e.name for e in scene_plan.entities]}")
        print(f"Retrieved {len(scene_plan.retrieved_context)} relevant memories")
        
        return state
    
    def _render_image(self, state: WorkflowState) -> WorkflowState:
        """Render image using Renderer agent"""
        agent_state = state["agent_state"]
        
        if not agent_state.scene_plan:
            agent_state.add_error("No scene plan available for rendering")
            return state
        
        print(f"\n=== Rendering Shot {agent_state.current_scene_id}, "
              f"Variation {agent_state.current_variation + 1}, "
              f"Retry {agent_state.current_retry_count} ===")
        
        # Get retry guidance if this is a retry
        retry_guidance = None
        if agent_state.current_retry_count > 0 and agent_state.generated_images:
            last_image = agent_state.generated_images[-1]
            if last_image.qa_result and not last_image.qa_result.passed:
                retry_guidance = self.vision_qa.get_retry_guidance(last_image.qa_result)
        
        # Generate image
        generated_image = self.renderer.render_image(
            agent_state.scene_plan,
            agent_state.current_variation,
            agent_state.current_retry_count,
            retry_guidance
        )
        
        # Store in state (replace if retry, append if new variation)
        if agent_state.current_retry_count > 0 and agent_state.generated_images:
            # Replace last image with retry
            agent_state.generated_images[-1] = generated_image
        else:
            # Add new image
            agent_state.generated_images.append(generated_image)
        
        return state
    
    def _qa_image(self, state: WorkflowState) -> WorkflowState:
        """Evaluate image using Vision QA agent"""
        agent_state = state["agent_state"]
        
        if not agent_state.generated_images:
            agent_state.add_error("No images to evaluate")
            return state
        
        # Get latest image
        latest_image = agent_state.generated_images[-1]
        
        if not latest_image.path:
            # Image generation failed
            qa_result = self.vision_qa.evaluate_image(
                agent_state.scene_plan,
                latest_image
            )
        else:
            print(f"\n=== Evaluating Image Quality ===")
            
            # Evaluate image
            qa_result = self.vision_qa.evaluate_image(
                agent_state.scene_plan,
                latest_image
            )
            
            print(f"QA Status: {qa_result.status}")
            print(f"Scores: {qa_result.scores}")
        
        # Update image with QA result
        latest_image.qa_result = qa_result
        
        return state
    
    def _check_retry(self, state: WorkflowState) -> WorkflowState:
        """Check if we should retry the current generation"""
        agent_state = state["agent_state"]
        
        # Check if latest image passed QA
        if agent_state.generated_images:
            latest_image = agent_state.generated_images[-1]
            if latest_image.qa_result and latest_image.qa_result.passed:
                # Passed! Move to next variation
                return state
        
        # Failed - increment retry count
        agent_state.current_retry_count += 1
        
        return state
    
    def _should_retry(self, state: WorkflowState) -> bool:
        """Determine if we should retry the current generation"""
        agent_state = state["agent_state"]
        
        # Check if latest image failed and we have retries left
        if agent_state.generated_images:
            latest_image = agent_state.generated_images[-1]
            if latest_image.qa_result and not latest_image.qa_result.passed:
                return agent_state.should_retry()
        
        return False
    
    def _check_variation(self, state: WorkflowState) -> WorkflowState:
        """Check if we should generate more variations"""
        agent_state = state["agent_state"]
        
        # Move to next variation
        has_more = agent_state.next_variation()
        
        return state
    
    def _should_continue_variations(self, state: WorkflowState) -> bool:
        """Determine if we should generate more variations"""
        agent_state = state["agent_state"]
        
        return agent_state.current_variation < agent_state.total_variations
    
    def _write_memory(self, state: WorkflowState) -> WorkflowState:
        """Write episodic memory for the completed scene"""
        agent_state = state["agent_state"]
        
        if not agent_state.scene_plan:
            return state
        
        print(f"\n=== Writing Episodic Memory for Scene {agent_state.current_scene_id} ===")
        
        # Collect all successful images
        successful_images = []
        visual_elements = []
        critic_tags = set()
        
        for img in agent_state.generated_images:
            if img.qa_result and img.qa_result.passed:
                successful_images.append(ImageRef(
                    path=img.path,
                    variation_id=img.variation_id,
                    retry_count=img.retry_count,
                    quality_score=img.qa_result.overall_score
                ))
                
                # Collect critic tags from issues
                for issue in img.qa_result.issues:
                    if issue.get("type"):
                        critic_tags.add(issue["type"])
        
        # Extract visual elements from scene plan
        visual_elements = agent_state.scene_plan.visual_beats
        
        # Convert entities to EntityState objects
        entity_states = []
        for entity in agent_state.scene_plan.entities:
            entity_states.append(EntityState(
                name=entity.name,
                pose=entity.pose,
                emotion=entity.emotion,
                visual_ref_ids=entity.visual_ref_ids
            ))
        
        # Convert camera and environment specs
        camera_spec = MemCameraSpec(
            type=agent_state.scene_plan.camera.type,
            angle=agent_state.scene_plan.camera.angle,
            movement=agent_state.scene_plan.camera.movement,
            fov=agent_state.scene_plan.camera.fov
        )
        
        environment_spec = MemEnvironmentSpec(
            setting=agent_state.scene_plan.environment.setting,
            lighting=agent_state.scene_plan.environment.lighting,
            mood=agent_state.scene_plan.environment.mood
        )
        
        # Write episodic memory
        memory_item = self.memory.write_episodic_memory(
            scene_id=agent_state.current_scene_id,
            shot_description=agent_state.current_shot_description,
            entities=[e.name for e in agent_state.scene_plan.entities],
            entity_states=entity_states,
            camera_spec=camera_spec,
            environment_spec=environment_spec,
            visual_elements=visual_elements,
            generated_images=successful_images,
            critic_tags=list(critic_tags)
        )
        
        print(f"Wrote episodic memory with {len(successful_images)} successful images")
        
        return state
    
    def process_scene(self, 
                     scene_id: int, 
                     shot_description: str,
                     run_id: str) -> List[GeneratedImage]:
        """
        Process a single scene through the workflow
        """
        # Create initial state
        agent_state = AgentState(
            current_scene_id=scene_id,
            current_shot_description=shot_description,
            run_id=run_id,
            output_dir=str(self.output_dir),
            total_variations=self.config.variations_per_shot,
            max_retries=self.config.max_retries_per_variation
        )
        
        initial_state: WorkflowState = {
            "agent_state": agent_state,
            "memory_service": self.memory,
            "config": self.config,
            "planner": self.planner,
            "renderer": self.renderer,
            "vision_qa": self.vision_qa
        }
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Return generated images
        return final_state["agent_state"].generated_images 