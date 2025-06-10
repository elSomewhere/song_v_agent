"""
Renderer Agent for AI Storyboard Generator
Responsible for generating images based on scene plans
"""

import json
import base64
import io
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import openai
from PIL import Image
import requests

from memory import (
    MemoryService,
    ScenePlan,
    GeneratedImage,
    RAGConfig
)


class RendererAgent:
    """
    Renderer agent that:
    1. Expands canonical placeholders
    2. Constructs optimized prompts
    3. Generates images using GPT-4o vision
    4. Applies style consistency
    """
    
    def __init__(self,
                 memory_service: MemoryService,
                 config: RAGConfig,
                 output_dir: str):
        self.memory = memory_service
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # OpenAI client
        self.client = openai.OpenAI()
        
        # Load style guide
        self._load_style_guide()
    
    def _load_style_guide(self):
        """Load style guide from canonical memory"""
        style_memory = self.memory.canonical.get_by_id("style_guide")
        self.style_guide = style_memory.content if style_memory else ""
        
        # Extract key style directives
        self.style_summary = self._summarize_style_guide()
    
    def _summarize_style_guide(self) -> str:
        """Create a concise summary of the style guide for prompts"""
        if not self.style_guide:
            return "cinematic storyboard style"
        
        # Extract key style elements
        summary_parts = []
        
        # Look for specific style keywords
        style_lower = self.style_guide.lower()
        if "gritty" in style_lower:
            summary_parts.append("gritty")
        if "anime" in style_lower or "manga" in style_lower:
            summary_parts.append("anime/manga influenced")
        if "noir" in style_lower:
            summary_parts.append("noir")
        if "cinematic" in style_lower:
            summary_parts.append("cinematic")
        
        # Default if nothing found
        if not summary_parts:
            summary_parts = ["cinematic", "storyboard"]
        
        return f"{', '.join(summary_parts)} style, 16:9 aspect ratio"
    
    def _load_reference_images(self, scene_plan: ScenePlan) -> Dict[str, List[str]]:
        """Load reference images for entities in the scene"""
        references = {}
        
        for entity in scene_plan.entities:
            entity_refs = []
            
            # Load actual image files
            for ref_id in entity.visual_ref_ids[:3]:  # Limit to 3 refs per entity
                ref_path = self._find_reference_image(entity.name, ref_id)
                if ref_path and ref_path.exists():
                    # Convert to base64 for inclusion in prompt
                    with open(ref_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                        entity_refs.append(f"data:image/jpeg;base64,{img_data}")
            
            if entity_refs:
                references[entity.canonical_id] = entity_refs
        
        return references
    
    def _find_reference_image(self, entity_name: str, ref_id: str) -> Optional[Path]:
        """Find reference image file for an entity"""
        refs_dir = Path("./data/refs") / entity_name
        
        if not refs_dir.exists():
            return None
        
        # Check for different extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            ref_path = refs_dir / f"{ref_id}{ext}"
            if ref_path.exists():
                return ref_path
        
        # If ref_id doesn't include extension, find any matching file
        for file in refs_dir.iterdir():
            if file.stem == ref_id:
                return file
        
        return None
    
    def _build_visual_context(self, scene_plan: ScenePlan) -> str:
        """Build visual context from retrieved memories"""
        if not scene_plan.retrieved_context:
            return ""
        
        context_parts = []
        
        for ctx_ref in scene_plan.retrieved_context[:3]:  # Limit to top 3
            # Get the actual episodic memory
            memory = self.memory.episodic.get_by_scene_id(ctx_ref.scene_id)
            if memory:
                # Extract key visual elements
                context_parts.append(
                    f"Scene {ctx_ref.scene_id} (relevance: {ctx_ref.relevance:.2f}): "
                    f"{', '.join(memory.visual_elements[:3])}"
                )
        
        if context_parts:
            return "Visual continuity context:\n" + "\n".join(context_parts)
        
        return ""
    
    def _construct_prompt(self, 
                         scene_plan: ScenePlan,
                         retry_guidance: Optional[str] = None) -> str:
        """Construct optimized prompt for image generation"""
        
        # Start with style directive
        prompt_parts = [
            f"Create a storyboard frame in {self.style_summary}.",
            f"Shot type: {scene_plan.camera.type} at {scene_plan.camera.angle}."
        ]
        
        # Add camera movement if specified
        if scene_plan.camera.movement:
            prompt_parts.append(f"Camera movement: {scene_plan.camera.movement}.")
        
        # Environment description
        prompt_parts.append(
            f"Setting: {scene_plan.environment.setting} with "
            f"{scene_plan.environment.lighting} lighting, "
            f"{scene_plan.environment.mood} mood."
        )
        
        # Entity descriptions with canonical placeholders
        if scene_plan.entities:
            entity_descriptions = []
            for entity in scene_plan.entities:
                desc = f"{entity.canonical_id} ({entity.name}): {entity.pose}, {entity.emotion}"
                entity_descriptions.append(desc)
            
            prompt_parts.append("Characters/Entities: " + "; ".join(entity_descriptions))
        
        # Visual beats
        if scene_plan.visual_beats:
            prompt_parts.append("Key visual elements: " + ", ".join(scene_plan.visual_beats))
        
        # Add retry guidance if provided
        if retry_guidance:
            prompt_parts.append(f"IMPORTANT: {retry_guidance}")
        
        # Combine all parts
        prompt = " ".join(prompt_parts)
        
        # Add technical requirements
        prompt += " Cinematic composition, professional storyboard quality, 16:9 aspect ratio."
        
        return prompt
    
    def _expand_canonical_references(self, prompt: str, references: Dict[str, List[str]]) -> str:
        """Expand canonical placeholders with actual descriptions"""
        expanded_prompt = prompt
        
        # For each canonical ID, add detailed description
        for canonical_id, ref_images in references.items():
            # In GPT-4o, we can reference the images directly
            # For now, we'll keep the canonical IDs in the prompt
            # The actual image references will be passed separately
            pass
        
        return expanded_prompt
    
    def render_image(self,
                    scene_plan: ScenePlan,
                    variation_id: int,
                    retry_count: int = 0,
                    retry_guidance: Optional[str] = None) -> GeneratedImage:
        """
        Generate a single image based on the scene plan
        """
        # Load reference images
        references = self._load_reference_images(scene_plan)
        
        # Build visual context
        visual_context = self._build_visual_context(scene_plan)
        
        # Construct prompt
        base_prompt = self._construct_prompt(scene_plan, retry_guidance)
        
        # Add visual context if available
        if visual_context:
            full_prompt = f"{base_prompt}\n\n{visual_context}"
        else:
            full_prompt = base_prompt
        
        # Expand canonical references
        final_prompt = self._expand_canonical_references(full_prompt, references)
        
        # Prepare messages for GPT-4o
        messages = [
            {
                "role": "system",
                "content": "You are a professional storyboard artist. Generate images that match the exact specifications provided."
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ]
        
        # Add reference images if available
        # Note: GPT-4o can process images in the conversation
        # but for DALL-E generation, we'll focus on text descriptions
        
        try:
            # Generate image using DALL-E 3
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=final_prompt,
                size=self.config.image_size,
                quality=self.config.image_quality,
                style=self.config.image_style,
                n=1
            )
            
            # Get image URL
            image_url = response.data[0].url
            
            # Download and save image
            image_response = requests.get(image_url)
            image = Image.open(io.BytesIO(image_response.content))
            
            # Create output path
            scene_dir = self.output_dir / f"Shot-{scene_plan.scene_id:03d}"
            scene_dir.mkdir(exist_ok=True)
            
            image_filename = f"var{variation_id + 1}"
            if retry_count > 0:
                image_filename += f"_retry{retry_count}"
            image_filename += ".png"
            
            image_path = scene_dir / image_filename
            image.save(image_path)
            
            # Save prompt for reference
            prompt_filename = f"prompt_var{variation_id + 1}"
            if retry_count > 0:
                prompt_filename += f"_retry{retry_count}"
            prompt_filename += ".json"
            
            prompt_path = scene_dir / prompt_filename
            with open(prompt_path, 'w') as f:
                json.dump({
                    "prompt": final_prompt,
                    "scene_plan": scene_plan.dict(),
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": retry_count,
                    "retry_guidance": retry_guidance
                }, f, indent=2)
            
            # Create GeneratedImage object
            generated_image = GeneratedImage(
                path=str(image_path),
                prompt=final_prompt,
                scene_id=scene_plan.scene_id,
                variation_id=variation_id,
                retry_count=retry_count,
                generation_params={
                    "model": "dall-e-3",
                    "size": self.config.image_size,
                    "quality": self.config.image_quality,
                    "style": self.config.image_style
                }
            )
            
            print(f"Generated image for Shot {scene_plan.scene_id}, "
                  f"Variation {variation_id + 1}, Retry {retry_count}: {image_path}")
            
            return generated_image
            
        except Exception as e:
            print(f"Error generating image: {e}")
            
            # Create error record
            error_image = GeneratedImage(
                path="",
                prompt=final_prompt,
                scene_id=scene_plan.scene_id,
                variation_id=variation_id,
                retry_count=retry_count,
                generation_params={"error": str(e)}
            )
            
            return error_image 