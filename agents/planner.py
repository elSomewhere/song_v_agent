"""
Planner agent - analyzes scenes and creates structured generation plans.
"""
from typing import Dict, Any, List
from openai import AsyncOpenAI
import json
import os
import re

from memory.schemas import ScenePlan, EntityState, CameraSpec, EnvironmentSpec


class PlannerAgent:
    """Analyzes script scenes and creates detailed visual plans."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    async def plan_scene(
        self,
        scene_text: str,
        scene_number: int,
        working_memory: List[Dict[str, Any]],
        canonical_style: str,
        canonical_entities: Dict[str, Any]
    ) -> ScenePlan:
        """Create a detailed plan for the scene."""
        
        # Build the prompt
        prompt = self._build_prompt(
            scene_text, scene_number, working_memory, 
            canonical_style, canonical_entities
        )
        
        # Get response from GPT-4
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a film director planning visual compositions for scenes. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            plan_data = json.loads(response.choices[0].message.content)
            
            # Ensure all required fields are present
            plan_data = self._validate_and_complete_plan(plan_data, scene_number)
            
            # Convert to ScenePlan object
            return ScenePlan(**plan_data)
            
        except Exception as e:
            print(f"Error in planner: {e}")
            # Return a minimal plan as fallback
            return self._create_fallback_plan(scene_text, scene_number)
    
    def _build_prompt(
        self,
        scene_text: str,
        scene_number: int,
        working_memory: List[Dict[str, Any]],
        canonical_style: str,
        canonical_entities: Dict[str, Any]
    ) -> str:
        """Build the prompt for scene planning."""
        
        # Format working memory
        memory_context = self._format_working_memory(working_memory)
        
        # Format entity information
        entity_info = self._format_entity_info(canonical_entities)
        
        prompt = f"""You are a film director planning the visual composition for a scene.

VISUAL STYLE GUIDE:
{canonical_style}

CHARACTER REFERENCE:
{entity_info}

RELEVANT PREVIOUS SCENES (for continuity):
{memory_context}

CURRENT SCENE (Scene {scene_number}):
{scene_text}

Create a detailed visual plan for this scene. Consider:
1. Character positioning, poses, and emotional states
2. Camera angles, movements, and framing
3. Environmental details, lighting, and atmosphere
4. Visual continuity with the established style
5. Key action beats that need to be captured

Output your plan as a JSON object with this exact structure:
{{
    "scene_id": {scene_number},
    "shot_id": {scene_number},
    "entities": [
        {{
            "name": "character name",
            "pose": "specific pose description",
            "emotion": "emotional state",
            "position": "position in frame",
            "clothing_state": "any changes to clothing/appearance"
        }}
    ],
    "camera": {{
        "type": "shot type (e.g., close-up, wide shot)",
        "angle": "camera angle",
        "distance": "distance from subject",
        "movement": "any camera movement"
    }},
    "environment": {{
        "location": "specific location",
        "time_of_day": "time",
        "weather": "weather conditions",
        "lighting": "lighting description"
    }},
    "action_beats": ["key action 1", "key action 2"],
    "visual_focus": "primary visual element to emphasize",
    "mood": "overall mood/tone",
    "continuity_notes": "notes on continuity from previous scenes",
    "image_prompt": "complete prompt for image generation"
}}

Ensure the image_prompt is detailed and incorporates all visual elements, style guidelines, and character appearances."""
        
        return prompt
    
    def _format_working_memory(self, working_memory: List[Dict[str, Any]]) -> str:
        """Format working memory for the prompt."""
        if not working_memory:
            return "No previous scenes available for context."
        
        formatted = []
        for memory in working_memory[:3]:  # Limit to top 3 most relevant
            scene_summary = f"Scene {memory['shot_id']}: "
            
            # Add entity information
            if memory.get('entities'):
                scene_summary += f"Characters: {', '.join(memory['entities'])}. "
            
            # Add environment
            if memory.get('environment'):
                env = memory['environment']
                scene_summary += f"Location: {env.get('location', 'unknown')}. "
            
            # Add mood
            if memory.get('mood'):
                scene_summary += f"Mood: {memory['mood']}. "
            
            formatted.append(scene_summary)
        
        return "\n".join(formatted)
    
    def _format_entity_info(self, canonical_entities: Dict[str, Any]) -> str:
        """Format entity information for the prompt."""
        if not canonical_entities:
            return "No character reference available."
        
        formatted = []
        for name, details in canonical_entities.items():
            char_info = f"{name}:\n"
            if details.get('appearance'):
                char_info += f"- Appearance: {details['appearance']}\n"
            if details.get('personality'):
                char_info += f"- Personality: {details['personality']}\n"
            formatted.append(char_info)
        
        return "\n".join(formatted)
    
    def _validate_and_complete_plan(self, plan_data: Dict[str, Any], scene_number: int) -> Dict[str, Any]:
        """Validate and complete the plan with any missing fields."""
        
        # Ensure basic fields
        plan_data.setdefault("scene_id", scene_number)
        plan_data.setdefault("shot_id", scene_number)
        plan_data.setdefault("entities", [])
        plan_data.setdefault("action_beats", [])
        plan_data.setdefault("visual_focus", "scene composition")
        plan_data.setdefault("mood", "neutral")
        
        # Ensure camera spec
        if "camera" not in plan_data:
            plan_data["camera"] = {
                "type": "medium shot",
                "angle": "eye level",
                "distance": "medium",
                "movement": None
            }
        
        # Ensure environment spec
        if "environment" not in plan_data:
            plan_data["environment"] = {
                "location": "unspecified",
                "time_of_day": "day",
                "weather": "clear",
                "lighting": "natural"
            }
        
        # Ensure image prompt
        if "image_prompt" not in plan_data:
            plan_data["image_prompt"] = self._generate_basic_prompt(plan_data)
        
        return plan_data
    
    def _generate_basic_prompt(self, plan_data: Dict[str, Any]) -> str:
        """Generate a basic image prompt from plan data."""
        parts = []
        
        # Add environment
        env = plan_data.get("environment", {})
        parts.append(f"{env.get('location', 'scene')} at {env.get('time_of_day', 'day')}")
        
        # Add camera
        camera = plan_data.get("camera", {})
        parts.append(f"{camera.get('type', 'shot')} from {camera.get('angle', 'angle')}")
        
        # Add entities
        entities = plan_data.get("entities", [])
        if entities:
            entity_desc = ", ".join([f"{e['name']} {e['pose']}" for e in entities if isinstance(e, dict)])
            parts.append(entity_desc)
        
        # Add mood
        parts.append(f"{plan_data.get('mood', 'atmospheric')} mood")
        
        return ", ".join(parts)
    
    def _create_fallback_plan(self, scene_text: str, scene_number: int) -> ScenePlan:
        """Create a minimal fallback plan."""
        # Extract any entities mentioned in the scene
        entities = []
        
        # Simple entity extraction (can be improved)
        words = scene_text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and i > 0:  # Capitalized word not at start
                entities.append({
                    "name": word,
                    "pose": "standing",
                    "emotion": "neutral",
                    "position": "center frame",
                    "clothing_state": None
                })
        
        return ScenePlan(
            scene_id=scene_number,
            shot_id=scene_number,
            entities=entities[:3],  # Limit to 3 entities
            camera=CameraSpec(
                type="medium shot",
                angle="eye level", 
                distance="medium",
                movement=None
            ),
            environment=EnvironmentSpec(
                location="interior",
                time_of_day="day",
                weather=None,
                lighting="ambient"
            ),
            action_beats=["scene action"],
            visual_focus="characters",
            mood="dramatic",
            continuity_notes=None,
            image_prompt=f"Film still: {scene_text[:100]}..."
        ) 