"""
Renderer agent - generates images using GPT-4o vision capabilities.
"""
from typing import Dict, Any, List, Optional, Union
from openai import AsyncOpenAI
import base64
import os
from pathlib import Path
import aiohttp
import asyncio
from datetime import datetime
import json

from memory.schemas import ScenePlan, GenerationResult, EntityState, CameraSpec, EnvironmentSpec


class RendererAgent:
    """Generates images using GPT-4o mini with contextual memory."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    async def render_image(
        self,
        scene_plan: Union[ScenePlan, Dict[str, Any]],
        variation_number: int,
        previous_attempts: List[Dict[str, Any]],
        canonical_appearances: Dict[str, Dict[str, str]],
        output_path: str
    ) -> GenerationResult:
        """Generate an image based on the scene plan."""
        
        # Convert dict to ScenePlan if needed
        if isinstance(scene_plan, dict):
            # Handle nested entity dicts
            if 'entities' in scene_plan:
                entities = []
                for e in scene_plan['entities']:
                    if isinstance(e, dict):
                        entities.append(EntityState(**e))
                    else:
                        entities.append(e)
                scene_plan['entities'] = entities
            
            # Handle camera dict
            if 'camera' in scene_plan and isinstance(scene_plan['camera'], dict):
                scene_plan['camera'] = CameraSpec(**scene_plan['camera'])
            
            # Handle environment dict
            if 'environment' in scene_plan and isinstance(scene_plan['environment'], dict):
                scene_plan['environment'] = EnvironmentSpec(**scene_plan['environment'])
            
            scene_plan = ScenePlan(**scene_plan)
        
        try:
            # First, use GPT-4o to generate a contextually-aware detailed prompt
            contextual_prompt = await self._generate_contextual_prompt(
                scene_plan,
                variation_number,
                previous_attempts,
                canonical_appearances
            )
            
            # Generate image using GPT-4o mini's image generation capability
            start_time = datetime.now()
            
            # Prepare input messages for the image generation
            input_messages = [
                {
                    "role": "system",
                    "content": "You are a professional cinematic artist creating film stills. Generate images that maintain visual consistency and capture the specified mood and composition."
                },
                {
                    "role": "user", 
                    "content": contextual_prompt
                }
            ]
            
            # Try to use the responses API if available in the client
            if hasattr(self.client, 'responses'):
                # Use the client's responses.create method
                response = await self.client.responses.create(
                    model="gpt-4.1-mini",
                    input=input_messages,
                    tools=[{"type": "image_generation"}],
                )
                
                # Convert response to dict if needed
                if hasattr(response, 'model_dump'):
                    result = response.model_dump()
                elif hasattr(response, 'dict'):
                    result = response.dict()
                else:
                    result = dict(response)
            else:
                # Fall back to direct API call
                print("Note: OpenAI client doesn't have responses.create, using direct API call")
                api_key = os.environ.get("OPENAI_API_KEY")
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "gpt-4.1-mini",
                    "input": input_messages,
                    "tools": [{"type": "image_generation"}]
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.openai.com/v1/responses",
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"API request failed with status {response.status}: {error_text}")
                        
                        result = await response.json()
            
            generation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Extract image from the response
            image_url = None
            
            # Check for tool_calls structure (new API format)
            if 'tool_calls' in result and len(result['tool_calls']) > 0:
                for tool_call in result['tool_calls']:
                    if tool_call.get('type') == 'image_generation_call' and 'data' in tool_call:
                        # The data is base64 encoded image
                        image_url = tool_call['data']
                        break
            
            # Check for data array structure
            elif 'data' in result and len(result['data']) > 0:
                for item in result['data']:
                    if item.get('type') == 'image':
                        image_url = item.get('url') or item.get('image') or item.get('data')
                        break
            
            # Try alternative response structure
            elif 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    # The image might be in the content
                    content = choice['message']['content']
                    if isinstance(content, str) and (content.startswith('http') or content.startswith('data:image')):
                        image_url = content
            
            if not image_url:
                raise Exception(f"No image URL found in response: {json.dumps(result, indent=2)}")
            
            # Save the image
            image_path = await self._save_image_data(
                image_url,
                output_path,
                scene_plan.shot_id,
                variation_number
            )
            
            return GenerationResult(
                status="success",
                image_data=image_path,
                generation_metadata={
                    "model": "gpt-4.1-mini",
                    "timestamp": datetime.now().isoformat(),
                    "generation_time_ms": int(generation_time),
                    "prompt_used": contextual_prompt,
                    "variation": variation_number
                }
            )
            
        except Exception as e:
            print(f"Error in renderer: {e}")
            return GenerationResult(
                status="error",
                error=str(e),
                generation_metadata={
                    "timestamp": datetime.now().isoformat(),
                    "variation": variation_number
                }
            )
    
    async def _generate_contextual_prompt(
        self,
        scene_plan: ScenePlan,
        variation_number: int,
        previous_attempts: List[Dict[str, Any]],
        canonical_appearances: Dict[str, Dict[str, str]]
    ) -> str:
        """Use GPT-4o to generate a contextually-aware image prompt."""
        
        # Build the context for GPT-4o
        context = self._build_generation_context(
            scene_plan, canonical_appearances, previous_attempts
        )
        
        # Variation-specific instructions
        variation_instructions = self._get_variation_instructions(variation_number)
        
        messages = [
            {
                "role": "system",
                "content": """You are a cinematic visual director specializing in creating detailed image generation prompts.
Your task is to translate scene plans into rich, visually descriptive prompts that maintain consistency across a storyboard.
Focus on:
1. Precise character appearances and poses
2. Specific camera angles and framing
3. Detailed environment and lighting
4. Consistent visual style throughout the narrative
5. Clear action and emotional beats

Always output a single, detailed prompt that will generate a cinematic film still."""
            },
            {
                "role": "user",
                "content": f"""Create a detailed image generation prompt for this scene:

Scene Context:
{json.dumps(context, indent=2)}

Variation Instructions:
{variation_instructions}

Generate a prompt that:
- Maintains visual consistency with established character designs
- Captures the specific mood and atmosphere
- Includes all essential visual elements
- Follows cinematic composition principles
- Is suitable for a high-quality film still

Output only the image generation prompt, no other text."""
            }
        ]
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def _build_generation_context(
        self,
        scene_plan: ScenePlan,
        canonical_appearances: Dict[str, Dict[str, str]],
        previous_attempts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build comprehensive context for prompt generation."""
        
        # Convert camera and environment to dicts, handling both dict and model cases
        camera_dict = {}
        if scene_plan.camera:
            if hasattr(scene_plan.camera, 'model_dump'):
                camera_dict = scene_plan.camera.model_dump()
            elif isinstance(scene_plan.camera, dict):
                camera_dict = scene_plan.camera
        
        env_dict = {}
        if scene_plan.environment:
            if hasattr(scene_plan.environment, 'model_dump'):
                env_dict = scene_plan.environment.model_dump()
            elif isinstance(scene_plan.environment, dict):
                env_dict = scene_plan.environment
        
        context = {
            "shot_id": scene_plan.shot_id,
            "scene_description": scene_plan.image_prompt,
            "entities": [],
            "camera": camera_dict,
            "environment": env_dict,
            "mood": scene_plan.mood,
            "visual_focus": scene_plan.visual_focus,
            "action_beats": scene_plan.action_beats
        }
        
        # Add detailed entity information
        for entity in scene_plan.entities:
            entity_info = {
                "name": entity.name if hasattr(entity, 'name') else str(entity),
                "pose": entity.pose if hasattr(entity, 'pose') else "standing",
                "emotion": entity.emotion if hasattr(entity, 'emotion') else "neutral",
                "position": entity.position if hasattr(entity, 'position') else "center"
            }
            
            # Add canonical appearance
            entity_name = entity_info["name"]
            if entity_name in canonical_appearances:
                entity_info["canonical_appearance"] = canonical_appearances[entity_name]
            
            context["entities"].append(entity_info)
        
        # Add previous attempt feedback if retrying
        if previous_attempts:
            latest_feedback = previous_attempts[-1]
            context["previous_feedback"] = {
                "issues": latest_feedback.get("issues", []),
                "suggestions": latest_feedback.get("suggestions", [])
            }
        
        return context
    
    def _get_variation_instructions(self, variation_number: int) -> str:
        """Get specific instructions for each variation."""
        
        variations = {
            1: "Standard composition following the scene plan exactly. Focus on clarity and readability.",
            2: "Alternative angle or perspective while maintaining the same action and mood. Consider a different camera position.",
            3: "Emphasis on emotional impact through closer framing or dramatic lighting. Highlight character expressions.",
            4: "Environmental or atmospheric focus. Wider shot showing more context and setting details.",
            5: "Dynamic or action-oriented composition. Capture movement and energy in the scene."
        }
        
        return variations.get(variation_number, variations[1])
    
    async def _save_image_data(
        self,
        image_data: str,
        output_path: str,
        shot_id: int,
        variation: int
    ) -> str:
        """Save image data (URL or base64) locally."""
        
        # Create output directory if needed
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"shot_{shot_id:03d}_var_{variation}.png"
        filepath = output_dir / filename
        
        # Check if image_data is a URL or base64
        if image_data.startswith('http'):
            # Download from URL
            async with aiohttp.ClientSession() as session:
                async with session.get(image_data) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        with open(filepath, 'wb') as f:
                            f.write(image_bytes)
        elif image_data.startswith('data:image'):
            # Handle data URL
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
        else:
            # Assume it's base64 encoded
            image_bytes = base64.b64decode(image_data)
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
        
        return str(filepath)
 