"""
Artist Agent - Generates storyboard frames using OpenAI's image generation.
Collects references, builds prompts, and creates images.
"""
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import aiofiles
import base64
from io import BytesIO
import httpx
from agents.shared import (
    load_style_primer, 
    to_data_url, 
    get_refs_for, 
    get_global_style_refs,
    get_similar_shots,
    load_style_guidelines,
    load_entities_document
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize async OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def save_base64_image(base64_str: str, save_path: Path) -> None:
    """Save a base64 encoded image to file.
    
    Args:
        base64_str: Base64 encoded image string
        save_path: Path to save the image
    """
    # Decode base64 to bytes
    image_data = base64.b64decode(base64_str)
    
    # Save image
    async with aiofiles.open(save_path, 'wb') as f:
        await f.write(image_data)

async def extract_style_from_data(base_context: Dict[str, Any] = None) -> str:
    """Extract style information from base context or data folder.
    
    Args:
        base_context: Base context containing documents
    
    Returns:
        Extracted style description
    """
    style_components = []
    
    # If base context is provided, use it
    if base_context and 'base_documents' in base_context:
        base_docs = base_context['base_documents']
        
        # Use style guidelines from base context
        if 'style_guidelines' in base_docs:
            style_components.append(base_docs['style_guidelines'])
        
        # Use entities document from base context
        if 'entities_document' in base_docs:
            entities_doc = base_docs['entities_document']
            # Extract key visual descriptions
            if "Color Palette" in entities_doc:
                style_components.append("Reference the color palette descriptions from the entities document.")
    else:
        # Fallback to loading directly (for backwards compatibility)
        # Load style guidelines from style.md
        try:
            style_guidelines = load_style_guidelines()
            style_components.append(style_guidelines)
        except:
            pass
        
        # Load entities document for character/environment descriptions
        try:
            entities_doc = load_entities_document()
            # Extract key visual descriptions
            if "Color Palette" in entities_doc:
                style_components.append("Reference the color palette descriptions from the entities document.")
        except:
            pass
    
    # Get reference images to analyze style
    refs_path = Path("data/refs")
    if refs_path.exists():
        # Collect some reference images for style analysis
        ref_images = []
        for subdir in refs_path.iterdir():
            if subdir.is_dir():
                for img_file in list(subdir.glob("*.png"))[:1] + list(subdir.glob("*.jpg"))[:1]:
                    if img_file.exists() and len(ref_images) < 3:
                        ref_images.append(str(img_file))
        
        if ref_images:
            # Use GPT-4 vision to analyze style from reference images
            try:
                content = [
                    {
                        "type": "text",
                        "text": "Analyze the visual style of these reference images. Describe the art style, color palette, rendering technique, and any distinctive visual characteristics:"
                    }
                ]
                
                for img_path in ref_images[:3]:  # Limit to 3 images
                    try:
                        data_url = to_data_url(img_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        })
                    except:
                        pass
                
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": content
                    }],
                    max_tokens=200,
                    temperature=0.3
                )
                
                style_analysis = response.choices[0].message.content
                style_components.append(f"Visual style analysis from references: {style_analysis}")
            except Exception as e:
                print(f"Error analyzing reference styles: {e}")
    
    # Combine all style components
    if style_components:
        return "\n\n".join(style_components)
    else:
        # Fallback to a generic style
        return "cinematic storyboard style with detailed compositions"

async def generate_variation_prompt(state: Dict[str, Any], variation_num: int) -> str:
    """Generate a dynamic variation prompt using GPT.
    
    Args:
        state: Current workflow state containing shot info and historical context
        variation_num: The variation number (1-based)
        
    Returns:
        Dynamic variation instructions for this specific shot
    """
    shot_text = state['shot_text']
    shot_id = state['shot_id']
    historical_context = state.get('historical_context', [])
    context_window = state.get('context_window', 0)
    
    # For the first variation, return empty string (use original shot as-is)
    if variation_num == 1:
        return ""
    
    # Build prompt for GPT to generate variation
    messages = [
        {
            "role": "system",
            "content": """You are a film director creating storyboard variations. For each shot, suggest creative variations that:
1. Use completely different camera angles (e.g., bird's eye, worm's eye, dutch angle, over-the-shoulder)
2. Focus on different narrative elements or characters in the scene
3. Create detail shots that emphasize specific parts of the description
4. Provide alternative compositions that enhance the storytelling

Each variation should feel like a different "sub-shot" that could appear in a film or comic book sequence, telling the same story moment from a fresh perspective.

Return ONLY the variation instruction, starting with '\n\nFor this variation,'."""
        }
    ]
    
    # Build the user message
    user_content = f"""Shot {shot_id} Description:
{shot_text}

This is variation {variation_num} of this shot."""
    
    # Add context about previous variations if available
    if context_window != 0 and historical_context:
        # Find previous variations of this same shot
        prev_variations = [h for h in historical_context if h.get('shot_id') == shot_id]
        if prev_variations:
            user_content += "\n\nPrevious variations of this shot:"
            for prev in prev_variations:
                var_num = prev.get('variation', 1)
                var_prompt = prev.get('variation_prompt', 'standard shot')
                if var_prompt:
                    # Extract just the key instruction part
                    if "For this variation," in var_prompt:
                        var_prompt = var_prompt.split("For this variation,")[1].strip()
                    user_content += f"\n- Variation {var_num}: {var_prompt[:100]}..."
                else:
                    user_content += f"\n- Variation {var_num}: standard shot"
        
        # Also mention other recent shots for continuity
        recent_shots = [h for h in historical_context[-3:] if h.get('shot_id') != shot_id]
        if recent_shots:
            user_content += "\n\nRecent shots in the sequence for context:"
            for recent in recent_shots:
                user_content += f"\n- Shot {recent.get('shot_id')}"
    
    user_content += f"\n\nSuggest a creative camera angle or composition variation for variation {variation_num}. Make it distinct from the original and any previous variations."
    
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150,
            temperature=0.8  # Higher temperature for more creative variations
        )
        
        variation_prompt = response.choices[0].message.content.strip()
        
        # Ensure it starts with the expected format
        if not variation_prompt.startswith("\n\nFor this variation,"):
            variation_prompt = f"\n\nFor this variation, {variation_prompt}"
        
        return variation_prompt
        
    except Exception as e:
        print(f"Error generating variation prompt: {e}")
        # Fallback to some creative defaults
        fallback_variations = [
            "\n\nFor this variation, use an extreme close-up to emphasize emotional details or key objects in the scene.",
            "\n\nFor this variation, pull back to an ultra-wide establishing shot showing the broader context.",
            "\n\nFor this variation, use a dramatic low angle (worm's eye view) to make subjects appear powerful or imposing.",
            "\n\nFor this variation, employ a bird's eye view looking straight down at the scene.",
            "\n\nFor this variation, use a dutch angle (tilted camera) to create tension or unease.",
            "\n\nFor this variation, focus on a detail or background element that adds narrative depth."
        ]
        return fallback_variations[(variation_num - 2) % len(fallback_variations)]

def build_multimodal_input(state: Dict[str, Any], style_description: str, variation_prompt: str) -> List[Dict[str, Any]]:
    """Build multimodal input for GPT-4.1-mini with image generation.
    
    Args:
        state: Current workflow state
        style_description: Extracted style description
        variation_prompt: Dynamic variation instructions
        
    Returns:
        List of input content items
    """
    shot_text = state['shot_text']
    shot_id = state['shot_id']
    variation_num = state.get('current_variation', 1)
    historical_context = state.get('historical_context', [])
    context_window = state.get('context_window', 0)
    base_context = state.get('base_context', {})
    
    content = []
    
    # Include key base documents in context
    if base_context and 'base_documents' in base_context:
        base_docs = base_context['base_documents']
        
        # Add script context (abbreviated to avoid token limits)
        if 'script' in base_docs:
            script_text = base_docs['script']
            # Find surrounding shots for context
            import re
            shot_pattern = r'### \*?\*?Shot (\d+)'
            current_shot_match = re.search(f'### \\*?\\*?Shot {shot_id}\\b', script_text)
            if current_shot_match:
                # Extract a window around the current shot
                start_pos = max(0, current_shot_match.start() - 2000)
                end_pos = min(len(script_text), current_shot_match.end() + 2000)
                script_context = script_text[start_pos:end_pos]
                
                content.append({
                    "role": "system",
                    "content": f"Script context (surrounding shots):\n{script_context}\n\nUse this context to maintain narrative continuity."
                })
    
    # Add historical context if enabled
    if context_window != 0 and historical_context:
        # Add context header
        content.append({
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Previous storyboard frames for context and style consistency:"
                }
            ]
        })
        
        # Add historical images and their prompts
        for i, hist_item in enumerate(historical_context):
            # Add the prompt that generated this image
            content[0]["content"].append({
                "type": "input_text",
                "text": f"\nShot {hist_item['shot_id']}, Variation {hist_item['variation']}:\n{hist_item['prompt_text']}"
            })
            
            # Add the generated image
            try:
                image_data_url = to_data_url(hist_item['image_path'])
                content[0]["content"].append({
                    "type": "input_image",
                    "image_url": image_data_url
                })
            except Exception as e:
                print(f"Error loading historical image {hist_item['image_path']}: {e}")
    
    # Add main prompt
    main_content = []
    
    # Add style description
    main_content.append({
        "type": "input_text",
        "text": f"""Create a 16:9 storyboard frame for Shot {shot_id}, Variation {variation_num}.

Style Requirements:
{style_description}

Shot Description:
{shot_text}

Technical Requirements:
- Aspect ratio: 16:9 (widescreen)
- Maintain character consistency based on any reference images
- Focus on the specific camera angle, framing, and composition described
- Maintain consistent style across all variations{variation_prompt}"""
    })
    
    # Add context note if using historical context
    if context_window != 0 and historical_context:
        main_content[0]["text"] += "\n\nIMPORTANT: Maintain visual consistency with the previous frames shown above."
    
    # Add entity-specific references
    for entity in state.get('entities', []):
        refs = get_refs_for(entity, k=1)  # Get 1 ref per entity
        for ref in refs:
            try:
                data_url = to_data_url(ref['path'])
                main_content.append({
                    "type": "input_image",
                    "image_url": data_url
                })
                main_content.append({
                    "type": "input_text",
                    "text": f"Use this as reference for {entity}"
                })
            except Exception as e:
                print(f"Error loading reference {ref['path']}: {e}")
    
    # Combine all content
    if content:  # If we have historical context
        content.append({
            "role": "user",
            "content": main_content
        })
    else:  # No historical context
        content = [{
            "role": "user", 
            "content": main_content
        }]
    
    return content

async def artist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Artist node for the LangGraph workflow.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with generated image path
    """
    print(f"  Artist node received state with keys: {list(state.keys())}")
    print(f"  Artist state content: {state}")
    
    shot_id = state['shot_id']
    attempt = state['attempt']
    variation_num = state.get('current_variation', 1)
    output_base_path = state.get('output_base_path', 'output')
    
    # Create output directory with timestamp path
    output_dir = Path(output_base_path) / f"Shot-{shot_id:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract style from data
    base_context = state.get('base_context', {})
    style_description = await extract_style_from_data(base_context)
    
    # Generate dynamic variation prompt
    variation_prompt = await generate_variation_prompt(state, variation_num)
    
    # Build multimodal input
    input_messages = build_multimodal_input(state, style_description, variation_prompt)
    
    # Save prompt for debugging
    prompt_path = output_dir / f"prompt_var{variation_num}_attempt{attempt}.json"
    
    # Extract the main prompt text for historical context
    prompt_text = f"{style_description}\n\n{state['shot_text']}{variation_prompt}"
    
    prompt_data = {
        "shot_id": shot_id,
        "variation": variation_num,
        "attempt": attempt,
        "style_description": style_description,
        "variation_prompt": variation_prompt,
        "entities": state.get('entities', []),
        "shot_text": state['shot_text'],
        "prompt_text": prompt_text  # Store for historical context
    }
    
    async with aiofiles.open(prompt_path, 'w') as f:
        await f.write(json.dumps(prompt_data, indent=2))
    
    print(f"  Generating image for Shot {shot_id}, variation {variation_num}, attempt {attempt} using gpt-4.1-mini...")
    
    try:
        # Call OpenAI Responses API with gpt-4.1-mini for image generation
        response = await client.responses.create(
            model="gpt-4.1-mini",
            input=input_messages,
            tools=[{"type": "image_generation"}],
        )
        
        # Extract generated images
        image_outputs = [
            output for output in response.output
            if output.type == "image_generation_call"
        ]
        
        if image_outputs:
            # Save the first generated image
            image_base64 = image_outputs[0].result
            image_path = output_dir / f"var{variation_num}_try{attempt}.png"
            
            await save_base64_image(image_base64, image_path)
            
            print(f"  Image saved to {image_path}")
            
            # Update image paths list
            image_paths = state.get('image_paths', [])
            image_paths.append(str(image_path))
            
            # Prepare historical data for this successful generation
            historical_data = {
                "shot_id": shot_id,
                "variation": variation_num,
                "image_path": str(image_path),
                "prompt_text": prompt_text,
                "variation_prompt": variation_prompt  # Store the specific variation instruction
            }
            
            # Add to historical context
            historical_context = state.get('historical_context', []).copy()
            historical_context.append(historical_data)
            
            # Update state
            return {
                **state,
                "image_path": str(image_path),
                "image_paths": image_paths,
                "status": "generated",
                "historical_context": historical_context
            }
        else:
            # No image was generated
            return {
                **state,
                "status": "error",
                "error": "No image generated in response"
            }
            
    except Exception as e:
        error_msg = str(e)
        print(f"  Error generating image: {error_msg}")
        
        # Try a simpler approach if the main one fails
        try:
            print(f"  Retrying with simplified input...")
            
            # Simplified input without references
            simple_response = await client.responses.create(
                model="gpt-4.1-mini",
                input=f"Create a 16:9 storyboard frame, variation {variation_num}: {state['shot_text'][:500]}",
                tools=[{"type": "image_generation"}],
            )
            
            image_outputs = [
                output for output in simple_response.output
                if output.type == "image_generation_call"
            ]
            
            if image_outputs:
                image_base64 = image_outputs[0].result
                image_path = output_dir / f"var{variation_num}_try{attempt}.png"
                await save_base64_image(image_base64, image_path)
                
                print(f"  Image saved to {image_path} (simplified version)")
                
                # Update image paths list
                image_paths = state.get('image_paths', [])
                image_paths.append(str(image_path))
                
                # Prepare historical data (simplified version)
                historical_data = {
                    "shot_id": shot_id,
                    "variation": variation_num,
                    "image_path": str(image_path),
                    "prompt_text": f"Simplified: {state['shot_text'][:500]}",
                    "variation_prompt": variation_prompt  # Store even for simplified
                }
                
                # Add to historical context
                historical_context = state.get('historical_context', []).copy()
                historical_context.append(historical_data)
                
                return {
                    **state,
                    "image_path": str(image_path),
                    "image_paths": image_paths,
                    "status": "generated",
                    "historical_context": historical_context
                }
        except Exception as e2:
            print(f"  Simplified approach also failed: {e2}")
            error_msg = f"Original: {error_msg}, Simplified: {str(e2)}"
        
        return {
            **state,
            "status": "error",
            "error": error_msg
        }

# For testing
if __name__ == "__main__":
    # Test style extraction
    async def test():
        style = await extract_style_from_data()
        print("Extracted style:")
        print(style)
    
    asyncio.run(test()) 