"""
Critic Agent - Evaluates generated storyboard frames using GPT-4o vision.
Checks adherence to style, shot requirements, and character consistency.
"""
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import aiofiles
from agents.shared import (
    load_style_primer, 
    to_data_url, 
    get_refs_for,
    get_similar_shots,
    load_style_guidelines,
    load_entities_document
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MAX_RETRIES = 1  # Maximum retry attempts (default)

# Initialize async OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def extract_style_guidelines(base_context: Dict[str, Any] = None) -> str:
    """Extract style guidelines from base context or data folder dynamically.
    
    Args:
        base_context: Base context containing documents
    
    Returns:
        Style guidelines as a string
    """
    guidelines = []
    
    # If base context is provided, use it
    if base_context and 'base_documents' in base_context:
        base_docs = base_context['base_documents']
        
        # Use style guidelines from base context
        if 'style_guidelines' in base_docs:
            guidelines.append(base_docs['style_guidelines'])
        
        # Use entities document from base context
        if 'entities_document' in base_docs:
            entities_doc = base_docs['entities_document']
            # Extract key visual requirements
            if "Color Palette" in entities_doc:
                color_section = entities_doc.split("Color Palette")[1].split("##")[0]
                guidelines.append(f"Color palette requirements: {color_section[:500]}")
    else:
        # Fallback to loading directly
        # Load style guidelines from style.md
        try:
            style_guidelines = load_style_guidelines()
            guidelines.append(style_guidelines)
        except:
            pass
        
        # Load entities document for additional visual guidance
        try:
            entities_doc = load_entities_document()
            # Extract key visual requirements
            if "Color Palette" in entities_doc:
                color_section = entities_doc.split("Color Palette")[1].split("##")[0]
                guidelines.append(f"Color palette requirements: {color_section[:500]}")
        except:
            pass
    
    # Analyze reference images for style
    refs_path = Path("data/refs")
    if refs_path.exists():
        # Collect sample reference images
        ref_samples = []
        for subdir in refs_path.iterdir():
            if subdir.is_dir():
                for img_file in list(subdir.glob("*.png"))[:1] + list(subdir.glob("*.jpg"))[:1]:
                    if img_file.exists() and len(ref_samples) < 3:
                        ref_samples.append(str(img_file))
        
        if ref_samples:
            try:
                # Use GPT-4 vision to extract style guidelines
                content = [
                    {
                        "type": "text",
                        "text": "Analyze these reference images and list the key visual style characteristics that should be maintained in storyboard frames. Focus on: art style, color palette, line work, shading technique, and overall aesthetic."
                    }
                ]
                
                for img_path in ref_samples:
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
                    max_tokens=250,
                    temperature=0.3
                )
                
                style_analysis = response.choices[0].message.content
                guidelines.append(f"Reference style analysis: {style_analysis}")
            except Exception as e:
                print(f"Error extracting style guidelines: {e}")
    
    # If no guidelines found, use generic ones
    if not guidelines:
        guidelines.append("Maintain consistent visual style across all frames")
    
    return "\n\n".join(guidelines)

def build_critic_messages(state: Dict[str, Any], image_path: str, style_guidelines: str) -> List[Dict[str, Any]]:
    """Build messages for the critic evaluation.
    
    Args:
        state: Current workflow state
        image_path: Path to the generated image
        style_guidelines: Extracted style guidelines
        
    Returns:
        List of message dictionaries for GPT-4o
    """
    messages = []
    
    # Get historical context
    historical_context = state.get('historical_context', [])
    context_window = state.get('context_window', 0)
    base_context = state.get('base_context', {})
    
    # System message - update to include continuity requirements
    system_content = f"""You are an expert storyboard critic evaluating generated frames.

Style Guidelines:
{style_guidelines}"""

    # Add script context if available
    if base_context and 'base_documents' in base_context:
        base_docs = base_context['base_documents']
        if 'script' in base_docs:
            # Add a note about the full script being available
            system_content += "\n\nNote: You have access to the full script for narrative context."

    system_content += """

Your task is to evaluate if the generated image:
1. Matches the shot description accurately
2. Follows the established visual style from the references
3. Maintains character consistency with reference images
4. Has proper 16:9 composition
5. Uses appropriate camera angles and framing as described"""

    if context_window != 0 and historical_context:
        system_content += """
6. Maintains visual continuity with previous frames in the sequence
7. Keeps consistent character designs, color palettes, and style across shots"""

    system_content += """

Respond with JSON only:
{"status": "PASS" or "RETRY", "notes": "Brief explanation"}"""

    messages.append({
        "role": "system",
        "content": system_content
    })
    
    # Build multimodal content
    content = []
    
    # Add historical context if enabled
    if context_window != 0 and historical_context:
        content.append({
            "type": "text",
            "text": "Previous frames for continuity reference:"
        })
        
        # Show previous frames
        for hist_item in historical_context:
            try:
                hist_image_url = to_data_url(hist_item['image_path'])
                content.append({
                    "type": "text",
                    "text": f"\nShot {hist_item['shot_id']}, Variation {hist_item['variation']}:"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": hist_image_url}
                })
            except Exception as e:
                print(f"Error loading historical image {hist_item['image_path']}: {e}")
        
        content.append({
            "type": "text",
            "text": "\n--- Current Frame to Evaluate ---\n"
        })
    
    # Add shot description
    content.append({
        "type": "text",
        "text": f"Shot {state['shot_id']} Description:\n{state['shot_text']}"
    })
    
    # Add reference images for entities
    if state.get('entities'):
        content.append({
            "type": "text",
            "text": "\nCharacter/Entity References:"
        })
        
        for entity in state.get('entities', []):
            refs = get_refs_for(entity, k=1)
            for ref in refs:
                try:
                    data_url = to_data_url(ref['path'])
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
                    content.append({
                        "type": "text",
                        "text": f"Reference for {entity}"
                    })
                except Exception as e:
                    print(f"Error loading reference {ref['path']}: {e}")
    
    # Add the generated image
    content.append({
        "type": "text",
        "text": "\nGenerated Storyboard Frame:"
    })
    
    try:
        image_data_url = to_data_url(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": image_data_url}
        })
    except Exception as e:
        print(f"Error loading generated image {image_path}: {e}")
        return messages
    
    evaluation_prompt = "\nEvaluate this image against the shot description and style guidelines."
    if context_window != 0 and historical_context:
        evaluation_prompt += " Pay special attention to visual continuity with the previous frames shown above."
    evaluation_prompt += " Respond with JSON."
    
    content.append({
        "type": "text",
        "text": evaluation_prompt
    })
    
    messages.append({
        "role": "user",
        "content": content
    })
    
    return messages

async def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Critic node for the LangGraph workflow.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with critic verdict
    """
    shot_id = state['shot_id']
    attempt = state['attempt']
    variation_num = state.get('current_variation', 1)
    image_path = state.get('image_path')
    max_retries = state.get('max_retries', MAX_RETRIES)  # Use from state, fallback to default
    
    if not image_path or not Path(image_path).exists():
        print(f"  Error: Image not found at {image_path}")
        return {
            **state,
            "status": "error",
            "error": "Generated image not found"
        }
    
    print(f"  Evaluating Shot {shot_id}, variation {variation_num}, attempt {attempt}...")
    
    # Extract style guidelines
    base_context = state.get('base_context', {})
    style_guidelines = await extract_style_guidelines(base_context)
    
    # Build messages for GPT-4o
    messages = build_critic_messages(state, image_path, style_guidelines)
    
    try:
        # Call GPT-4o vision with async client
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini for cost efficiency
            messages=messages,
            max_tokens=200,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        verdict_text = response.choices[0].message.content
        verdict = json.loads(verdict_text)
        
        # Save critic feedback
        output_dir = Path(image_path).parent
        critic_path = output_dir / f"critic_var{variation_num}_try{attempt}.json"
        
        critic_data = {
            "shot_id": shot_id,
            "variation": variation_num,
            "attempt": attempt,
            "verdict": verdict,
            "image_path": image_path,
            "style_guidelines": style_guidelines
        }
        
        async with aiofiles.open(critic_path, 'w') as f:
            await f.write(json.dumps(critic_data, indent=2))
        
        print(f"  Critic verdict: {verdict['status']} - {verdict.get('notes', '')}")
        
        # Update state based on verdict
        if verdict['status'] == 'PASS':
            # Check if we need more variations
            variations_per_shot = state.get('variations_per_shot', 1)
            if variation_num < variations_per_shot:
                # Prepare for next variation
                return {
                    **state,
                    "status": "pass",
                    "current_variation": variation_num + 1,
                    "attempt": 0  # Reset attempt for new variation
                }
            else:
                # All variations done
                return {
                    **state,
                    "status": "pass",
                    "current_variation": 1,  # Reset for next shot
                    "attempt": 0
                }
        elif verdict['status'] == 'RETRY' and attempt < max_retries:
            return {
                **state,
                "status": "retry",
                "attempt": attempt + 1,
                "critic_notes": verdict.get('notes', '')
            }
        else:
            # Too many retries or FAIL status
            variations_per_shot = state.get('variations_per_shot', 1)
            if variation_num < variations_per_shot:
                # Try next variation
                return {
                    **state,
                    "status": "fail",
                    "current_variation": variation_num + 1,
                    "attempt": 0,
                    "critic_notes": verdict.get('notes', 'Max retries exceeded')
                }
            else:
                # All variations attempted
                return {
                    **state,
                    "status": "fail",
                    "current_variation": 1,
                    "attempt": 0,
                    "critic_notes": verdict.get('notes', 'Max retries exceeded')
                }
            
    except Exception as e:
        print(f"  Error in critic evaluation: {e}")
        return {
            **state,
            "status": "error",
            "error": str(e)
        }

# For testing
if __name__ == "__main__":
    # Test style extraction
    async def test():
        guidelines = await extract_style_guidelines()
        print("Extracted style guidelines:")
        print(guidelines)
    
    asyncio.run(test()) 