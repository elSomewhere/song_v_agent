"""
Vision QA agent - validates generated images against scene requirements.
"""
from typing import Dict, Any, List, Optional, Union
from openai import AsyncOpenAI
import json
import os
import base64
from pathlib import Path

from memory.schemas import ScenePlan, VisionQAResult, EntityState, CameraSpec, EnvironmentSpec


class VisionQAAgent:
    """Evaluates generated images for quality and accuracy."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    async def evaluate_image(
        self,
        scene_plan: Union[ScenePlan, Dict[str, Any]],
        image_path: str,
        attempt_number: int,
        previous_feedback: List[Dict[str, Any]],
        quality_threshold: float = 0.7
    ) -> VisionQAResult:
        """Evaluate a generated image against the scene plan."""
        
        # Convert dict to ScenePlan if needed
        if isinstance(scene_plan, dict):
            # Handle nested dicts for complex fields
            if 'entities' in scene_plan:
                entities = []
                for e in scene_plan['entities']:
                    if isinstance(e, dict):
                        entities.append(EntityState(**e))
                    else:
                        entities.append(e)
                scene_plan['entities'] = entities
            
            if 'camera' in scene_plan and isinstance(scene_plan['camera'], dict):
                scene_plan['camera'] = CameraSpec(**scene_plan['camera'])
            
            if 'environment' in scene_plan and isinstance(scene_plan['environment'], dict):
                scene_plan['environment'] = EnvironmentSpec(**scene_plan['environment'])
            
            scene_plan = ScenePlan(**scene_plan)
        
        # Load and encode the image
        image_base64 = self._encode_image(image_path)
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            scene_plan,
            attempt_number,
            previous_feedback
        )
        
        try:
            # Use GPT-4 Vision to evaluate
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a film continuity supervisor evaluating storyboard frames. Provide detailed, constructive feedback in JSON format."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse the evaluation
            evaluation = json.loads(response.choices[0].message.content)
            
            # Ensure all required fields
            evaluation = self._validate_evaluation(evaluation)
            
            # Determine status based on quality score and threshold
            quality_score = evaluation.get("quality_score", 0.5)
            
            if quality_score >= quality_threshold:
                status = "pass"
            elif attempt_number < 3 and quality_score >= quality_threshold * 0.7:
                status = "retry"
            else:
                status = "fail"
            
            evaluation["status"] = status
            
            return VisionQAResult(**evaluation)
            
        except Exception as e:
            print(f"Error in vision QA: {e}")
            # Return a conservative evaluation
            return VisionQAResult(
                status="retry" if attempt_number < 3 else "fail",
                quality_score=0.5,
                feedback={
                    "composition": "Unable to evaluate",
                    "entity_accuracy": "Unable to evaluate",
                    "continuity": "Unable to evaluate",
                    "technical_quality": "Evaluation error occurred"
                },
                specific_issues=[f"Evaluation error: {str(e)}"],
                retry_guidance="Try regenerating with clearer specifications" if attempt_number < 3 else None
            )
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _build_evaluation_prompt(
        self,
        scene_plan: ScenePlan,
        attempt_number: int,
        previous_feedback: List[Dict[str, Any]]
    ) -> str:
        """Build the evaluation prompt."""
        
        # Format scene requirements
        requirements = self._format_scene_requirements(scene_plan)
        
        # Format previous feedback if any
        feedback_context = self._format_previous_feedback(previous_feedback)
        
        prompt = f"""Evaluate this storyboard frame against the scene requirements.

SCENE REQUIREMENTS:
{requirements}

{feedback_context}

Evaluate the image for:
1. **Composition**: Overall visual composition, framing, and artistic quality
2. **Entity Accuracy**: Are all required characters present with correct poses and emotions?
3. **Continuity**: Does it maintain visual consistency with the established style?
4. **Technical Quality**: Image clarity, lighting, and professional quality

Provide your evaluation as a JSON object:
{{
    "quality_score": 0.0-1.0 (overall quality rating),
    "feedback": {{
        "composition": "detailed feedback on composition",
        "entity_accuracy": "detailed feedback on character accuracy",
        "continuity": "detailed feedback on visual continuity",
        "technical_quality": "detailed feedback on technical aspects"
    }},
    "specific_issues": ["list of specific issues found"],
    "retry_guidance": "specific guidance for improvement if retry is needed"
}}

Be constructive but thorough. This is attempt {attempt_number}."""
        
        return prompt
    
    def _format_scene_requirements(self, scene_plan: ScenePlan) -> str:
        """Format scene requirements for the prompt."""
        
        parts = []
        
        # Entities
        if scene_plan.entities:
            entity_desc = []
            for entity in scene_plan.entities:
                if hasattr(entity, 'name'):
                    desc = f"- {entity.name}: {entity.pose}, {entity.emotion} emotion, positioned {entity.position}"
                elif isinstance(entity, dict):
                    desc = f"- {entity.get('name', 'Unknown')}: {entity.get('pose', 'unspecified pose')}"
                else:
                    continue
                entity_desc.append(desc)
            
            if entity_desc:
                parts.append("Required Characters:\n" + "\n".join(entity_desc))
        
        # Camera
        camera = scene_plan.camera
        if hasattr(camera, 'type'):
            parts.append(f"Camera: {camera.type} at {camera.angle}, {camera.distance} distance")
        elif isinstance(camera, dict):
            parts.append(f"Camera: {camera.get('type', 'unspecified shot')}")
        
        # Environment
        env = scene_plan.environment
        if hasattr(env, 'location'):
            parts.append(f"Environment: {env.location}, {env.lighting} lighting")
        elif isinstance(env, dict):
            parts.append(f"Environment: {env.get('location', 'unspecified')}")
        
        # Mood and focus
        parts.append(f"Mood: {scene_plan.mood}")
        parts.append(f"Visual Focus: {scene_plan.visual_focus}")
        
        # Action beats
        if scene_plan.action_beats:
            parts.append(f"Key Actions: {', '.join(scene_plan.action_beats)}")
        
        return "\n".join(parts)
    
    def _format_previous_feedback(self, previous_feedback: List[Dict[str, Any]]) -> str:
        """Format previous feedback for context."""
        
        if not previous_feedback:
            return ""
        
        feedback_parts = ["PREVIOUS FEEDBACK TO ADDRESS:"]
        
        for i, feedback in enumerate(previous_feedback[-2:], 1):  # Last 2 attempts
            if "specific_issues" in feedback:
                issues = feedback["specific_issues"]
                if isinstance(issues, list):
                    feedback_parts.append(f"Attempt {i} issues: {', '.join(issues)}")
        
        return "\n".join(feedback_parts) if len(feedback_parts) > 1 else ""
    
    def _validate_evaluation(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and complete the evaluation."""
        
        # Ensure quality score
        if "quality_score" not in evaluation:
            evaluation["quality_score"] = 0.5
        else:
            # Clamp to valid range
            evaluation["quality_score"] = max(0.0, min(1.0, float(evaluation["quality_score"])))
        
        # Ensure feedback dict
        if "feedback" not in evaluation:
            evaluation["feedback"] = {}
        
        feedback = evaluation["feedback"]
        feedback.setdefault("composition", "No composition feedback provided")
        feedback.setdefault("entity_accuracy", "No entity accuracy feedback provided")
        feedback.setdefault("continuity", "No continuity feedback provided")
        feedback.setdefault("technical_quality", "No technical quality feedback provided")
        
        # Ensure other fields
        evaluation.setdefault("specific_issues", [])
        evaluation.setdefault("retry_guidance", None)
        
        return evaluation 