"""
Vision QA Agent for AI Storyboard Generator
Responsible for evaluating generated images for quality and adherence
"""

import json
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path

import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from memory import (
    MemoryService,
    ScenePlan,
    GeneratedImage,
    VisionQAResult,
    RAGConfig
)


class VisionQAAgent:
    """
    Vision QA agent that:
    1. Analyzes generated images against scene plans
    2. Checks style consistency
    3. Verifies entity accuracy
    4. Provides structured feedback
    """
    
    def __init__(self,
                 memory_service: MemoryService,
                 config: RAGConfig):
        self.memory = memory_service
        self.config = config
        
        # Initialize vision-capable LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",  # GPT-4o has vision capabilities
            temperature=0.3,  # Lower temperature for more consistent evaluation
            max_tokens=1000
        )
        
        # Load evaluation criteria
        self._load_evaluation_criteria()
    
    def _load_evaluation_criteria(self):
        """Load evaluation criteria from style guide and config"""
        # Get style guide
        style_memory = self.memory.canonical.get_by_id("style_guide")
        self.style_guide = style_memory.content if style_memory else ""
        
        # Define evaluation rubric
        self.evaluation_rubric = {
            "composition_match": {
                "weight": 0.3,
                "criteria": [
                    "Camera angle matches specification",
                    "Shot type (wide/medium/close) is correct",
                    "Framing follows cinematic principles",
                    "16:9 aspect ratio is maintained"
                ]
            },
            "style_adherence": {
                "weight": 0.25,
                "criteria": [
                    "Visual style matches the style guide",
                    "Consistent artistic treatment",
                    "Appropriate mood and atmosphere",
                    "Professional storyboard quality"
                ]
            },
            "entity_accuracy": {
                "weight": 0.3,
                "criteria": [
                    "Characters match their descriptions",
                    "Poses and emotions are accurately depicted",
                    "Visual consistency with references",
                    "All specified entities are present"
                ]
            },
            "narrative_coherence": {
                "weight": 0.15,
                "criteria": [
                    "Scene tells the intended story",
                    "Visual beats are clearly shown",
                    "Continuity with retrieved context",
                    "Environmental details match description"
                ]
            }
        }
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for GPT-4o vision"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _build_evaluation_prompt(self, 
                                scene_plan: ScenePlan,
                                generated_image: GeneratedImage) -> str:
        """Build evaluation prompt for GPT-4o vision"""
        
        prompt = f"""
You are a professional storyboard supervisor evaluating a generated storyboard frame.

SCENE SPECIFICATIONS:
- Shot ID: {scene_plan.scene_id}
- Camera: {scene_plan.camera.type} at {scene_plan.camera.angle}
- Setting: {scene_plan.environment.setting}
- Lighting: {scene_plan.environment.lighting}
- Mood: {scene_plan.environment.mood}

REQUIRED ENTITIES:
"""
        
        for entity in scene_plan.entities:
            prompt += f"- {entity.name}: {entity.pose}, {entity.emotion}\n"
        
        prompt += f"""
VISUAL BEATS TO INCLUDE:
{chr(10).join(f"- {beat}" for beat in scene_plan.visual_beats)}

STYLE REQUIREMENTS:
{self._extract_style_requirements()}

Please evaluate the image against these specifications and provide a JSON response with:
1. "status": "pass" or "fail"
2. "scores": {{"composition_match": 0-1, "style_adherence": 0-1, "entity_accuracy": 0-1, "narrative_coherence": 0-1}}
3. "issues": [list of specific issues found]
4. "retry_guidance": specific instructions if the image needs to be regenerated

Be strict but fair. Only pass images that meet professional storyboard standards.
"""
        
        return prompt
    
    def _extract_style_requirements(self) -> str:
        """Extract key style requirements from style guide"""
        if not self.style_guide:
            return "Professional cinematic storyboard style"
        
        # Extract first few lines or key points
        lines = self.style_guide.split('\n')
        key_lines = []
        
        for line in lines[:10]:  # First 10 lines
            if line.strip() and not line.startswith('#'):
                key_lines.append(line.strip())
        
        return '\n'.join(key_lines[:5])  # Top 5 requirements
    
    def _parse_evaluation_response(self, response_text: str) -> VisionQAResult:
        """Parse GPT-4o evaluation response into VisionQAResult"""
        try:
            # Extract JSON from response
            data = json.loads(response_text)
            
            # Create VisionQAResult
            result = VisionQAResult(
                status=data.get("status", "fail"),
                scores=data.get("scores", {}),
                issues=data.get("issues", []),
                retry_guidance=data.get("retry_guidance")
            )
            
            # Ensure all required scores are present
            for score_key in ["composition_match", "style_adherence", 
                            "entity_accuracy", "narrative_coherence"]:
                if score_key not in result.scores:
                    result.scores[score_key] = 0.0
            
            return result
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return VisionQAResult(
                status="fail",
                scores={
                    "composition_match": 0.0,
                    "style_adherence": 0.0,
                    "entity_accuracy": 0.0,
                    "narrative_coherence": 0.0
                },
                issues=[{"type": "parse_error", "description": "Failed to parse evaluation"}],
                retry_guidance="Please regenerate following the specifications more closely"
            )
    
    def evaluate_image(self,
                      scene_plan: ScenePlan,
                      generated_image: GeneratedImage) -> VisionQAResult:
        """
        Evaluate a generated image against its scene plan
        """
        
        if not generated_image.path or not Path(generated_image.path).exists():
            # Handle missing image
            return VisionQAResult(
                status="fail",
                scores={
                    "composition_match": 0.0,
                    "style_adherence": 0.0,
                    "entity_accuracy": 0.0,
                    "narrative_coherence": 0.0
                },
                issues=[{"type": "missing_image", "description": "Image file not found"}],
                retry_guidance="Image generation failed, please retry"
            )
        
        # Encode image
        image_base64 = self._encode_image(generated_image.path)
        
        # Build evaluation prompt
        eval_prompt = self._build_evaluation_prompt(scene_plan, generated_image)
        
        # Create messages for GPT-4o vision
        messages = [
            {
                "role": "system",
                "content": "You are a professional storyboard supervisor. Evaluate images strictly against specifications and provide JSON responses only."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": eval_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        try:
            # Get evaluation from GPT-4o
            response = self.llm.invoke(messages)
            
            # Parse response
            result = self._parse_evaluation_response(response.content)
            
            # Apply quality thresholds
            if result.overall_score < self.config.min_overall_qa_score:
                result.status = "fail"
                if not result.retry_guidance:
                    result.retry_guidance = f"Overall quality score {result.overall_score:.2f} is below threshold {self.config.min_overall_qa_score}"
            
            # Check individual scores
            for score_name, score_value in result.scores.items():
                if score_value < self.config.min_individual_qa_score:
                    result.status = "fail"
                    result.issues.append({
                        "type": "low_score",
                        "description": f"{score_name} score {score_value:.2f} is below threshold"
                    })
            
            # Save evaluation result
            self._save_evaluation(scene_plan.scene_id, generated_image.variation_id, 
                                generated_image.retry_count, result)
            
            return result
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            
            # Return error result
            return VisionQAResult(
                status="fail",
                scores={
                    "composition_match": 0.0,
                    "style_adherence": 0.0,
                    "entity_accuracy": 0.0,
                    "narrative_coherence": 0.0
                },
                issues=[{"type": "evaluation_error", "description": str(e)}],
                retry_guidance="Evaluation failed, please regenerate"
            )
    
    def _save_evaluation(self, 
                        scene_id: int,
                        variation_id: int,
                        retry_count: int,
                        result: VisionQAResult):
        """Save evaluation result to file"""
        output_dir = Path("output") / f"Shot-{scene_id:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"critic_var{variation_id + 1}_try{retry_count}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.dict(), f, indent=2)
    
    def get_retry_guidance(self, result: VisionQAResult) -> Optional[str]:
        """Extract specific retry guidance from evaluation result"""
        if result.passed:
            return None
        
        # Build comprehensive retry guidance
        guidance_parts = []
        
        # Add specific score-based guidance
        if result.scores["composition_match"] < 0.7:
            guidance_parts.append("Fix camera angle and framing to match specifications")
        
        if result.scores["style_adherence"] < 0.7:
            guidance_parts.append("Ensure visual style matches the gritty anime/manga aesthetic")
        
        if result.scores["entity_accuracy"] < 0.7:
            guidance_parts.append("Characters must match their descriptions and reference images exactly")
        
        if result.scores["narrative_coherence"] < 0.7:
            guidance_parts.append("Include all specified visual beats and environmental details")
        
        # Add issue-specific guidance
        for issue in result.issues[:3]:  # Top 3 issues
            if issue.get("description"):
                guidance_parts.append(issue["description"])
        
        # Add custom retry guidance if provided
        if result.retry_guidance:
            guidance_parts.insert(0, result.retry_guidance)
        
        # Combine and limit length
        combined_guidance = ". ".join(guidance_parts[:3])  # Max 3 points
        
        return combined_guidance if combined_guidance else "Please follow specifications more closely" 