"""
Planner Agent for AI Storyboard Generator
Responsible for scene analysis, entity detection, and creating structured scene plans
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from memory import (
    MemoryService, 
    ScenePlan, 
    EntitySpec, 
    CameraSpec, 
    EnvironmentSpec,
    ContextRef,
    RAGConfig,
    RetrievalParams
)


class PlannerAgent:
    """
    Planner agent that:
    1. Parses scene descriptions
    2. Detects entities using canonical memory
    3. Retrieves relevant episodic memories
    4. Generates structured scene plans
    """
    
    def __init__(self, 
                 memory_service: MemoryService,
                 config: RAGConfig,
                 data_path: str = "./data"):
        self.memory = memory_service
        self.config = config
        self.data_path = Path(data_path)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Load canonical memories
        self._load_canonical_context()
        
        # JSON output parser
        self.output_parser = JsonOutputParser(pydantic_object=ScenePlan)
    
    def _load_canonical_context(self):
        """Load canonical context from memory service"""
        # Get style guide
        style_memory = self.memory.canonical.get_by_id("style_guide")
        self.style_guide = style_memory.content if style_memory else ""
        
        # Get entities document
        entities_memory = self.memory.canonical.get_by_id("entities_doc")
        self.entities_doc = entities_memory.content if entities_memory else ""
        
        # Get full script
        script_memory = self.memory.canonical.get_by_id("full_script")
        self.full_script = script_memory.content if script_memory else ""
        
        # Extract entity information
        self.entity_map = self._extract_entities_from_doc()
    
    def _extract_entities_from_doc(self) -> Dict[str, Dict[str, Any]]:
        """Extract entities and their aliases from entities document"""
        entity_map = {}
        
        if not self.entities_doc:
            return entity_map
        
        # Use GPT to extract structured entity information
        extraction_prompt = """
        Extract all entities (characters, objects, environments) from the following document.
        For each entity, identify:
        1. Primary name
        2. All aliases/alternative names
        3. Type (character, object, environment)
        4. Brief description
        
        Return as JSON in format:
        {
            "entities": [
                {
                    "name": "primary name",
                    "aliases": ["alias1", "alias2"],
                    "type": "character|object|environment",
                    "description": "brief description"
                }
            ]
        }
        """
        
        messages = [
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=self.entities_doc)
        ]
        
        try:
            response = self.llm.invoke(messages)
            entities_data = json.loads(response.content)
            
            # Build entity map
            for entity in entities_data.get("entities", []):
                name = entity["name"]
                entity_map[name.lower()] = entity
                
                # Also map aliases
                for alias in entity.get("aliases", []):
                    entity_map[alias.lower()] = entity
                    
        except Exception as e:
            print(f"Error extracting entities: {e}")
        
        return entity_map
    
    def _detect_entities_in_text(self, text: str) -> List[str]:
        """Detect which entities are mentioned in the text"""
        detected = set()
        text_lower = text.lower()
        
        # Check each entity and alias
        for key, entity_data in self.entity_map.items():
            if key in text_lower:
                detected.add(entity_data["name"])  # Always use primary name
        
        return list(detected)
    
    def _get_entity_references(self, entity_name: str) -> List[str]:
        """Get reference image IDs for an entity"""
        ref_ids = []
        
        # Check for reference images in data/refs/entity_name/
        entity_refs_path = self.data_path / "refs" / entity_name
        if entity_refs_path.exists():
            # Get all image files
            for img_file in entity_refs_path.glob("*.png"):
                ref_ids.append(img_file.stem)
            for img_file in entity_refs_path.glob("*.jpg"):
                ref_ids.append(img_file.stem)
        
        return ref_ids[:5]  # Limit to 5 references
    
    def _parse_camera_info(self, shot_text: str) -> Tuple[CameraSpec, EnvironmentSpec]:
        """Extract camera and environment info from shot description"""
        # Default values
        camera_type = "medium shot"
        camera_angle = "eye level"
        camera_movement = None
        
        environment_setting = "unspecified"
        lighting = "natural"
        mood = "neutral"
        
        # Extract framing info
        framing_match = re.search(r'Framing:\s*([^\n]+)', shot_text, re.IGNORECASE)
        if framing_match:
            framing_text = framing_match.group(1).lower()
            
            # Detect shot type
            if any(term in framing_text for term in ["wide", "establishing"]):
                camera_type = "wide shot"
            elif any(term in framing_text for term in ["close", "tight"]):
                camera_type = "close up"
            elif "medium" in framing_text:
                camera_type = "medium shot"
            
            # Detect angle
            if any(term in framing_text for term in ["low angle", "from below"]):
                camera_angle = "low angle"
            elif any(term in framing_text for term in ["high angle", "from above"]):
                camera_angle = "high angle"
            elif "dutch" in framing_text:
                camera_angle = "dutch angle"
        
        # Extract camera angle info
        angle_match = re.search(r'Camera Angle:\s*([^\n]+)', shot_text, re.IGNORECASE)
        if angle_match:
            angle_text = angle_match.group(1).lower()
            if "tracking" in angle_text:
                camera_movement = "tracking shot"
            elif "dolly" in angle_text:
                camera_movement = "dolly"
            elif "pan" in angle_text:
                camera_movement = "pan"
        
        # Use GPT to extract environment info from description
        desc_match = re.search(r'Description:\s*([^\n]+(?:\n(?!Framing:|Camera Angle:)[^\n]+)*)', 
                              shot_text, re.IGNORECASE)
        if desc_match:
            description = desc_match.group(1)
            
            # Simple keyword extraction for now
            if any(term in description.lower() for term in ["night", "dark", "shadow"]):
                lighting = "low key"
            elif any(term in description.lower() for term in ["bright", "day", "sun"]):
                lighting = "high key"
            
            if any(term in description.lower() for term in ["tense", "danger", "fear"]):
                mood = "tense"
            elif any(term in description.lower() for term in ["calm", "peace", "serene"]):
                mood = "peaceful"
            elif any(term in description.lower() for term in ["action", "explosive", "chaos"]):
                mood = "chaotic"
        
        camera_spec = CameraSpec(
            type=camera_type,
            angle=camera_angle,
            movement=camera_movement
        )
        
        environment_spec = EnvironmentSpec(
            setting=environment_setting,
            lighting=lighting,
            mood=mood
        )
        
        return camera_spec, environment_spec
    
    def _extract_visual_beats(self, description: str) -> List[str]:
        """Extract key visual elements/beats from description"""
        visual_beats = []
        
        # Use GPT to extract visual beats
        extraction_prompt = """
        Extract 2-4 key visual elements or moments from this shot description.
        Focus on:
        - Specific visual details
        - Actions or movements
        - Environmental elements
        - Visual effects
        
        Return as a simple JSON list of strings.
        """
        
        messages = [
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=description)
        ]
        
        try:
            response = self.llm.invoke(messages)
            beats_data = json.loads(response.content)
            if isinstance(beats_data, list):
                visual_beats = beats_data[:4]  # Limit to 4 beats
        except:
            # Fallback to simple extraction
            if "explosion" in description.lower():
                visual_beats.append("explosion effect")
            if "debris" in description.lower():
                visual_beats.append("debris particles")
        
        return visual_beats
    
    def plan_scene(self, 
                   scene_id: int, 
                   shot_description: str) -> ScenePlan:
        """
        Generate a complete scene plan for the given shot
        """
        # Detect entities in the shot
        detected_entities = self._detect_entities_in_text(shot_description)
        
        # Retrieve relevant memories
        retrieval_params = RetrievalParams(
            k=self.config.retrieval_k,
            semantic_weight=self.config.semantic_weight,
            entity_weight=self.config.entity_weight,
            distance_weight=self.config.distance_weight
        )
        
        retrieved_memories = self.memory.retrieve_for_scene(
            current_scene_id=scene_id,
            scene_description=shot_description,
            entities=detected_entities,
            params=retrieval_params
        )
        
        # Parse camera and environment info
        camera_spec, environment_spec = self._parse_camera_info(shot_description)
        
        # Extract visual beats
        visual_beats = self._extract_visual_beats(shot_description)
        
        # Build entity specifications
        entity_specs = []
        for entity_name in detected_entities:
            # Get entity data
            entity_data = self.entity_map.get(entity_name.lower(), {})
            
            # Use GPT to determine pose and emotion for this scene
            entity_prompt = f"""
            Given this shot description:
            {shot_description}
            
            For the character/entity "{entity_name}", determine:
            1. Their pose/position in the scene
            2. Their emotional state
            
            Return as JSON: {{"pose": "...", "emotion": "..."}}
            """
            
            messages = [
                SystemMessage(content=entity_prompt),
                HumanMessage(content="Analyze and respond with JSON only.")
            ]
            
            try:
                response = self.llm.invoke(messages)
                entity_state = json.loads(response.content)
                pose = entity_state.get("pose", "standing")
                emotion = entity_state.get("emotion", "neutral")
            except:
                pose = "present in scene"
                emotion = "as described"
            
            # Get visual references
            ref_ids = self._get_entity_references(entity_name)
            
            entity_specs.append(EntitySpec(
                name=entity_name,
                canonical_id=f"CHAR_{entity_name.upper().replace(' ', '_')}",
                pose=pose,
                emotion=emotion,
                visual_ref_ids=ref_ids
            ))
        
        # Build context references
        context_refs = []
        for mem_data in retrieved_memories:
            memory = mem_data['memory']
            context_refs.append(ContextRef(
                scene_id=memory.scene_id,
                relevance=mem_data['score'],
                summary=memory.shot_description[:100] + "..."
            ))
        
        # Enhance environment spec with GPT analysis
        env_prompt = f"""
        Analyze this shot and provide environment details:
        {shot_description}
        
        Provide:
        1. Setting/location
        2. Lighting conditions
        3. Overall mood
        
        Return as JSON: {{"setting": "...", "lighting": "...", "mood": "..."}}
        """
        
        messages = [
            SystemMessage(content=env_prompt),
            HumanMessage(content="Analyze and respond with JSON only.")
        ]
        
        try:
            response = self.llm.invoke(messages)
            env_data = json.loads(response.content)
            environment_spec.setting = env_data.get("setting", environment_spec.setting)
            environment_spec.lighting = env_data.get("lighting", environment_spec.lighting)
            environment_spec.mood = env_data.get("mood", environment_spec.mood)
        except:
            pass
        
        # Create scene plan
        scene_plan = ScenePlan(
            scene_id=scene_id,
            entities=entity_specs,
            camera=camera_spec,
            environment=environment_spec,
            visual_beats=visual_beats,
            retrieved_context=context_refs,
            raw_description=shot_description
        )
        
        return scene_plan 