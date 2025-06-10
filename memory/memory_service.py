"""
Memory Service for the AI Storyboard Generator
Implements three-tier memory architecture: Canonical, Episodic, and Working Memory
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path

import lancedb
from pydantic import BaseModel, Field
import openai


# Use regular BaseModel instead of LanceModel for Pydantic v1 compatibility
class CanonicalMemoryItem(BaseModel):
    """Schema for canonical memory items in LanceDB"""
    id: str
    type: str  # "style", "entity", "script"
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EpisodicMemoryItem(BaseModel):
    """Schema for episodic memory items in LanceDB"""
    scene_id: int
    shot_description: str
    entities_present: List[str]
    entity_states: Dict[str, Dict[str, Any]]  # Serialized EntityState objects
    camera_spec: Dict[str, Any]  # Serialized CameraSpec
    environment_spec: Dict[str, Any]  # Serialized EnvironmentSpec
    visual_elements: List[str]
    embedding: List[float]
    generated_images: List[Dict[str, Any]]  # Serialized ImageRef objects
    critic_tags: List[str]
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class EntityState:
    """State of an entity in a scene"""
    name: str
    pose: str
    emotion: str
    visual_ref_ids: List[str]


@dataclass
class CameraSpec:
    """Camera specification for a scene"""
    type: str
    angle: str
    movement: Optional[str] = None
    fov: Optional[int] = None


@dataclass
class EnvironmentSpec:
    """Environment specification for a scene"""
    setting: str
    lighting: str
    mood: str


@dataclass
class ImageRef:
    """Reference to a generated image"""
    path: str
    variation_id: int
    retry_count: int
    quality_score: Optional[float] = None


class RetrievalParams(BaseModel):
    """Parameters for memory retrieval"""
    k: int = 5
    semantic_weight: float = 0.6
    entity_weight: float = 0.3
    distance_weight: float = 0.1
    min_similarity: float = 0.5
    entity_boost: Dict[str, float] = Field(default_factory=lambda: {
        "protagonist": 1.5,
        "antagonist": 1.3,
        "supporting": 1.0
    })
    time_decay: float = 0.01


class EmbeddingService:
    """Service for generating embeddings"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = openai.OpenAI()
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def embed_scene(self, 
                   description: str, 
                   entities: List[str], 
                   visual_elements: List[str]) -> List[float]:
        """Generate embedding for a scene combining multiple elements"""
        # Combine all relevant text
        combined_text = f"{description} {' '.join(entities)} {' '.join(visual_elements)}"
        return self.embed_text(combined_text)


class CanonicalMemory:
    """Manages static canonical memories"""
    
    def __init__(self, db: lancedb.DBConnection):
        self.db = db
        self.table_name = "canonical_memories"
        self._init_table()
    
    def _init_table(self):
        """Initialize canonical memories table if it doesn't exist"""
        if self.table_name not in self.db.table_names():
            # Create empty table with initial dummy data
            dummy_embedding = [0.0] * 1536  # Create dummy embedding
            dummy_data = [{
                "id": "_dummy",
                "type": "_dummy",
                "content": "_dummy",
                "embedding": dummy_embedding,
                "metadata": {}
            }]
            
            self.table = self.db.create_table(
                self.table_name,
                data=dummy_data
            )
            
            # Delete the dummy record
            self.table.delete("id = '_dummy'")
        else:
            self.table = self.db.open_table(self.table_name)
    
    def add(self, item: CanonicalMemoryItem):
        """Add a canonical memory item"""
        self.table.add([item.dict()])
    
    def get_by_type(self, memory_type: str) -> List[CanonicalMemoryItem]:
        """Get all canonical memories of a specific type"""
        try:
            results = self.table.search().where(f"type = '{memory_type}'").to_list()
            return [CanonicalMemoryItem(**r) for r in results]
        except:
            return []
    
    def get_by_id(self, memory_id: str) -> Optional[CanonicalMemoryItem]:
        """Get a specific canonical memory by ID"""
        try:
            results = self.table.search().where(f"id = '{memory_id}'").limit(1).to_list()
            if results:
                return CanonicalMemoryItem(**results[0])
        except:
            pass
        return None


class EpisodicMemory:
    """Manages episodic memories (scene summaries)"""
    
    def __init__(self, db: lancedb.DBConnection):
        self.db = db
        self.table_name = "episodic_memories"
        self._init_table()
    
    def _init_table(self):
        """Initialize episodic memories table if it doesn't exist"""
        if self.table_name not in self.db.table_names():
            # Create empty table with initial dummy data
            dummy_embedding = [0.0] * 1536  # Create dummy embedding
            dummy_data = [{
                "scene_id": -1,
                "shot_description": "_dummy",
                "entities_present": [],
                "entity_states": {},
                "camera_spec": {},
                "environment_spec": {},
                "visual_elements": [],
                "embedding": dummy_embedding,
                "generated_images": [],
                "critic_tags": [],
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            }]
            
            self.table = self.db.create_table(
                self.table_name,
                data=dummy_data
            )
            
            # Delete the dummy record
            self.table.delete("scene_id = -1")
        else:
            self.table = self.db.open_table(self.table_name)
    
    def add(self, item: EpisodicMemoryItem):
        """Add an episodic memory item"""
        self.table.add([item.dict()])
    
    def get_by_scene_id(self, scene_id: int) -> Optional[EpisodicMemoryItem]:
        """Get episodic memory for a specific scene"""
        try:
            results = self.table.search().where(f"scene_id = {scene_id}").limit(1).to_list()
            if results:
                return EpisodicMemoryItem(**results[0])
        except:
            pass
        return None
    
    def get_all(self) -> List[EpisodicMemoryItem]:
        """Get all episodic memories"""
        try:
            results = self.table.to_pandas().to_dict('records')
            return [EpisodicMemoryItem(**r) for r in results if r.get('scene_id', -1) != -1]
        except:
            return []
    
    def search_by_embedding(self, 
                          embedding: List[float], 
                          limit: int = 10) -> List[Tuple[EpisodicMemoryItem, float]]:
        """Search episodic memories by embedding similarity"""
        try:
            results = self.table.search(embedding).limit(limit).to_list()
            
            # Return items with their distances
            items_with_scores = []
            for r in results:
                if r.get('scene_id', -1) != -1:  # Skip dummy records
                    item = EpisodicMemoryItem(**r)
                    # LanceDB returns _distance field
                    score = 1.0 - r.get('_distance', 0.0)  # Convert distance to similarity
                    items_with_scores.append((item, score))
            
            return items_with_scores
        except:
            return []


class MemoryService:
    """Main memory service orchestrating all memory tiers"""
    
    def __init__(self, db_path: str = "./memory_db"):
        """Initialize memory service with database path"""
        self.db = lancedb.connect(db_path)
        self.canonical = CanonicalMemory(self.db)
        self.episodic = EpisodicMemory(self.db)
        self.embedding_service = EmbeddingService()
    
    def retrieve_for_scene(self, 
                         current_scene_id: int,
                         scene_description: str,
                         entities: List[str],
                         params: Optional[RetrievalParams] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a scene using semantic similarity and entity overlap
        """
        if params is None:
            params = RetrievalParams()
        
        # Generate embedding for current scene
        current_embedding = self.embedding_service.embed_scene(
            scene_description, 
            entities, 
            []  # No visual elements yet
        )
        
        # Get all episodic memories
        all_memories = self.episodic.get_all()
        
        if not all_memories:
            return []
        
        # Calculate scores for each memory
        scored_memories = []
        current_entities_set = set(entities)
        
        for memory in all_memories:
            # Skip the current scene itself
            if memory.scene_id == current_scene_id:
                continue
            
            # 1. Semantic similarity (using pre-computed search)
            memory_embedding = memory.embedding
            semantic_sim = self._cosine_similarity(current_embedding, memory_embedding)
            
            # 2. Entity overlap score
            memory_entities_set = set(memory.entities_present)
            if current_entities_set and memory_entities_set:
                entity_overlap = len(current_entities_set & memory_entities_set) / \
                               len(current_entities_set | memory_entities_set)
            else:
                entity_overlap = 0.0
            
            # 3. Narrative distance penalty
            distance = abs(current_scene_id - memory.scene_id)
            distance_penalty = 1 / (1 + distance * params.time_decay)
            
            # Combined score with parametric weights
            score = (semantic_sim * params.semantic_weight + 
                    entity_overlap * params.entity_weight + 
                    distance_penalty * params.distance_weight)
            
            # Apply entity boost if applicable
            for entity in current_entities_set & memory_entities_set:
                entity_type = self._get_entity_type(entity)  # Would need entity metadata
                boost = params.entity_boost.get(entity_type, 1.0)
                score *= boost
            
            # Only include if above minimum similarity
            if score >= params.min_similarity:
                scored_memories.append({
                    'memory': memory,
                    'score': score,
                    'semantic_similarity': semantic_sim,
                    'entity_overlap': entity_overlap,
                    'distance_penalty': distance_penalty
                })
        
        # Sort by score and return top K
        scored_memories.sort(key=lambda x: x['score'], reverse=True)
        return scored_memories[:params.k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_entity_type(self, entity_name: str) -> str:
        """Get entity type from canonical memory (simplified for now)"""
        # In a full implementation, this would look up the entity in canonical memory
        # For now, return a default
        return "supporting"
    
    def write_episodic_memory(self, 
                            scene_id: int,
                            shot_description: str,
                            entities: List[str],
                            entity_states: List[EntityState],
                            camera_spec: CameraSpec,
                            environment_spec: EnvironmentSpec,
                            visual_elements: List[str],
                            generated_images: List[ImageRef],
                            critic_tags: List[str]) -> EpisodicMemoryItem:
        """Write a new episodic memory after scene generation"""
        
        # Generate embedding
        embedding = self.embedding_service.embed_scene(
            shot_description,
            entities,
            visual_elements
        )
        
        # Serialize complex objects
        entity_states_dict = {
            state.name: asdict(state) for state in entity_states
        }
        
        # Create episodic memory item
        memory_item = EpisodicMemoryItem(
            scene_id=scene_id,
            shot_description=shot_description,
            entities_present=entities,
            entity_states=entity_states_dict,
            camera_spec=asdict(camera_spec),
            environment_spec=asdict(environment_spec),
            visual_elements=visual_elements,
            embedding=embedding,
            generated_images=[asdict(img) for img in generated_images],
            critic_tags=critic_tags,
            timestamp=datetime.now().isoformat(),
            metadata={}
        )
        
        # Store in database
        self.episodic.add(memory_item)
        
        return memory_item
    
    def initialize_canonical_memories(self, data_path: str = "./data"):
        """Initialize canonical memories from data files"""
        data_path = Path(data_path)
        
        # Load style guide
        style_path = data_path / "style.md"
        if style_path.exists():
            with open(style_path, 'r') as f:
                style_content = f.read()
                style_embedding = self.embedding_service.embed_text(style_content)
                
                self.canonical.add(CanonicalMemoryItem(
                    id="style_guide",
                    type="style",
                    content=style_content,
                    embedding=style_embedding,
                    metadata={"source": "style.md"}
                ))
        
        # Load entities
        entities_path = data_path / "entites.md"
        if entities_path.exists():
            with open(entities_path, 'r') as f:
                entities_content = f.read()
                entities_embedding = self.embedding_service.embed_text(entities_content)
                
                self.canonical.add(CanonicalMemoryItem(
                    id="entities_doc",
                    type="entity",
                    content=entities_content,
                    embedding=entities_embedding,
                    metadata={"source": "entites.md"}
                ))
        
        # Load script
        script_path = data_path / "script.md"
        if script_path.exists():
            with open(script_path, 'r') as f:
                script_content = f.read()
                script_embedding = self.embedding_service.embed_text(script_content)
                
                self.canonical.add(CanonicalMemoryItem(
                    id="full_script",
                    type="script",
                    content=script_content,
                    embedding=script_embedding,
                    metadata={"source": "script.md"}
                )) 