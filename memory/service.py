"""
Memory service for managing canonical, episodic, and working memory.
"""
from typing import List, Dict, Any, Optional
import lancedb
import numpy as np
from datetime import datetime
from pathlib import Path
import asyncio
import pyarrow as pa

from .canonical import CanonicalMemory
from .embeddings import EmbeddingGenerator, EmbeddingCache
from .schemas import EpisodicMemory, ScenePlan


class MemoryService:
    """Central service for all memory operations."""
    
    def __init__(self, db_path: str = "./lancedb_data", data_path: str = "data"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.canonical_memory = CanonicalMemory(data_path)
        self.embedder = EmbeddingGenerator()
        self.embedding_cache = EmbeddingCache(self.embedder)
        
        # Initialize database
        self.db = lancedb.connect(str(self.db_path))
        self._init_tables()
        
    def _init_tables(self):
        """Initialize database tables."""
        # Check if table exists
        existing_tables = self.db.table_names()
        
        if "episodic_memories" not in existing_tables:
            # Create schema using pyarrow
            schema = pa.schema([
                pa.field("scene_id", pa.int32()),
                pa.field("shot_id", pa.int32()),
                pa.field("timestamp", pa.timestamp('ms')),
                pa.field("entities", pa.string()),
                pa.field("entity_states", pa.string()),
                pa.field("camera", pa.string()),
                pa.field("environment", pa.string()),
                pa.field("visual_elements", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), 1536)),
                pa.field("image_paths", pa.string()),
                pa.field("quality_score", pa.float32())
            ])
            
            # Create empty table with schema
            self.db.create_table(
                "episodic_memories",
                schema=schema
            )
    
    async def store_episodic_memory(self, scene_data: Dict[str, Any]):
        """Store a successfully generated scene in episodic memory."""
        # Create scene summary for embedding
        scene_summary = self._create_scene_summary(scene_data)
        
        # Generate embedding
        embedding = await self.embedding_cache.get_or_generate(scene_summary)
        
        # Prepare data for storage
        import json
        memory_data = {
            "scene_id": scene_data["scene_id"],
            "shot_id": scene_data["shot_id"],
            "timestamp": datetime.now(),
            "entities": json.dumps(scene_data.get("entities", [])),
            "entity_states": json.dumps(scene_data.get("entity_states", {})),
            "camera": json.dumps(scene_data.get("camera", {})),
            "environment": json.dumps(scene_data.get("environment", {})),
            "visual_elements": json.dumps(scene_data.get("visual_elements", [])),
            "embedding": embedding,
            "image_paths": json.dumps(scene_data.get("image_paths", [])),
            "quality_score": scene_data.get("quality_score", 0.0)
        }
        
        # Store in database
        table = self.db.open_table("episodic_memories")
        table.add([memory_data])
        
    async def retrieve_working_memory(
        self,
        scene_context: Dict[str, Any],
        k: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve k most relevant memories for the scene."""
        
        if weights is None:
            weights = {
                "semantic": 0.6,
                "entity": 0.3,
                "temporal": 0.1
            }
        
        # Generate embedding for current scene
        scene_text = self._create_scene_summary(scene_context)
        scene_embedding = await self.embedding_cache.get_or_generate(scene_text)
        
        # Extract entities from scene
        scene_entities = set(scene_context.get("entities", []))
        
        # Query episodic memory table
        table = self.db.open_table("episodic_memories")
        
        # Check if table has any data
        try:
            # Try to get count first
            df = table.to_pandas()
            if df.empty:
                return []
            
            # If we have data, perform vector search
            results = table.search(scene_embedding).limit(min(k * 3, len(df))).to_pandas()
            
            if results.empty:
                return []
                
        except Exception as e:
            print(f"Error during memory retrieval: {e}")
            return []
        
        # Calculate composite scores
        import json
        scored_results = []
        
        for _, row in results.iterrows():
            try:
                # Parse stored JSON data
                memory_entities = set(json.loads(row["entities"]))
                
                # Semantic similarity (from vector search - using cosine distance)
                # LanceDB returns L2 distance, convert to similarity
                semantic_score = 1.0 / (1.0 + row["_distance"])
                
                # Entity overlap score
                if scene_entities and memory_entities:
                    entity_score = len(scene_entities & memory_entities) / len(scene_entities | memory_entities)
                else:
                    entity_score = 0
                
                # Temporal proximity score (normalized)
                temporal_score = 1.0 / (1.0 + abs(scene_context.get("shot_id", 0) - row["shot_id"]) / 100)
                
                # Composite score
                final_score = (
                    weights["semantic"] * semantic_score +
                    weights["entity"] * entity_score +
                    weights["temporal"] * temporal_score
                )
                
                # Convert row to dict with parsed JSON fields
                memory_dict = {
                    "scene_id": int(row["scene_id"]),
                    "shot_id": int(row["shot_id"]),
                    "timestamp": row["timestamp"],
                    "entities": json.loads(row["entities"]),
                    "entity_states": json.loads(row["entity_states"]),
                    "camera": json.loads(row["camera"]),
                    "environment": json.loads(row["environment"]),
                    "visual_elements": json.loads(row["visual_elements"]),
                    "image_paths": json.loads(row["image_paths"]),
                    "quality_score": float(row["quality_score"]),
                    "relevance_score": final_score
                }
                
                scored_results.append((final_score, memory_dict))
            except Exception as e:
                print(f"Error processing memory row: {e}")
                continue
        
        # Sort by composite score and return top k
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [memory for _, memory in scored_results[:k]]
    
    def _create_scene_summary(self, scene_context: Dict[str, Any]) -> str:
        """Create a text summary of a scene for embedding generation."""
        parts = []
        
        # Add scene text if available
        if "scene_text" in scene_context:
            parts.append(f"Scene: {scene_context['scene_text']}")
        
        # Add entities
        if "entities" in scene_context:
            entities_str = ", ".join(scene_context["entities"])
            parts.append(f"Entities: {entities_str}")
        
        # Add environment
        if "environment" in scene_context:
            env = scene_context["environment"]
            if isinstance(env, dict):
                env_str = f"Location: {env.get('location', 'unknown')}, Lighting: {env.get('lighting', 'unknown')}"
                parts.append(env_str)
        
        # Add mood/tone
        if "mood" in scene_context:
            parts.append(f"Mood: {scene_context['mood']}")
        
        # Add visual focus
        if "visual_focus" in scene_context:
            parts.append(f"Focus: {scene_context['visual_focus']}")
        
        return " | ".join(parts)
    
    def calculate_retrieval_weights(self, scene_context: Dict[str, Any]) -> Dict[str, float]:
        """Dynamically adjust retrieval weights based on scene type."""
        
        # Default weights
        weights = {"semantic": 0.6, "entity": 0.3, "temporal": 0.1}
        
        # Check for action indicators
        scene_text = scene_context.get("scene_text", "").lower()
        action_keywords = ["fight", "chase", "explosion", "battle", "run", "escape", "crash"]
        
        if any(keyword in scene_text for keyword in action_keywords):
            # Action scenes need more temporal continuity
            weights["temporal"] = 0.3
            weights["semantic"] = 0.5
            weights["entity"] = 0.2
        
        # Check for character-focused scenes
        elif len(scene_context.get("entities", [])) > 2:
            # Multi-character scenes need entity focus
            weights["entity"] = 0.5
            weights["semantic"] = 0.4
            weights["temporal"] = 0.1
        
        return weights 