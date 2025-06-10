# Technical Design Document: RAG-Based Storyboard Generation System

## 1. Executive Summary

This document outlines the complete technical design for a next-generation storyboard generation system that leverages a pure RAG (Retrieval-Augmented Generation) approach with GPT-4o vision capabilities. The system is designed to be:

- **Memory-efficient**: Using semantic retrieval instead of chronological sliding windows
- **Generalized**: Works with any structured story input following the defined schema
- **Consistent**: Maintains visual and narrative coherence across hundreds of scenes
- **Scalable**: Minimizes token usage through intelligent memory management
- **Robust**: Includes quality assurance and retry mechanisms

### Key Innovations

1. **Semantic Memory Retrieval**: Replaces chronological context windows with intelligent retrieval based on scene similarity and entity overlap
2. **Structured JSON Communication**: All agent communication uses strictly typed JSON schemas
3. **Tiered Memory Architecture**: Separates canonical (static), episodic (historical), and working (active) memory
4. **Minimal Agent Design**: Reduces from 4 agents to 3 core agents plus a memory service
5. **GPT-4o Native Integration**: Leverages GPT-4o's contextual image generation capabilities

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Input Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  script.md  │  │  style.md    │  │  entities.md       │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Processing Layer                            │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   Planner   │  │   Renderer   │  │   Vision QA        │    │
│  │   (GPT-4)   │  │   (GPT-4o)   │  │   (GPT-4o-vision)  │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       Memory Layer                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Memory Service                        │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐ │   │
│  │  │  Canonical  │  │   Episodic   │  │   Working     │ │   │
│  │  │   Memory    │  │    Memory    │  │   Memory      │ │   │
│  │  └─────────────┘  └──────────────┘  └───────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                    ┌──────▼────────┐                           │
│                    │   LanceDB     │                           │
│                    └───────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       Output Layer                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  /output/run_YYYYMMDD_HHMMSS/                          │   │
│  │    ├── shot_001_var_1.png                              │   │
│  │    ├── shot_001_var_2.png                              │   │
│  │    └── metadata.json                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Memory Architecture

#### 2.2.1 Canonical Memory (Static)
- **Content**: Base documents (style.md, entities.md), parsed entity appearance vectors
- **Storage**: Indexed once at startup, read-only during execution
- **Access Pattern**: Direct retrieval by entity name or style reference

#### 2.2.2 Episodic Memory (Append-Only)
- **Content**: JSON summaries of each successfully generated scene
- **Schema**:
  ```json
  {
    "scene_id": "int",
    "shot_id": "int",
    "timestamp": "iso8601",
    "entities": ["list of entity names"],
    "entity_states": {
      "entity_name": {
        "pose": "string",
        "emotion": "string",
        "position": "string"
      }
    },
    "camera": {
      "type": "string",
      "angle": "string",
      "distance": "string"
    },
    "environment": {
      "location": "string",
      "lighting": "string",
      "mood": "string"
    },
    "visual_elements": ["list of key visual elements"],
    "embedding": "vector[1536]",
    "image_paths": ["list of generated image paths"],
    "quality_score": "float"
  }
  ```
- **Storage**: LanceDB with vector indexing
- **Access Pattern**: Semantic search via embeddings

#### 2.2.3 Working Memory (Dynamic)
- **Content**: K most relevant episodic memories for current scene
- **Selection Algorithm**:
  ```python
  relevance_score = α * cosine_similarity(scene_embedding, memory_embedding) +
                    β * entity_overlap_score +
                    γ * temporal_proximity_score
  ```
  Where:
  - α = 0.6 (semantic similarity weight)
  - β = 0.3 (entity overlap weight)
  - γ = 0.1 (temporal proximity weight)

### 2.3 Agent Specifications

#### 2.3.1 Planner Agent
**Role**: Analyzes script scenes and creates structured generation plans

**Input Schema**:
```json
{
  "scene_text": "string",
  "scene_number": "int",
  "working_memory": ["array of episodic memory objects"],
  "canonical_style": "object",
  "canonical_entities": "object"
}
```

**Output Schema**:
```json
{
  "scene_id": "int",
  "shot_id": "int",
  "entities": [
    {
      "name": "string",
      "pose": "string",
      "emotion": "string",
      "position": "string",
      "clothing_state": "string"
    }
  ],
  "camera": {
    "type": "string",
    "angle": "string",
    "distance": "string",
    "movement": "string"
  },
  "environment": {
    "location": "string",
    "time_of_day": "string",
    "weather": "string",
    "lighting": "string"
  },
  "action_beats": ["array of action descriptions"],
  "visual_focus": "string",
  "mood": "string",
  "continuity_notes": "string",
  "image_prompt": "string"
}
```

**Processing Logic**:
1. Parse scene text to extract entities, actions, and environment
2. Query working memory for relevant context
3. Identify continuity requirements from previous scenes
4. Generate structured scene plan with all visual specifications
5. Create optimized image generation prompt

#### 2.3.2 Renderer Agent
**Role**: Generates images using GPT-4o vision capabilities

**Input Schema**:
```json
{
  "scene_plan": "object (from Planner)",
  "variation_number": "int",
  "previous_attempts": ["array of previous attempt results"],
  "canonical_appearances": "object"
}
```

**Output Schema**:
```json
{
  "status": "success|error",
  "image_data": "base64|url",
  "generation_metadata": {
    "model": "gpt-4o",
    "timestamp": "iso8601",
    "token_usage": "int",
    "generation_time_ms": "int"
  },
  "error": "string (if status=error)"
}
```

**Processing Logic**:
1. Expand entity placeholders with canonical appearances
2. Inject style consistency tokens
3. Add variation-specific modifiers if variation_number > 1
4. Submit to GPT-4o with contextual generation enabled
5. Handle response and extract image data

#### 2.3.3 Vision QA Agent
**Role**: Validates generated images against scene requirements

**Input Schema**:
```json
{
  "scene_plan": "object",
  "generated_image": "base64|url",
  "attempt_number": "int",
  "previous_feedback": ["array of previous QA feedback"]
}
```

**Output Schema**:
```json
{
  "status": "pass|retry|fail",
  "quality_score": "float (0-1)",
  "feedback": {
    "composition": "string",
    "entity_accuracy": "object",
    "continuity": "string",
    "technical_quality": "string"
  },
  "specific_issues": ["array of issue descriptions"],
  "retry_guidance": "string (if status=retry)"
}
```

**Evaluation Criteria**:
1. Entity presence and accuracy
2. Pose and emotion matching
3. Environmental consistency
4. Technical quality (composition, lighting, clarity)
5. Continuity with established visual language

### 2.4 Memory Service

**Core Functions**:

1. **initialize_canonical_memory()**
   - Parses style.md and entities.md
   - Creates entity appearance vectors
   - Builds style token mappings

2. **store_episodic_memory(scene_data)**
   - Generates embedding for scene
   - Stores in LanceDB with metadata
   - Updates indices

3. **retrieve_working_memory(scene_context, k=5)**
   - Computes scene embedding
   - Performs vector similarity search
   - Applies entity overlap scoring
   - Returns top-k relevant memories

4. **compute_embeddings(text)**
   - Uses text-embedding-ada-002 or GPT-4 embeddings
   - Caches results for efficiency

## 3. Implementation Details

### 3.1 Directory Structure

```
project_root/
├── run.py                 # Main entry point
├── config.py             # Configuration management
├── requirements.txt      # Dependencies
├── .env                 # Environment variables
│
├── agents/
│   ├── __init__.py
│   ├── planner.py       # Planner agent implementation
│   ├── renderer.py      # Renderer agent implementation
│   ├── vision_qa.py     # Vision QA agent implementation
│   └── base.py          # Base agent class
│
├── memory/
│   ├── __init__.py
│   ├── service.py       # Memory service implementation
│   ├── canonical.py     # Canonical memory parser
│   ├── episodic.py      # Episodic memory manager
│   ├── embeddings.py    # Embedding generation
│   └── schemas.py       # Pydantic models
│
├── workflow/
│   ├── __init__.py
│   ├── graph.py         # LangGraph workflow definition
│   ├── state.py         # State management
│   └── nodes.py         # Node implementations
│
├── utils/
│   ├── __init__.py
│   ├── parsing.py       # Input parsing utilities
│   ├── prompts.py       # Prompt templates
│   └── logging.py       # Logging configuration
│
├── data/                # Input data (unchanged)
│   ├── script.md
│   ├── style.md
│   ├── entities.md
│   └── refs/
│
└── output/              # Generated outputs (unchanged)
    └── run_YYYYMMDD_HHMMSS/
```

### 3.2 Key Classes and Interfaces

#### 3.2.1 Memory Schemas (Pydantic)

```python
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class EntityState(BaseModel):
    name: str
    pose: str
    emotion: str
    position: str
    clothing_state: Optional[str] = None

class CameraSpec(BaseModel):
    type: str
    angle: str
    distance: str
    movement: Optional[str] = None

class EnvironmentSpec(BaseModel):
    location: str
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    lighting: str

class ScenePlan(BaseModel):
    scene_id: int
    shot_id: int
    entities: List[EntityState]
    camera: CameraSpec
    environment: EnvironmentSpec
    action_beats: List[str]
    visual_focus: str
    mood: str
    continuity_notes: Optional[str] = None
    image_prompt: str

class EpisodicMemory(BaseModel):
    scene_id: int
    shot_id: int
    timestamp: datetime
    entities: List[str]
    entity_states: Dict[str, Dict[str, str]]
    camera: Dict[str, str]
    environment: Dict[str, str]
    visual_elements: List[str]
    embedding: List[float]
    image_paths: List[str]
    quality_score: float
```

#### 3.2.2 Workflow State

```python
from typing import TypedDict, List, Dict, Any, Optional

class WorkflowState(TypedDict):
    # Current processing state
    current_scene_number: int
    current_shot_id: int
    current_variation: int
    max_variations: int
    
    # Scene data
    scene_text: str
    scene_plan: Optional[Dict[str, Any]]
    
    # Memory context
    working_memory: List[Dict[str, Any]]
    canonical_memory: Dict[str, Any]
    
    # Generation state
    status: str  # planning|rendering|evaluating|complete|error
    attempt_number: int
    max_attempts: int
    
    # Results
    generated_images: List[str]
    quality_scores: List[float]
    feedback_history: List[Dict[str, Any]]
    
    # Configuration
    output_base_path: str
    embedding_cache: Dict[str, List[float]]
```

### 3.3 Workflow Definition

```python
def create_workflow(config: WorkflowConfig) -> CompiledGraph:
    """Create the LangGraph workflow."""
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("renderer", renderer_node)
    workflow.add_node("vision_qa", vision_qa_node)
    workflow.add_node("memory_update", memory_update_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Define edges
    workflow.add_edge("planner", "renderer")
    workflow.add_edge("renderer", "vision_qa")
    
    # Conditional routing from vision_qa
    def route_from_qa(state: WorkflowState) -> str:
        if state["status"] == "pass":
            if state["current_variation"] < state["max_variations"]:
                return "renderer"  # Generate another variation
            else:
                return "memory_update"  # All variations complete
        elif state["status"] == "retry" and state["attempt_number"] < state["max_attempts"]:
            return "renderer"  # Retry current variation
        else:
            return "memory_update"  # Max attempts reached or failed
    
    workflow.add_conditional_edges(
        "vision_qa",
        route_from_qa,
        {
            "renderer": "renderer",
            "memory_update": "memory_update"
        }
    )
    
    # Memory update always goes to END
    workflow.add_edge("memory_update", END)
    
    return workflow.compile()
```

### 3.4 Semantic Retrieval Implementation

```python
class MemoryService:
    def __init__(self, db_path: str):
        self.db = lancedb.connect(db_path)
        self.canonical_memory = CanonicalMemory()
        self.embedder = EmbeddingGenerator()
        
    def retrieve_working_memory(
        self,
        scene_context: Dict[str, Any],
        k: int = 5,
        weights: Dict[str, float] = None
    ) -> List[EpisodicMemory]:
        """Retrieve k most relevant memories for the scene."""
        
        if weights is None:
            weights = {
                "semantic": 0.6,
                "entity": 0.3,
                "temporal": 0.1
            }
        
        # Generate embedding for current scene
        scene_text = self._create_scene_summary(scene_context)
        scene_embedding = self.embedder.generate(scene_text)
        
        # Extract entities from scene
        scene_entities = set(scene_context.get("entities", []))
        
        # Query episodic memory table
        memories = self.db.open_table("episodic_memories")
        
        # Perform vector search
        results = memories.search(
            query_vector=scene_embedding,
            limit=k * 3  # Get more candidates for reranking
        ).to_pandas()
        
        # Calculate composite scores
        scored_results = []
        for _, row in results.iterrows():
            memory_entities = set(row["entities"])
            
            # Semantic similarity (from vector search)
            semantic_score = row["_distance"]
            
            # Entity overlap score
            if scene_entities and memory_entities:
                entity_score = len(scene_entities & memory_entities) / len(scene_entities | memory_entities)
            else:
                entity_score = 0
            
            # Temporal proximity score (normalized)
            temporal_score = 1.0 / (1.0 + abs(scene_context["shot_id"] - row["shot_id"]) / 100)
            
            # Composite score
            final_score = (
                weights["semantic"] * semantic_score +
                weights["entity"] * entity_score +
                weights["temporal"] * temporal_score
            )
            
            scored_results.append((final_score, row))
        
        # Sort by composite score and return top k
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [
            EpisodicMemory(**row.to_dict())
            for _, row in scored_results[:k]
        ]
```

### 3.5 Prompt Engineering

#### 3.5.1 Planner Prompt Template

```python
PLANNER_PROMPT = """You are a film director planning the visual composition for a scene.

CANONICAL STYLE:
{canonical_style}

CANONICAL ENTITIES:
{canonical_entities}

RELEVANT CONTEXT (similar scenes):
{working_memory}

CURRENT SCENE:
{scene_text}

Create a detailed visual plan for this scene. Focus on:
1. Entity positioning and emotional states
2. Camera angles and movements
3. Environmental details and lighting
4. Visual continuity with established style
5. Key action beats to capture

Output your plan as a JSON object following this exact schema:
{output_schema}
"""
```

#### 3.5.2 Vision QA Prompt Template

```python
VISION_QA_PROMPT = """You are a film continuity supervisor reviewing a generated storyboard frame.

SCENE PLAN:
{scene_plan}

Evaluate the generated image for:
1. Accuracy of entity representation (appearance, pose, emotion)
2. Adherence to camera specifications
3. Environmental consistency
4. Overall composition quality
5. Continuity with established visual style

Previous feedback for this scene:
{previous_feedback}

Provide your evaluation as a JSON object following this schema:
{output_schema}
"""
```

## 4. Advanced Features

### 4.1 Dynamic Weight Adjustment

The system can adjust retrieval weights based on scene characteristics:

```python
def calculate_retrieval_weights(scene_context: Dict) -> Dict[str, float]:
    """Dynamically adjust retrieval weights based on scene type."""
    
    # Default weights
    weights = {"semantic": 0.6, "entity": 0.3, "temporal": 0.1}
    
    # Adjust for action scenes (prioritize temporal continuity)
    if "action" in scene_context.get("tags", []):
        weights["temporal"] = 0.3
        weights["semantic"] = 0.5
        weights["entity"] = 0.2
    
    # Adjust for character-focused scenes
    elif len(scene_context.get("entities", [])) > 2:
        weights["entity"] = 0.5
        weights["semantic"] = 0.4
        weights["temporal"] = 0.1
    
    return weights
```

### 4.2 Embedding Cache Management

```python
class EmbeddingCache:
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_counts = {}
        
    def get_or_generate(self, text: str, generator_func) -> List[float]:
        """Get embedding from cache or generate if not present."""
        
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.cache:
            self.access_counts[cache_key] += 1
            return self.cache[cache_key]
        
        # Generate new embedding
        embedding = generator_func(text)
        
        # Evict least accessed if at capacity
        if len(self.cache) >= self.max_size:
            least_accessed = min(self.access_counts.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_counts[least_accessed]
        
        # Store new embedding
        self.cache[cache_key] = embedding
        self.access_counts[cache_key] = 1
        
        return embedding
```

### 4.3 Batch Processing Optimization

```python
async def process_scene_batch(
    scenes: List[Dict],
    workflow: CompiledGraph,
    memory_service: MemoryService,
    max_concurrent: int = 3
) -> List[Dict]:
    """Process multiple scenes with controlled concurrency."""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(scene):
        async with semaphore:
            # Retrieve working memory for this scene
            working_memory = memory_service.retrieve_working_memory(scene)
            
            # Process through workflow
            state = {
                "current_scene_number": scene["number"],
                "scene_text": scene["text"],
                "working_memory": working_memory,
                "canonical_memory": memory_service.canonical_memory.to_dict()
            }
            
            result = await workflow.ainvoke(state)
            
            # Store successful results in episodic memory
            if result["status"] == "complete":
                memory_service.store_episodic_memory(result)
            
            return result
    
    # Process all scenes concurrently with semaphore
    tasks = [process_with_semaphore(scene) for scene in scenes]
    return await asyncio.gather(*tasks)
```

## 5. Command-Line Interface

### 5.1 Arguments

```python
parser = argparse.ArgumentParser(description="RAG-based Storyboard Generator")

# Basic options
parser.add_argument("--start-shot", type=int, default=1,
                   help="Starting shot ID")
parser.add_argument("--max-shots", type=int, default=60,
                   help="Maximum number of shots to process")
parser.add_argument("--variations", type=int, default=3,
                   help="Number of variations per shot")

# Memory options
parser.add_argument("--memory-k", type=int, default=5,
                   help="Number of memories to retrieve per scene")
parser.add_argument("--memory-weights", type=str, default="0.6,0.3,0.1",
                   help="Weights for semantic,entity,temporal (comma-separated)")

# Retry options
parser.add_argument("--max-retries", type=int, default=3,
                   help="Maximum retries per image")
parser.add_argument("--quality-threshold", type=float, default=0.7,
                   help="Minimum quality score to accept")

# Performance options
parser.add_argument("--batch-size", type=int, default=1,
                   help="Number of scenes to process in parallel")
parser.add_argument("--cache-embeddings", action="store_true",
                   help="Enable embedding cache")

# Output options
parser.add_argument("--output-dir", type=str, default="output",
                   help="Base output directory")
parser.add_argument("--save-metadata", action="store_true",
                   help="Save detailed metadata for each generation")
```

### 5.2 Usage Examples

```bash
# Basic usage with defaults
python run.py

# Process specific range with custom variations
python run.py --start-shot 10 --max-shots 20 --variations 5

# Adjust memory retrieval
python run.py --memory-k 10 --memory-weights "0.7,0.2,0.1"

# High-quality mode with stricter thresholds
python run.py --quality-threshold 0.85 --max-retries 5

# Batch processing for efficiency
python run.py --batch-size 3 --cache-embeddings
```

## 6. Error Handling and Recovery

### 6.1 Graceful Degradation

```python
class ErrorHandler:
    def __init__(self, config: Dict):
        self.config = config
        self.error_counts = defaultdict(int)
        
    def handle_agent_error(self, agent_name: str, error: Exception, state: Dict) -> Dict:
        """Handle errors from individual agents."""
        
        self.error_counts[agent_name] += 1
        
        if isinstance(error, RateLimitError):
            # Exponential backoff for rate limits
            wait_time = 2 ** self.error_counts[agent_name]
            return {"action": "retry", "wait": wait_time}
            
        elif isinstance(error, ValidationError):
            # Skip this shot if validation consistently fails
            if self.error_counts[agent_name] > 3:
                return {"action": "skip", "reason": str(error)}
            else:
                return {"action": "retry", "modified_state": self.fix_validation(state)}
                
        else:
            # Log and continue with degraded functionality
            logger.error(f"Unhandled error in {agent_name}: {error}")
            return {"action": "continue", "fallback": True}
```

### 6.2 Checkpoint and Resume

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, state: Dict, scene_id: int):
        """Save processing state for resume capability."""
        
        checkpoint = {
            "scene_id": scene_id,
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "episodic_memory_snapshot": self.get_memory_snapshot()
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{scene_id}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
            
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Load the most recent checkpoint."""
        
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None
            
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            return json.load(f)
```

## 7. Performance Optimizations

### 7.1 Token Usage Optimization

1. **Prompt Compression**: Use GPT-4 to summarize lengthy context before including in prompts
2. **Selective Field Inclusion**: Only include relevant memory fields based on scene requirements
3. **Caching**: Cache embeddings, canonical memory parsing, and style expansions

### 7.2 Parallel Processing

1. **Scene-level Parallelism**: Process independent scenes concurrently
2. **Variation-level Parallelism**: Generate variations in parallel when possible
3. **Async I/O**: Use async operations for all API calls and file operations

### 7.3 Memory Efficiency

1. **Streaming Processing**: Process script in chunks rather than loading entire document
2. **Lazy Loading**: Load reference images only when needed
3. **Memory Pooling**: Reuse memory allocations for image data

## 8. Monitoring and Observability

### 8.1 Metrics Collection

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "scenes_processed": 0,
            "images_generated": 0,
            "api_calls": defaultdict(int),
            "token_usage": defaultdict(int),
            "quality_scores": [],
            "retry_counts": defaultdict(int),
            "processing_times": defaultdict(list)
        }
        
    def record_api_call(self, service: str, tokens: int, duration: float):
        self.metrics["api_calls"][service] += 1
        self.metrics["token_usage"][service] += tokens
        self.metrics["processing_times"][service].append(duration)
        
    def get_summary(self) -> Dict:
        return {
            "total_scenes": self.metrics["scenes_processed"],
            "total_images": self.metrics["images_generated"],
            "avg_quality": np.mean(self.metrics["quality_scores"]),
            "total_tokens": sum(self.metrics["token_usage"].values()),
            "avg_processing_time": {
                service: np.mean(times)
                for service, times in self.metrics["processing_times"].items()
            }
        }
```

### 8.2 Logging Configuration

```python
# Structured logging with context
logging.config.dictConfig({
    "version": 1,
    "formatters": {
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "storyboard_generator.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["file"]
    }
})
```

## 9. Testing Strategy

### 9.1 Unit Tests

- Test each agent independently with mocked dependencies
- Test memory service retrieval algorithms
- Test prompt generation and parsing
- Test error handling scenarios

### 9.2 Integration Tests

- Test full workflow with sample scenes
- Test memory persistence and retrieval
- Test checkpoint/resume functionality
- Test batch processing

### 9.3 Quality Tests

- Automated evaluation of generated images against scene plans
- Consistency scoring across variations
- Performance benchmarking

## 10. Deployment Considerations

### 10.1 Environment Setup

```bash
# Required environment variables
OPENAI_API_KEY=your_api_key
LANCEDB_PATH=./lancedb_data
LOG_LEVEL=INFO
MAX_WORKERS=4
```

### 10.2 Resource Requirements

- **Memory**: 8GB minimum, 16GB recommended
- **Storage**: 100GB for output images and vector database
- **Network**: Stable connection for API calls
- **GPU**: Not required (using cloud APIs)

### 10.3 Scaling Considerations

1. **Horizontal Scaling**: Deploy multiple instances with shared database
2. **Queue-based Processing**: Use message queue for scene distribution
3. **CDN Integration**: Store generated images in CDN for faster access
4. **Database Sharding**: Shard episodic memory by time range or scene range

## 11. Future Enhancements

1. **Multi-modal Embeddings**: Use CLIP or similar for image-text alignment
2. **Style Transfer Networks**: Fine-tune style consistency models
3. **Interactive Mode**: Real-time adjustments based on user feedback
4. **A/B Testing Framework**: Compare different retrieval strategies
5. **Advanced Caching**: Predictive prefetching of likely memories

## 12. Conclusion

This design provides a robust, scalable, and efficient system for generating consistent storyboards using a pure RAG approach with GPT-4o. The semantic retrieval mechanism ensures relevant context without token explosion, while the structured JSON communication enables reliable agent coordination. The system is generalized to work with any story following the defined input schema, making it adaptable to various narrative projects. 