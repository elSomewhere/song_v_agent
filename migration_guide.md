# Migration Guide: From Current Implementation to RAG-Based System

## Overview

This guide outlines the key changes and migration steps from the current sliding-window-based implementation to the new semantic RAG-based system.

## Key Architectural Changes

### 1. Memory Management

**Current System:**
- Uses chronological sliding window (last N images)
- Stores entire context in state
- Linear growth in token usage

**New System:**
- Semantic retrieval based on scene similarity and entity overlap
- Tiered memory architecture (Canonical, Episodic, Working)
- Constant token usage regardless of story length

### 2. Agent Structure

**Current System:**
- 4 agents: Director, Artist, Critic, Reviewer (implicit)
- Multiple round-trips between agents
- Redundant processing

**New System:**
- 3 agents: Planner, Renderer, Vision QA
- Streamlined workflow with minimal loops
- Memory Service as a separate component

### 3. Communication Format

**Current System:**
- Mix of structured and unstructured data
- Markdown-based prompts and responses
- Potential for hallucinated fields

**New System:**
- Strict JSON schemas for all communication
- Pydantic models for validation
- Type-safe interfaces

### 4. Context Selection

**Current System:**
```python
# Simple chronological window
historical_context = historical_context[-context_window:]
```

**New System:**
```python
# Intelligent semantic retrieval
relevance_score = 0.6 * semantic_similarity + 
                  0.3 * entity_overlap + 
                  0.1 * temporal_proximity
```

## Migration Steps

### Step 1: Install New Dependencies

```bash
pip install lancedb pydantic python-json-logger
```

### Step 2: Create Memory Service

Replace the current context management with the new memory service:

```python
# Old approach
historical_context.extend(new_data)
historical_context = historical_context[-context_window:]

# New approach
memory_service.store_episodic_memory(scene_data)
working_memory = memory_service.retrieve_working_memory(scene_context, k=5)
```

### Step 3: Refactor Agent Logic

#### Director → Planner
- Extract entity detection logic
- Merge with Artist's prompt creation
- Output structured JSON instead of markdown

#### Artist → Renderer  
- Simplify to pure image generation
- Remove prompt creation logic (moved to Planner)
- Add canonical appearance expansion

#### Critic + Reviewer → Vision QA
- Combine evaluation logic
- Implement structured feedback schema
- Add quality scoring

### Step 4: Update Workflow Graph

```python
# Simplified workflow
workflow.add_edge("planner", "renderer")
workflow.add_edge("renderer", "vision_qa")
workflow.add_conditional_edges("vision_qa", route_from_qa, {...})
```

### Step 5: Implement Semantic Embeddings

Add embedding generation for scene summaries:

```python
from openai import OpenAI

def generate_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding
```

### Step 6: Create Database Schema

Initialize LanceDB tables for episodic memory:

```python
import lancedb

db = lancedb.connect("./lancedb_data")
table = db.create_table(
    "episodic_memories",
    schema=EpisodicMemory.schema()
)
```

### Step 7: Update Command-Line Interface

Add new arguments while maintaining backward compatibility:

```python
# Keep existing arguments
parser.add_argument("--context-window", type=int, help="(Deprecated) Use --memory-k instead")

# Add new arguments
parser.add_argument("--memory-k", type=int, default=5)
parser.add_argument("--memory-weights", type=str, default="0.6,0.3,0.1")
```

## Code Mapping

### Context Building

**Before:**
```python
def build_complete_context(historical_context: List[Dict]) -> Dict:
    # Loads all base documents every time
    base_docs = load_base_documents()
    return {"base": base_docs, "history": historical_context}
```

**After:**
```python
class MemoryService:
    def __init__(self):
        # Load canonical memory once
        self.canonical_memory = CanonicalMemory()
        
    def get_context(self, scene: Dict) -> Dict:
        # Only retrieve relevant memories
        working_memory = self.retrieve_working_memory(scene)
        return {
            "canonical": self.canonical_memory,
            "working": working_memory
        }
```

### Image Generation

**Before:**
```python
# In artist_node
prompt = create_detailed_prompt(shot_text, entities, style, history)
image = generate_dalle_image(prompt)
```

**After:**
```python
# In renderer_node
prompt = scene_plan["image_prompt"]  # Already optimized by Planner
prompt = expand_canonical_references(prompt, canonical_memory)
image = generate_gpt4o_image(prompt)
```

## Performance Comparison

| Metric | Current System | New System |
|--------|---------------|------------|
| Tokens per scene (after 50 scenes) | ~15,000 | ~3,000 |
| API calls per scene | 4-6 | 3 |
| Context relevance | Time-based | Semantic |
| Memory usage | O(n) | O(1) |

## Backward Compatibility

To ensure smooth transition:

1. Keep `/data` and `/output` structure unchanged
2. Support legacy command-line arguments
3. Provide migration tool for existing outputs
4. Allow fallback to chronological mode

## Testing the Migration

1. Run both systems in parallel on same input
2. Compare output quality and consistency
3. Measure token usage reduction
4. Validate semantic retrieval accuracy

## Rollback Plan

If issues arise:

1. Keep original code in `legacy/` directory
2. Use feature flags to toggle between systems
3. Maintain database export to JSON capability
4. Document any manual interventions needed 