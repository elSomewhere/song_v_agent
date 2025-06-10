# AI Storyboard Generator - Project Overview

## What This Project Achieves

The AI Storyboard Generator is an intelligent system that automatically creates professional storyboard frames from a script using a multi-agent workflow powered by GPT-4 and GPT-4.1-mini. The system is designed to generate cinematic, visually consistent storyboard images in a specific artistic style (gritty anime/manga) while maintaining character consistency across multiple frames.

### Key Achievements:
1. **Automated Visual Storytelling**: Converts written shot descriptions into high-quality storyboard images
2. **Consistent Character Representation**: Maintains visual consistency of characters across multiple frames using reference images
3. **Style Adherence**: Generates images in a specific artistic style defined by style guidelines
4. **Quality Control**: Implements an automated critique and retry system to ensure image quality
5. **Contextual Awareness**: Optionally maintains context across sequential shots for narrative coherence
6. **Scalable Processing**: Supports parallel processing while managing API rate limits

## Architecture & Agent Workflow

The system uses LangGraph to orchestrate three specialized AI agents in a Director → Artist → Critic workflow:

### 1. **Director Agent** (`agents/director.py`)
- **Purpose**: Orchestrates the entire workflow and manages shot processing
- **Responsibilities**:
  - Reads shot descriptions sequentially from the database
  - Dynamically detects entities (characters, objects, environments) mentioned in each shot
  - Manages the workflow state and determines which shot to process next
  - Checks if shots have already been processed to support resume functionality

### 2. **Artist Agent** (`agents/artist.py`)
- **Purpose**: Generates storyboard images using GPT-4.1-mini
- **Responsibilities**:
  - Constructs detailed image generation prompts based on:
    - Shot description and framing details
    - Reference images for detected entities
    - Style guidelines from `data/style.md`
    - Optional historical context from previous shots
  - Calls OpenAI's image generation API
  - Saves generated images with proper naming conventions

### 3. **Critic Agent** (`agents/critic.py`)
- **Purpose**: Evaluates generated images for quality and adherence to requirements
- **Responsibilities**:
  - Uses GPT-4o vision capabilities to analyze generated images
  - Checks for:
    - Correct aspect ratio (16:9)
    - Style adherence
    - Character accuracy
    - Shot composition matching the description
  - Provides specific feedback for improvements
  - Determines whether to retry, accept, or reject images

## Context Management

The system implements sophisticated context management to maintain coherence across shots:

### 1. **Entity Detection & Reference System**
- **Dynamic Entity Detection**: The Director uses GPT-4 to analyze `data/entites.md` and extract all entities with their aliases
- **Reference Image Lookup**: For each detected entity, the system retrieves relevant reference images from `data/refs/[entity_name]/`
- **Vector Database**: Uses LanceDB to store and efficiently retrieve:
  - Shot descriptions with embeddings
  - Reference images with metadata
  - Entity relationships

### 2. **Base Context (Always Included)**
The system maintains a base context that is **always** provided to all AI agents, containing:
- **Style Guidelines** (`data/style.md`): Artistic direction and visual requirements
- **Entities Document** (`data/entites.md`): Character descriptions and relationships
- **Full Script** (`data/script.md`): Complete narrative context
- **Additional Documents**: Any other markdown files in the `data/` folder

This ensures that every AI agent has access to the complete creative vision and narrative context, regardless of the sliding window settings.

### 3. **Historical Context Options**
The system supports three context modes (controlled via `--context-window` parameter):
- **Fresh Context (0)**: Each shot is generated independently (but still includes base documents)
- **Sliding Window (N)**: Includes the last N successfully generated images as context
- **Full History (-1)**: Includes all previous images as context

Additionally, the `--context-variations` parameter controls how many variations per shot are included in the historical context:
- **All Variations (-1)**: Include all variations from each shot (default)
- **Limited Variations (N)**: Include only the first N variations from each shot

This allows for better temporal coverage. For example:
- With `--variations 3 --context-window 15 --context-variations 1`:
  - Generate 3 variations per shot
  - But only include 1 variation per shot in context
  - Result: Context spans 15 different shots instead of just 5

### 4. **Context Flow**
```
Base Documents (Always Present)
        ↓
Previous Shots → Historical Context → Artist Prompt → Generated Image
                                   ↓
                         Entity References + Style Guidelines
```

### 5. **Context Usage by Agents**
- **Director Agent**: Uses base context to detect entities and understand narrative flow
- **Artist Agent**: 
  - Includes script context (surrounding shots) in prompts
  - Uses style guidelines and entity descriptions
  - Incorporates historical images for visual continuity
- **Critic Agent**: 
  - Evaluates against style guidelines from base context
  - Checks narrative consistency using script context
  - Compares with historical images for continuity

## Data Organization

### Input Data Structure (`/data/`)
The `data/` folder is designed to be replaceable with any content while maintaining the same structure:

```
data/
├── script.md         # Shot-by-shot descriptions in markdown format
├── style.md          # Visual style guidelines and artistic direction  
├── entites.md        # Character/entity descriptions and relationships
└── refs/            # Reference images organized by entity
    ├── Helena/      # Character reference images
    ├── Joy/         # Another character
    ├── Silicate Army/  # Environment/object references
    └── ...
```

### Data Processing Pipeline
1. **Text Ingestion** (`ingest/ingest_text.py`):
   - Parses `script.md` to extract individual shots
   - Creates embeddings for semantic search
   - Stores in LanceDB `shots` table

2. **Reference Ingestion** (`ingest/ingest_refs.py`):
   - Processes images in `refs/` subdirectories
   - Generates CLIP embeddings for visual similarity
   - Stores in LanceDB `refs` table with entity tags

### Output Structure (`/output/`)
```
output/
└── run_YYYYMMDD_HHMMSS/     # Timestamped run folder
    ├── Shot-001/             # Per-shot output
    │   ├── var1.png         # First variation
    │   ├── var2.png         # Second variation
    │   ├── var3.png         # Third variation
    │   ├── prompt_var1.json # Generation prompt
    │   ├── prompt_var2.json
    │   ├── prompt_var3.json
    │   ├── critic_var1_try0.json  # Critic feedback
    │   └── ...
    └── Shot-002/
        └── ...
```

## Key Features & Mechanisms

### 1. **Retry Mechanism**
- Each image can be retried up to `--max-retries` times (default: 1)
- Critic provides specific feedback that's incorporated into retry prompts
- Exponential backoff for API failures

### 2. **Variation Generation**
- Generates multiple variations per shot (default: 3, configurable via `--variations`)
- Each variation uses the same shot description but may produce different interpretations

### 3. **Rate Limiting & Parallel Processing**
- Configurable API rate limits to avoid OpenAI throttling
- Supports parallel workers for faster processing (when not using context mode)
- Smart queuing system to maximize throughput

### 4. **Resume Capability**
- Automatically detects already-processed shots
- Can resume from any shot ID
- Preserves all previous outputs

## Customization & Extensibility

### Replacing Input Data
To use this system with different content:

1. **Script Format** (`data/script.md`):
   - Must contain shots marked with `### Shot N` or `### **Shot N**`
   - Each shot should include Description, Framing, and Camera Angle

2. **Style Guidelines** (`data/style.md`):
   - Describe the desired artistic style
   - Include color palette, mood, techniques
   - Can reference specific artistic movements or examples

3. **Entity Descriptions** (`data/entites.md`):
   - List all characters, objects, and environments
   - Include visual descriptions and relationships
   - The system will automatically extract entities and aliases

4. **Reference Images** (`data/refs/`):
   - Create subdirectories for each entity
   - Add reference images (PNG/JPG) to appropriate folders
   - Directory names should match entity names

### Configuration Options
- `--shot-id N`: Start from shot N
- `--max-shots N`: Process N shots total
- `--variations N`: Generate N variations per shot
- `--context-window N`: Include N previous images as context
- `--context-variations N`: Include N variations per shot in context (-1 for all)
- `--max-retries N`: Maximum retry attempts per image

## Technical Stack
- **Orchestration**: LangGraph for agent workflow management
- **AI Models**: 
  - GPT-4 for text analysis and entity extraction
  - GPT-4.1-mini for image generation
  - GPT-4o for vision-based critique
- **Vector Database**: LanceDB for efficient similarity search
- **Embeddings**: 
  - Text embeddings for shot similarity
  - CLIP embeddings for image similarity
- **Image Processing**: PIL/Pillow for image manipulation
- **Async Processing**: Python asyncio for concurrent operations

## Summary

This project creates an intelligent, context-aware storyboard generation system that can adapt to any narrative content while maintaining visual consistency and artistic style. The modular design allows for easy replacement of input data, making it suitable for various storytelling projects, from films to games to graphic novels. The multi-agent architecture ensures high-quality outputs through automated critique and refinement, while the context management system maintains narrative coherence across sequential frames. 