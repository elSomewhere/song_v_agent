# RAG-Based Storyboard Generator

A next-generation AI storyboard generation system that uses semantic memory retrieval (RAG) instead of chronological context windows, resulting in ~80% token savings and improved consistency.

## Key Features

- **Semantic Memory Retrieval**: Retrieves contextually relevant scenes based on similarity, not just recency
- **Three-Tier Memory Architecture**: Canonical (static), Episodic (historical), and Working (active) memory
- **Token Efficient**: Constant memory usage regardless of story length
- **Backward Compatible**: Maintains the same CLI and file structure as the original system
- **GPT-4o Native**: Leverages GPT-4o's contextual image generation capabilities

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Basic Usage

```bash
python run_rag.py
```

### Advanced Usage

```bash
# Generate 5 variations per shot with custom memory settings
python run_rag.py --variations 5 --memory-k 10 --memory-weights "0.7,0.2,0.1"

# High quality mode with stricter thresholds
python run_rag.py --quality-threshold 0.85 --max-retries 5 --save-metadata

# Process specific shot range
python run_rag.py --shot-id 10 --max-shots 20
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--shot-id` | 1 | Starting shot ID |
| `--max-shots` | 60 | Maximum number of shots to process |
| `--variations` | 3 | Number of variations per shot |
| `--max-retries` | 3 | Maximum retries per image |
| `--memory-k` | 5 | Number of relevant memories to retrieve |
| `--memory-weights` | "0.6,0.3,0.1" | Weights for semantic,entity,temporal |
| `--quality-threshold` | 0.7 | Minimum quality score to accept |
| `--save-metadata` | False | Save detailed generation metadata |

## Memory Retrieval Algorithm

The system uses a weighted combination of three factors to retrieve relevant memories:

```
relevance_score = 0.6 × semantic_similarity + 
                  0.3 × entity_overlap + 
                  0.1 × temporal_proximity
```

These weights can be adjusted via `--memory-weights` for different types of stories.

## Architecture

### Agents

1. **Planner**: Analyzes scenes and creates structured generation plans
2. **Renderer**: Generates images using GPT-4o with canonical expansions
3. **Vision QA**: Validates images against requirements and provides feedback

### Memory System

- **Canonical Memory**: Loads style.md and entities.md once at startup
- **Episodic Memory**: Stores all successful generations in LanceDB
- **Working Memory**: Dynamically retrieves the K most relevant memories

## File Structure

```
project_root/
├── run_rag.py           # Main entry point (new)
├── run.py               # Original system (preserved)
├── agents/              # New agent implementations
├── memory/              # Memory management system
├── workflow/            # LangGraph workflow
├── utils/               # Utilities
├── data/                # Input data (unchanged)
│   ├── script.md
│   ├── style.md
│   └── entities.md
└── output/              # Generated outputs (unchanged)
```

## Performance Comparison

| Metric | Old System | New System |
|--------|------------|------------|
| Tokens per scene (after 50) | ~15,000 | ~3,000 |
| API calls per scene | 4-6 | 3 |
| Memory growth | O(n) | O(1) |
| Context relevance | Time-based | Semantic |

## Migration from Original System

The new system is a drop-in replacement:

1. Keep your existing `/data` folder
2. Your `/output` folder structure remains the same
3. The CLI is backward compatible (old flags still work)
4. Run `python run_rag.py` instead of `python run.py`

## Troubleshooting

### "No scenes found"
- Ensure your script.md follows the format: `Shot N: Description`

### High token usage on first runs
- The system needs to build up episodic memory; efficiency improves after ~5 scenes

### Images lack consistency
- Try increasing `--memory-k` to retrieve more context
- Adjust memory weights to prioritize entity consistency: `--memory-weights "0.4,0.5,0.1"`

## License

Same as original project 