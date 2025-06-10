# AI Storyboard Generator

An AI-powered system that generates professional storyboard frames using GPT-4 and GPT-4.1-mini for image generation.

## Features

- **Director Agent**: Reads shot descriptions and detects entities dynamically from available references
- **Artist Agent**: Creates images using GPT-4.1-mini with contextual understanding and style extraction
- **Critic Agent**: Evaluates images using GPT-4o vision and provides feedback based on dynamically extracted style guidelines
- Generates images with GPT-4.1-mini for better context preservation across multiple frames
- Automatic retry mechanism with critic feedback
- Vector database for efficient reference management
- Parallel processing for faster generation
- Dynamic style extraction from images and text in the data folder

## Overview

This system uses a Director → Artist → Critic agent workflow to automatically generate 16:9 storyboard frames in a gritty anime/manga style. The program:

- Ingests markdown scripts and reference images into a vector database
- Uses LangGraph to orchestrate AI agents
- Generates images with GPT-4.1-mini
- Evaluates quality with GPT-4 vision
- Automatically retries failed attempts
- Supports parallel processing with rate limiting

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```bash
cp env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### 1. Prepare Your Data

Place your story materials in the `data/` folder:
- `data/script.md` - Your markdown script with shot descriptions
- `data/style_primer.md` - Style guidelines for the artwork
- `data/refs/` - Reference images organized by entity (e.g., `refs/Helena/`, `refs/Joy/`)

### 2. Ingest Data

Run the ingestion scripts to build the vector database:

```bash
# Ingest the script (creates shots table)
python ingest/ingest_text.py

# Ingest reference images (creates refs table)
python ingest/ingest_refs.py
```

### 3. Generate Storyboards

Run the main generator:

```bash
python run.py
```

The program will:
- Process shots sequentially (but with parallel workers)
- Generate images using GPT-4.1-mini
- Evaluate each image with GPT-4 vision
- Retry up to 3 times if the critic requests changes
- Save all outputs to the `output/` folder

## Project Structure

```
.
├── data/                    # Input data
│   ├── script.md           # Markdown script with shot descriptions
│   ├── style_primer.md     # Style guidelines
│   └── refs/               # Reference images by entity
├── ingest/                 # Data ingestion scripts
│   ├── ingest_text.py      # Parse script into shots table
│   └── ingest_refs.py      # Process reference images
├── agents/                 # LangGraph agent nodes
│   ├── shared.py           # Shared utilities and DB connections
│   ├── director.py         # Manages workflow and extracts entities dynamically
│   ├── artist.py           # Generates images with GPT-4.1-mini
│   └── critic.py           # Evaluates with GPT-4o vision using dynamic style extraction
├── output/                 # Generated storyboards
│   └── Shot-XXX/           # Per-shot output folder
│       ├── try0.png        # First attempt
│       ├── critic_try0.json # Critic feedback
│       └── prompt.json     # Generation prompt
├── vecstore/               # LanceDB vector database
├── retry_wrap.py           # Retry decorator with exponential backoff
├── run.py                  # Main execution harness
└── requirements.txt        # Python dependencies
```

## Configuration

Edit these values in `run.py`:

- `CALLS_PER_MIN`: OpenAI API rate limit (default: 5)
- `PAR`: Number of parallel workers (default: 2)
- `TOTAL_SHOTS`: Total shots to process (default: 60)

## Output Format

Each shot generates a folder like `output/Shot-001/` containing:

- `tryN.png` - Generated images (N = attempt number)
- `critic_tryN.json` - Critic evaluations
- `prompt.json` - Image generation prompt

## Script Format

Your `script.md` should contain shots formatted like:

```markdown
### **Shot 1**
- **Description:** Wide shot of Helena in white armor...
- **Framing:** Epic wide shot, vast in scale...
- **Camera Angle:** Slightly high-angle from behind...
```

## Troubleshooting

1. **No shots found**: Check your script.md formatting - the parser looks for `### Shot N` or `### **Shot N**` patterns

2. **Reference images not loading**: Ensure images are in subdirectories of `data/refs/` named after entities

3. **Rate limit errors**: Reduce `CALLS_PER_MIN` in run.py

4. **Out of memory**: The CLIP model for image embeddings can be memory intensive. Consider using a smaller model or processing in batches.

## Resume After Interruption

The system automatically resumes from where it left off. If interrupted, simply run `python run.py` again and it will skip already-generated shots.

## Extending the System

- Add new agents by creating nodes in `agents/` and wiring them in `run.py`
- Swap embedding models in the ingest scripts
- Modify the style prompt in `agents/artist.py`
- Adjust retry logic in `agents/critic.py` # song_v_agent
