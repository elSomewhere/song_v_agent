"""
Shared utilities for all agents.
Provides database connections, style loading, image encoding, and reference caching.
"""
import os
import base64
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional
import lancedb
from PIL import Image
import io
import re
from openai import OpenAI

# Initialize LanceDB connection
DB_PATH = Path("vecstore")
DB_PATH.mkdir(exist_ok=True)
db = lancedb.connect(str(DB_PATH))

# Table handles (will be set by ingest scripts)
SHOTS_TBL = None
REFS_TBL = None

def init_tables():
    """Initialize table handles if they exist."""
    global SHOTS_TBL, REFS_TBL
    try:
        SHOTS_TBL = db.open_table("shots")
    except:
        print("Warning: shots table not found. Run ingest/ingest_text.py first.")
    
    try:
        REFS_TBL = db.open_table("refs")
    except:
        print("Warning: refs table not found. Run ingest/ingest_refs.py first.")

# Load style file once
@lru_cache(maxsize=1)
def load_style_guidelines() -> str:
    """Load the style guidelines from style.md file."""
    style_path = Path("data/style.md")
    if not style_path.exists():
        # Try old path for backwards compatibility
        old_path = Path("data/style_primer.md")
        if old_path.exists():
            return old_path.read_text()
        raise FileNotFoundError(f"Style guidelines not found at {style_path}")
    return style_path.read_text()

# Load entities document once
@lru_cache(maxsize=1)
def load_entities_document() -> str:
    """Load the entities document from entites.md file."""
    entities_path = Path("data/entites.md")
    if not entities_path.exists():
        raise FileNotFoundError(f"Entities document not found at {entities_path}")
    return entities_path.read_text()

# Cache the extracted entities
@lru_cache(maxsize=1)
def extract_entities_from_document() -> Dict[str, List[str]]:
    """Extract entity names and their aliases from the entities document using GPT.
    
    Returns:
        Dictionary mapping canonical entity names to their aliases
    """
    entities_text = load_entities_document()
    
    # Use GPT to extract entities dynamically
    client = OpenAI()
    
    prompt = """Analyze the following document and extract all entities (characters, environments, objects, etc.).
For each entity, provide:
1. A canonical name (the main name to use)
2. A list of all possible aliases, variations, or ways this entity might be referenced

Return the result as a Python dictionary where keys are canonical names and values are lists of aliases (including the canonical name in lowercase).

Example format:
{
    "Helena": ["helena", "adult helena", "she", "her"],
    "Silicate Army": ["silicate", "silicate army", "silicate forces", "mechs", "tanks"],
    "Mother Fortress": ["mother", "mother fortress", "fortress", "the fortress"]
}

Document to analyze:
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing documents and extracting entities with their aliases. Return only valid Python dictionary syntax."
                },
                {
                    "role": "user",
                    "content": prompt + entities_text
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        # Extract the dictionary from the response
        response_text = response.choices[0].message.content.strip()
        
        # Find dictionary in response (handling markdown code blocks)
        if "```python" in response_text:
            response_text = response_text.split("```python")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Safely evaluate the dictionary
        import ast
        entities_dict = ast.literal_eval(response_text)
        
        # Ensure all aliases are lowercase and include variations
        normalized_dict = {}
        for canonical, aliases in entities_dict.items():
            # Create a set to avoid duplicates
            alias_set = set()
            
            # Add the canonical name in various forms
            alias_set.add(canonical.lower())
            alias_set.add(canonical)
            
            # Add all provided aliases
            for alias in aliases:
                if isinstance(alias, str):
                    alias_set.add(alias.lower())
                    alias_set.add(alias)
            
            # Add common variations
            if " " in canonical:
                # Add variations without spaces or with underscores
                alias_set.add(canonical.lower().replace(" ", ""))
                alias_set.add(canonical.lower().replace(" ", "_"))
                alias_set.add(canonical.lower().replace(" ", "-"))
            
            normalized_dict[canonical] = list(alias_set)
        
        return normalized_dict
        
    except Exception as e:
        print(f"Error using GPT for entity extraction: {e}")
        print("Falling back to basic extraction...")
        
        # Fallback: Simple extraction based on markdown headers
        entities_dict = {}
        
        # Look for character/entity patterns like "### 1. **Name**"
        pattern = r'###\s+\d+\.\s+\*\*([^*]+)\*\*'
        matches = re.findall(pattern, entities_text)
        
        for match in matches:
            canonical = match.strip()
            # Create basic aliases
            aliases = [
                canonical.lower(),
                canonical,
                canonical.replace(" ", ""),
                canonical.replace(" ", "_")
            ]
            entities_dict[canonical] = list(set(aliases))
        
        return entities_dict

# For backwards compatibility
def load_style_primer() -> str:
    """Load the style primer - alias for load_style_guidelines."""
    return load_style_guidelines()

def to_data_url(image_path: str, max_size: tuple = (1024, 1024)) -> str:
    """Convert an image file to a base64 data URL for GPT-4o.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimensions (width, height) to resize to
        
    Returns:
        Base64 data URL string
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Open and potentially resize image
    img = Image.open(path)
    
    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if too large
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    base64_str = base64.b64encode(img_data).decode('utf-8')
    
    return f"data:image/png;base64,{base64_str}"

# Cache for reference lookups
@lru_cache(maxsize=128)
def get_refs_for(tag: str, k: int = 3) -> List[Dict[str, Any]]:
    """Get reference images for a given tag/entity.
    
    Args:
        tag: Entity tag to search for (e.g., "Helena", "Silicate Army")
        k: Number of references to return
        
    Returns:
        List of reference dictionaries with 'path', 'name', 'tag'
    """
    if REFS_TBL is None:
        init_tables()
        if REFS_TBL is None:
            return []
    
    # Search for references with matching tag
    try:
        results = REFS_TBL.search() \
            .where(f"tag = '{tag}'", prefilter=True) \
            .limit(k) \
            .to_list()
        
        # If no exact match, try case-insensitive search
        if not results:
            # Get all refs and filter manually
            all_refs = REFS_TBL.search().limit(100).to_list()
            results = [r for r in all_refs if r.get('tag', '').lower() == tag.lower()][:k]
        
        return results
    except Exception as e:
        print(f"Error searching refs for tag '{tag}': {e}")
        return []

def get_shot_by_id(shot_id: int) -> Optional[Dict[str, Any]]:
    """Get a shot by its ID."""
    if SHOTS_TBL is None:
        init_tables()
        if SHOTS_TBL is None:
            return None
    
    try:
        results = SHOTS_TBL.search() \
            .where(f"shot_id = {shot_id}", prefilter=True) \
            .limit(1) \
            .to_list()
        
        return results[0] if results else None
    except Exception as e:
        print(f"Error fetching shot {shot_id}: {e}")
        return None

def get_similar_shots(shot_text: str, k: int = 3) -> List[Dict[str, Any]]:
    """Get similar shots based on text similarity.
    
    Args:
        shot_text: Text to find similar shots for
        k: Number of similar shots to return
        
    Returns:
        List of shot dictionaries
    """
    if SHOTS_TBL is None:
        init_tables()
        if SHOTS_TBL is None:
            return []
    
    try:
        # For now, just return some shots
        # In a real implementation, we'd compute embeddings and search
        results = SHOTS_TBL.search().limit(k).to_list()
        return results
    except Exception as e:
        print(f"Error searching similar shots: {e}")
        return []

# Global style reference images
def get_global_style_refs() -> List[str]:
    """Get paths to global style reference images."""
    # Look for any general style references
    style_refs = []
    
    # Check if there are any style reference images in the refs folder
    refs_path = Path("data/refs")
    if refs_path.exists():
        # Look for files with 'style' in the name
        for file in refs_path.glob("**/style*.png"):
            style_refs.append(str(file))
        for file in refs_path.glob("**/style*.jpg"):
            style_refs.append(str(file))
    
    return style_refs

# Initialize tables on import
init_tables()

# Cache for base context documents
@lru_cache(maxsize=1)
def get_base_context_documents() -> Dict[str, str]:
    """Load all base context documents from the data folder.
    
    Returns:
        Dictionary mapping document names to their content
    """
    base_docs = {}
    
    # Load style guidelines
    try:
        base_docs['style_guidelines'] = load_style_guidelines()
    except Exception as e:
        print(f"Warning: Could not load style guidelines: {e}")
    
    # Load entities document
    try:
        base_docs['entities_document'] = load_entities_document()
    except Exception as e:
        print(f"Warning: Could not load entities document: {e}")
    
    # Load script
    try:
        script_path = Path("data/script.md")
        if script_path.exists():
            base_docs['script'] = script_path.read_text()
    except Exception as e:
        print(f"Warning: Could not load script: {e}")
    
    # Load any other markdown files in data folder
    data_path = Path("data")
    if data_path.exists():
        for md_file in data_path.glob("*.md"):
            if md_file.name not in ['script.md', 'style.md', 'entites.md']:
                try:
                    base_docs[f'data_{md_file.stem}'] = md_file.read_text()
                except Exception as e:
                    print(f"Warning: Could not load {md_file}: {e}")
    
    return base_docs

def build_complete_context(historical_context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build complete context including base documents and historical context.
    
    Args:
        historical_context: List of previous generation data
        
    Returns:
        Complete context dictionary
    """
    context = {
        'base_documents': get_base_context_documents(),
        'historical_generations': historical_context or []
    }
    
    # Add a formatted string version for easy inclusion in prompts
    context_parts = []
    
    # Add base documents
    for doc_name, doc_content in context['base_documents'].items():
        if doc_content:
            context_parts.append(f"=== {doc_name.upper()} ===\n{doc_content}\n")
    
    # Add historical context summary
    if historical_context:
        context_parts.append(f"=== PREVIOUS GENERATIONS ===")
        for hist in historical_context[-5:]:  # Show last 5 for summary
            context_parts.append(f"Shot {hist.get('shot_id')}, Variation {hist.get('variation')}: Generated successfully")
    
    context['formatted_context'] = "\n\n".join(context_parts)
    
    return context 