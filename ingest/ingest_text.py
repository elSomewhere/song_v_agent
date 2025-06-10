"""
Ingest text script - parses markdown script and creates shots table.
"""
import re
from pathlib import Path
from typing import List, Dict, Any
import lancedb
import openai
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embedding model
# Using sentence-transformers instead of OpenAI for cost efficiency
model = SentenceTransformer('all-MiniLM-L6-v2')

def parse_shots_from_markdown(md_path: Path) -> List[Dict[str, Any]]:
    """Parse shots from the markdown script file.
    
    Returns:
        List of shot dictionaries with shot_id and text
    """
    content = md_path.read_text()
    
    # Pattern to match shot headers and their content
    # Looking for patterns like "1. **Shot 1**" or "### Shot 1"
    # Updated to handle numbered list format
    shot_pattern = r'(\d+)\.\s+\*\*Shot\s+(\d+)\*\*\s*\n(.*?)(?=\d+\.\s+\*\*Shot\s+\d+|###|##|$)'
    
    shots = []
    matches = re.finditer(shot_pattern, content, re.DOTALL)
    
    for match in matches:
        shot_id = int(match.group(2))  # Get shot number from **Shot N**
        shot_text = match.group(3).strip()
        
        # Clean up the text - remove excessive whitespace and markdown artifacts
        shot_text = re.sub(r'\n\s*\n', '\n\n', shot_text)
        shot_text = re.sub(r'^\s*-\s*', '', shot_text, flags=re.MULTILINE)  # Remove leading dashes
        
        # Extract the content from the bullet points
        lines = []
        for line in shot_text.split('\n'):
            line = line.strip()
            if line.startswith('- **') and '**:' in line:
                # Extract the label and content
                parts = line.split('**:', 1)
                if len(parts) == 2:
                    label = parts[0].replace('- **', '').strip()
                    content = parts[1].strip()
                    lines.append(f"{label}: {content}")
            elif line:
                lines.append(line)
        
        shot_text = '\n'.join(lines)
        
        shots.append({
            'shot_id': shot_id,
            'text': shot_text
        })
    
    # If no shots found with numbered format, try the old format
    if not shots:
        shot_pattern = r'###\s+\*?\*?Shot\s+(\d+)\*?\*?\s*(?:\([^)]*\))?\s*\n(.*?)(?=###\s+\*?\*?Shot\s+\d+|$)'
        matches = re.finditer(shot_pattern, content, re.DOTALL)
        
        for match in matches:
            shot_id = int(match.group(1))
            shot_text = match.group(2).strip()
            
            # Clean up the text
            shot_text = re.sub(r'\n\s*\n', '\n\n', shot_text)
            shot_text = re.sub(r'- \*\*', '**', shot_text)
            
            shots.append({
                'shot_id': shot_id,
                'text': shot_text
            })
    
    return shots

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        List of embedding vectors
    """
    # Using sentence-transformers for embeddings
    embeddings = model.encode(texts)
    return embeddings.tolist()

def main():
    """Main ingestion function."""
    print("Starting text ingestion...")
    
    # Path to script
    script_path = Path("data/script.md")
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found at {script_path}")
    
    # Parse shots
    shots = parse_shots_from_markdown(script_path)
    print(f"Found {len(shots)} shots in script")
    
    if not shots:
        print("No shots found! Check the markdown format.")
        return
    
    # Create embeddings
    print("Creating embeddings...")
    texts = [shot['text'] for shot in shots]
    embeddings = create_embeddings(texts)
    
    # Prepare data for LanceDB
    data = []
    for i, (shot, embedding) in enumerate(zip(shots, embeddings)):
        data.append({
            'id': i,
            'shot_id': shot['shot_id'],
            'text': shot['text'],
            'vector': embedding
        })
    
    # Create/update LanceDB table
    print("Creating shots table in vector store...")
    db_path = Path("vecstore")
    db_path.mkdir(exist_ok=True)
    db = lancedb.connect(str(db_path))
    
    # Drop existing table if it exists
    try:
        db.drop_table("shots")
    except:
        pass
    
    # Create new table
    shots_table = db.create_table("shots", data=data)
    
    print(f"Successfully ingested {len(data)} shots into vector store")
    
    # Display first few shots as confirmation
    print("\nFirst 3 shots:")
    for shot in data[:3]:
        print(f"  Shot {shot['shot_id']}: {shot['text'][:100]}...")

if __name__ == "__main__":
    main() 