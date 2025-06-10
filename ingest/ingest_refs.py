"""
Ingest reference images - processes images and creates refs table with embeddings.
"""
from pathlib import Path
from typing import List, Dict, Any
import lancedb
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

# Initialize CLIP model for image embeddings
# Using sentence-transformers CLIP model
model = SentenceTransformer('clip-ViT-B-32')

def scan_reference_images(refs_path: Path) -> List[Dict[str, Any]]:
    """Scan the refs directory for images and categorize them.
    
    Returns:
        List of reference dictionaries with path, name, and tag
    """
    refs = []
    
    # Scan each subdirectory
    for subdir in refs_path.iterdir():
        if subdir.is_dir():
            tag = subdir.name  # Directory name is the tag
            
            # Find all image files in this directory
            for img_file in subdir.glob("*"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    refs.append({
                        'path': str(img_file),
                        'name': img_file.stem,
                        'tag': tag
                    })
    
    return refs

def create_image_embeddings(image_paths: List[str]) -> List[List[float]]:
    """Create CLIP embeddings for a list of images.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of embedding vectors
    """
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Create a blank image as fallback
            images.append(Image.new('RGB', (224, 224), color='white'))
    
    # Encode images using CLIP
    embeddings = model.encode(images)
    return embeddings.tolist()

def main():
    """Main ingestion function."""
    print("Starting reference image ingestion...")
    
    # Path to refs
    refs_path = Path("data/refs")
    if not refs_path.exists():
        raise FileNotFoundError(f"Reference images not found at {refs_path}")
    
    # Scan for images
    refs = scan_reference_images(refs_path)
    print(f"Found {len(refs)} reference images")
    
    if not refs:
        print("No reference images found!")
        return
    
    # Display found references
    print("\nFound references:")
    tags = {}
    for ref in refs:
        tag = ref['tag']
        tags[tag] = tags.get(tag, 0) + 1
    
    for tag, count in tags.items():
        print(f"  {tag}: {count} images")
    
    # Create embeddings
    print("\nCreating image embeddings...")
    image_paths = [ref['path'] for ref in refs]
    embeddings = create_image_embeddings(image_paths)
    
    # Prepare data for LanceDB
    data = []
    for i, (ref, embedding) in enumerate(zip(refs, embeddings)):
        data.append({
            'id': ref['path'],  # Use path as unique ID
            'path': ref['path'],
            'name': ref['name'],
            'tag': ref['tag'],
            'vector': embedding
        })
    
    # Create/update LanceDB table
    print("Creating refs table in vector store...")
    db_path = Path("vecstore")
    db_path.mkdir(exist_ok=True)
    db = lancedb.connect(str(db_path))
    
    # Drop existing table if it exists
    try:
        db.drop_table("refs")
    except:
        pass
    
    # Create new table
    refs_table = db.create_table("refs", data=data)
    
    print(f"Successfully ingested {len(data)} reference images into vector store")

if __name__ == "__main__":
    main() 