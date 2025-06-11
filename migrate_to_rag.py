"""
Migration script to help users transition from the old system to the RAG-based system.
"""
import json
import os
from pathlib import Path
from datetime import datetime
import asyncio
from rich.console import Console
from rich.prompt import Confirm

console = Console()


def check_old_system():
    """Check if the old system exists."""
    old_files = ["run.py", "agents/director.py", "agents/artist.py", "agents/critic.py"]
    return all(Path(f).exists() for f in old_files)


def backup_old_system():
    """Create a backup of the old system."""
    console.print("[yellow]Creating backup of old system...[/yellow]")
    
    backup_dir = Path("backup_old_system")
    backup_dir.mkdir(exist_ok=True)
    
    # Files to backup
    files_to_backup = [
        "run.py",
        "agents/director.py",
        "agents/artist.py", 
        "agents/critic.py",
        "agents/reviewer.py",
        "agents/shared.py"
    ]
    
    for file_path in files_to_backup:
        if Path(file_path).exists():
            dest = backup_dir / file_path
            dest.parent.mkdir(exist_ok=True)
            
            # Read and write file
            with open(file_path, 'rb') as src:
                with open(dest, 'wb') as dst:
                    dst.write(src.read())
            
            console.print(f"  ✓ Backed up {file_path}")
    
    console.print("[green]Backup complete![/green]")
    return True


def check_dependencies():
    """Check if new dependencies are installed."""
    console.print("\n[yellow]Checking dependencies...[/yellow]")
    
    missing = []
    
    try:
        import lancedb
    except ImportError:
        missing.append("lancedb")
    
    try:
        import pydantic
    except ImportError:
        missing.append("pydantic")
    
    try:
        import pyarrow
    except ImportError:
        missing.append("pyarrow")
    
    if missing:
        console.print(f"[red]Missing dependencies: {', '.join(missing)}[/red]")
        console.print("Please run: pip install -r requirements.txt")
        return False
    
    console.print("[green]All dependencies installed![/green]")
    return True


def migrate_output_metadata():
    """Convert old output metadata to new format."""
    console.print("\n[yellow]Migrating output metadata...[/yellow]")
    
    output_dir = Path("output")
    if not output_dir.exists():
        console.print("No output directory found, skipping metadata migration")
        return
    
    migrated = 0
    for run_dir in output_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            # Look for old metadata files
            old_metadata_files = list(run_dir.glob("*_metadata.json"))
            
            for old_file in old_metadata_files:
                try:
                    with open(old_file) as f:
                        old_data = json.load(f)
                    
                    # Convert to new format
                    new_metadata = {
                        "timestamp": old_data.get("timestamp", datetime.now().isoformat()),
                        "configuration": {
                            "start_shot": old_data.get("shot_id", 1),
                            "max_shots": 1,
                            "variations": 1,
                            "max_retries": old_data.get("attempt", 1),
                            "memory_k": 5,
                            "memory_weights": {"semantic": 0.6, "entity": 0.3, "temporal": 0.1},
                            "quality_threshold": 0.7
                        },
                        "results": [{
                            "shot_id": old_data.get("shot_id", 0),
                            "status": "pass" if old_data.get("status") == "pass" else "fail",
                            "image_paths": [str(old_file.parent / old_data.get("filename", ""))] if old_data.get("filename") else [],
                            "quality_scores": [0.8] if old_data.get("status") == "pass" else [0.5]
                        }]
                    }
                    
                    # Save new metadata
                    new_file = run_dir / "metadata.json"
                    with open(new_file, 'w') as f:
                        json.dump(new_metadata, f, indent=2)
                    
                    migrated += 1
                    
                except Exception as e:
                    console.print(f"[red]Error migrating {old_file}: {e}[/red]")
    
    console.print(f"[green]Migrated {migrated} metadata files[/green]")


def create_compatibility_script():
    """Create a compatibility script for old commands."""
    console.print("\n[yellow]Creating compatibility script...[/yellow]")
    
    compat_script = '''#!/usr/bin/env python
"""
Compatibility wrapper for old run.py commands.
Automatically translates to new run_rag.py commands.
"""
import sys
import subprocess

# Map old arguments to new ones
print("Note: run.py is deprecated. Translating to run_rag.py...")

# Replace run.py with run_rag.py in command
args = sys.argv[1:]
cmd = ["python", "run_rag.py"] + args

# Execute new command
subprocess.run(cmd)
'''
    
    # Rename old run.py if it exists
    if Path("run.py").exists() and not Path("run_old.py").exists():
        Path("run.py").rename("run_old.py")
        console.print("  ✓ Renamed run.py to run_old.py")
    
    # Create new compatibility script
    with open("run.py", 'w') as f:
        f.write(compat_script)
    
    # Make it executable on Unix-like systems
    try:
        os.chmod("run.py", 0o755)
    except:
        pass
    
    console.print("[green]Compatibility script created![/green]")


def print_migration_summary():
    """Print a summary of the migration."""
    console.print("\n[bold green]Migration Complete![/bold green]\n")
    
    console.print("Summary of changes:")
    console.print("1. Old system backed up to 'backup_old_system/' directory")
    console.print("2. New RAG-based system installed alongside old system")
    console.print("3. Compatibility wrapper created for old commands")
    console.print("4. Output metadata migrated to new format")
    
    console.print("\n[bold cyan]Next steps:[/bold cyan]")
    console.print("1. Test the new system: python test_rag.py")
    console.print("2. Run your first generation: python run_rag.py --max-shots 3")
    console.print("3. Compare results with old system")
    console.print("4. Adjust memory settings if needed: --memory-k 10 --memory-weights '0.7,0.2,0.1'")
    
    console.print("\n[yellow]Note:[/yellow] Your old commands will still work!")
    console.print("Example: python run.py --shot-id 1 --max-shots 10")
    console.print("Will automatically use the new system")


async def main():
    """Main migration process."""
    console.print("[bold blue]RAG Storyboard System Migration Tool[/bold blue]\n")
    
    # Check if this is an upgrade
    if not check_old_system():
        console.print("[yellow]No old system detected. This appears to be a fresh installation.[/yellow]")
        console.print("You can start using the new system directly with: python run_rag.py")
        return
    
    console.print("This tool will help you migrate from the old storyboard system to the new RAG-based system.\n")
    
    if not Confirm.ask("Do you want to proceed with the migration?"):
        console.print("Migration cancelled.")
        return
    
    # Step 1: Backup old system
    if not backup_old_system():
        console.print("[red]Backup failed. Migration cancelled.[/red]")
        return
    
    # Step 2: Check dependencies
    if not check_dependencies():
        console.print("[red]Please install missing dependencies before continuing.[/red]")
        return
    
    # Step 3: Migrate metadata
    migrate_output_metadata()
    
    # Step 4: Create compatibility script
    create_compatibility_script()
    
    # Step 5: Print summary
    print_migration_summary()


if __name__ == "__main__":
    asyncio.run(main()) 