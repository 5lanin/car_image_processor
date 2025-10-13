import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import os

class ProcessingMonitor:
    """Monitor and manage the processing progress."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", output_dir: str = "output"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
    
    def get_status(self) -> Dict:
        """Get current processing status."""
        status = {
            "total_processed": 0,
            "last_processed": None,
            "checkpoint_exists": False,
            "output_exists": False
        }
        
        # Check checkpoint
        checkpoint_file = self.checkpoint_dir / "processed_files.json"
        if checkpoint_file.exists():
            status["checkpoint_exists"] = True
            with open(checkpoint_file, 'r') as f:
                processed = json.load(f)
                status["total_processed"] = len(processed)
                if processed:
                    status["last_processed"] = processed[-1]
        
        # Check output
        output_file = self.output_dir / "car_captions_aesthetic.csv"
        if output_file.exists():
            status["output_exists"] = True
            df = pd.read_csv(output_file)
            status["output_rows"] = len(df)
        
        return status
    
    def rebuild_output_from_checkpoints(self):
        """Rebuild the output CSV from individual checkpoint files."""
        print("Rebuilding output from checkpoints...")
        
        results = []
        json_files = list(self.checkpoint_dir.glob("*.json"))
        
        # Filter out the main checkpoint file
        json_files = [f for f in json_files if f.name != "processed_files.json"]
        
        print(f"Found {len(json_files)} result files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        
        if results:
            # Create DataFrame and save
            df = pd.DataFrame(results)
            df = df.sort_values('filename')
            output_file = self.output_dir / "car_captions_aesthetic_rebuilt.csv"
            df.to_csv(output_file, index=False)
            print(f"Rebuilt {len(df)} entries saved to {output_file}")
            return df
        else:
            print("No results found to rebuild")
            return None
    
    def verify_completeness(self, csv_path: str):
        """Verify which images from the original CSV have been processed."""
        original_df = pd.read_csv(csv_path)
        original_files = set(original_df['saved_filename'].tolist())
        
        # Get processed files
        checkpoint_file = self.checkpoint_dir / "processed_files.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                processed_files = set(json.load(f))
        else:
            processed_files = set()
        
        # Find unprocessed
        unprocessed = original_files - processed_files
        
        print(f"Original dataset: {len(original_files)} images")
        print(f"Processed: {len(processed_files)} images")
        print(f"Remaining: {len(unprocessed)} images")
        print(f"Progress: {len(processed_files)/len(original_files)*100:.1f}%")
        
        if len(unprocessed) > 0 and len(unprocessed) <= 10:
            print("\nUnprocessed files:")
            for f in list(unprocessed)[:10]:
                print(f"  - {f}")
        
        return {
            "total": len(original_files),
            "processed": len(processed_files),
            "remaining": len(unprocessed),
            "unprocessed_files": list(unprocessed)
        }
    
    def cleanup_orphaned_checkpoints(self):
        """Remove checkpoint files for images not in the processed list."""
        checkpoint_file = self.checkpoint_dir / "processed_files.json"
        if not checkpoint_file.exists():
            print("No checkpoint file found")
            return
        
        with open(checkpoint_file, 'r') as f:
            processed_files = set(json.load(f))
        
        json_files = list(self.checkpoint_dir.glob("*.json"))
        json_files = [f for f in json_files if f.name != "processed_files.json"]
        
        orphaned = []
        for json_file in json_files:
            filename = json_file.stem + ".png"  # Assuming .png extension
            if filename not in processed_files:
                orphaned.append(json_file)
        
        if orphaned:
            print(f"Found {len(orphaned)} orphaned checkpoint files")
            response = input("Delete orphaned files? (y/n): ")
            if response.lower() == 'y':
                for f in orphaned:
                    f.unlink()
                print(f"Deleted {len(orphaned)} files")
        else:
            print("No orphaned files found")
    
    def export_high_quality_subset(self, min_aesthetic_score: float = 7.0):
        """Export a subset of high-quality images based on aesthetic score."""
        output_file = self.output_dir / "car_captions_aesthetic.csv"
        if not output_file.exists():
            print("Output file not found. Run rebuild_output_from_checkpoints() first.")
            return
        
        df = pd.read_csv(output_file)
        high_quality = df[df['aesthetic_score'] >= min_aesthetic_score]
        
        export_file = self.output_dir / f"high_quality_subset_score_{min_aesthetic_score}.csv"
        high_quality.to_csv(export_file, index=False)
        
        print(f"Exported {len(high_quality)} high-quality images (score >= {min_aesthetic_score})")
        print(f"Saved to: {export_file}")
        
        # Print statistics
        print("\nAesthetic Score Statistics:")
        print(f"Mean: {df['aesthetic_score'].mean():.2f}")
        print(f"Median: {df['aesthetic_score'].median():.2f}")
        print(f"Min: {df['aesthetic_score'].min():.2f}")
        print(f"Max: {df['aesthetic_score'].max():.2f}")
        
        return high_quality


def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor car dataset processing")
    parser.add_argument("--status", action="store_true", help="Show processing status")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild output from checkpoints")
    parser.add_argument("--verify", type=str, help="Verify completeness against original CSV")
    parser.add_argument("--cleanup", action="store_true", help="Clean up orphaned checkpoints")
    parser.add_argument("--export-hq", type=float, help="Export high-quality subset (min score)")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    monitor = ProcessingMonitor(args.checkpoint_dir, args.output_dir)
    
    if args.status:
        status = monitor.get_status()
        print("\n=== Processing Status ===")
        for key, value in status.items():
            print(f"{key}: {value}")
    
    if args.rebuild:
        monitor.rebuild_output_from_checkpoints()
    
    if args.verify:
        monitor.verify_completeness(args.verify)
    
    if args.cleanup:
        monitor.cleanup_orphaned_checkpoints()
    
    if args.export_hq:
        monitor.export_high_quality_subset(args.export_hq)


if __name__ == "__main__":
    main()