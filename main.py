import os
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple
import hashlib
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# For fallback quality scoring
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available for quality scoring. Install with: pip install opencv-python")

class CarImageProcessor:
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        output_dir: str,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "unsloth/gemma-3-27b-it-bnb-4bit",
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the car image processor.
        
        Args:
            csv_path: Path to the CSV file with car metadata
            images_dir: Directory containing car images
            output_dir: Directory to save outputs
            checkpoint_dir: Directory to save processing checkpoints
            model_name: Model to use for captioning
            batch_size: Batch size for processing
            device: Device to use (cuda/cpu)
        """
        self.csv_path = csv_path
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.batch_size = batch_size
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CSV data
        print("Loading CSV data...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} entries from CSV")
        
        # Initialize models
        self._init_caption_model(model_name)
        self._init_aesthetic_model()
        
        # Load checkpoint if exists
        self.processed_files = self._load_checkpoint()
        
        # System prompt for car descriptions
        self.system_prompt = """You are an expert automotive design consultant with experience in 3D rendering analysis. Your task is to create detailed, technical descriptions of rendered 3D vehicle models shown in the image.
        
You get an additional description of the car which you should use for an accurate description of the vehicle. Focus on:

1. VEHICLE IDENTIFICATION: Brand, model, year range, and body style when recognizable.

2. DESIGN SPECIFICS: Key design elements including body shape, distinctive features, surface treatments, and unique styling cues.

3. VIEWPOINT & PERSPECTIVE: Camera angle (front/rear/side/3-4 view, high/low angle) for spatial consistency.

4. VISUAL DETAILS: Wheels, lighting systems, grilles, badges, and distinctive design elements.

5. COLOR & MATERIALS: Vehicle color, surface finishes (metallic, matte, glossy), and material treatments.

FORMAT REQUIREMENTS:
- Maintain technical precision while being descriptive
- Limit to 300 characters
- Single flowing description without meta-commentary
- Emphasize elements important for image generation models
- Include specific model/brand information when available"""

    def _init_caption_model(self, model_name: str):
        """Initialize the caption generation model."""
        print(f"Loading caption model: {model_name}")
        
        # Check if using quantized model or full model
        if "bnb-4bit" in model_name or "bnb-8bit" in model_name:
            # For quantized models
            self.caption_pipe = pipeline(
                "image-text-to-text",
                model=model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        else:
            # For full precision models
            self.caption_pipe = pipeline(
                "image-text-to-text",
                model=model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
        print("Caption model loaded successfully")

    def _init_aesthetic_model(self):
        """Initialize the aesthetic scoring model."""
        try:
            from aesthetic_scorer import AestheticScorer
            self.aesthetic_scorer = AestheticScorer(device=self.device)
        except Exception as e:
            print(f"Failed to load aesthetic model: {e}")
            print("Using fallback quality scoring")
            self.aesthetic_scorer = None

    def _load_checkpoint(self) -> set:
        """Load processed files from checkpoint."""
        checkpoint_file = self.checkpoint_dir / "processed_files.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                print(f"Loaded checkpoint with {len(data)} processed files")
                return set(data)
        return set()

    def _save_checkpoint(self):
        """Save current progress to checkpoint."""
        checkpoint_file = self.checkpoint_dir / "processed_files.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(list(self.processed_files), f, indent=2)

    def generate_caption(self, image_path: str, description: str) -> str:
        """
        Generate a caption for the car image using the VLM.
        
        Args:
            image_path: Path to the image file
            description: Text description from CSV
            
        Returns:
            Generated caption string
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": f"Additional description: {description}\nPlease provide a detailed, technical description of the vehicle in the image for a text-to-image model."}
                ]
            }
        ]
        
        try:
            output = self.caption_pipe(text=messages, max_new_tokens=300)
            caption = output[0]["generated_text"][-1]["content"]
            return caption.strip()
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return ""

    def estimate_aesthetic_score(self, image_path: str) -> float:
        """
        Estimate the aesthetic score of an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Aesthetic score (0-10 scale)
        """
        if self.aesthetic_scorer is None:
            return 5.0
        try:
            score = self.aesthetic_scorer.score(str(image_path))
            return round(score, 2)
        except Exception as e:
            print(f"Error estimating aesthetic score for {image_path}: {e}")
            return 5.0

    def estimate_clip_hq_car_score(self, image_path: str) -> float:
        """
        Estimate the aesthetic score of an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Aesthetic score (0-10 scale)
        """
        if self.aesthetic_scorer is None:
            return 5.0
        try:
            score = self.aesthetic_scorer.score_clip_similarity(image_path)
            return round(score, 2)
        except Exception as e:
            print(f"Error estimating aesthetic score for {image_path}: {e}")
            return 5.0

    def process_single_image(self, row: pd.Series) -> Optional[Dict]:
        """
        Process a single image: generate caption and aesthetic score.
        
        Args:
            row: DataFrame row with image metadata
            
        Returns:
            Dictionary with processing results or None if error
        """
        filename = row['saved_filename']
        image_path = self.images_dir / filename
        
        # Check if image exists
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return None
        
        # Check if already processed
        if filename in self.processed_files:
            # Try to load existing result
            result_file = self.checkpoint_dir / f"{filename}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    return json.load(f)
        
        try:
            # Prepare description from CSV
            description_parts = []
            if pd.notna(row.get('brand')):
                description_parts.append(f"Brand: {row['brand']}")
            if pd.notna(row.get('model')):
                description_parts.append(f"Model: {row['model']}")
            if pd.notna(row.get('body_style')):
                description_parts.append(f"Body Style: {row['body_style']}")
            if pd.notna(row.get('title')):
                description_parts.append(f"Title: {row['title']}")
            if pd.notna(row.get('description')):
                description_parts.append(f"Details: {row['description'][:500]}")  # Limit description length
            
            full_description = " | ".join(description_parts)
            
            # Generate caption
            print(f"Processing: {filename}")
            caption = self.generate_caption(str(image_path), full_description)
            
            # Estimate aesthetic score
            aesthetic_score = self.estimate_aesthetic_score(str(image_path))

            clip_hq_car_score = self.estimate_clip_hq_car_score(str(image_path))

            # Prepare result
            result = {
                'filename': filename,
                'caption': caption,
                'aesthetic_score': aesthetic_score,
                'clip_hq_car_score': clip_hq_car_score,
                'brand': row.get('brand', ''),
                'model': row.get('model', ''),
                'body_style': row.get('body_style', ''),
                'from_year': row.get('from_year', ''),
                'to_year': row.get('to_year', '')
            }
            
            # Save individual result
            result_file = self.checkpoint_dir / f"{filename}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Update processed files
            self.processed_files.add(filename)
            self._save_checkpoint()
            
            return result
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None

    def process_dataset(self):
        """Process the entire dataset."""
        print(f"\nStarting processing of {len(self.df)} images...")
        print(f"Already processed: {len(self.processed_files)} images")
        
        results = []
        
        # Process each row in the dataframe
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing images"):
            result = self.process_single_image(row)
            if result:
                results.append(result)
            
            # Save intermediate results every 100 images
            if len(results) % 100 == 0 and len(results) > 0:
                self._save_results(results)
        
        # Save final results
        self._save_results(results)
        print(f"\nProcessing complete! Processed {len(results)} images")

    def _save_results(self, results: list):
        """Save results to CSV file."""
        if not results:
            return
            
        output_file = self.output_dir / "car_captions_aesthetic.csv"
        df_results = pd.DataFrame(results)
        
        # Sort by filename for consistency
        df_results = df_results.sort_values('filename')
        
        # Save to CSV
        df_results.to_csv(output_file, index=False)
        print(f"Saved {len(df_results)} results to {output_file}")

    def resume_from_checkpoint(self):
        """Resume processing from checkpoint."""
        print("Resuming from checkpoint...")
        
        # Collect all saved results
        results = []
        for filename in self.processed_files:
            result_file = self.checkpoint_dir / f"{filename}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results.append(json.load(f))
        
        if results:
            self._save_results(results)
        
        # Continue processing remaining images
        self.process_dataset()


def main():
    """Main execution function."""
    # Configuration
    CSV_PATH = "/mnt/damian/Projects/minRF/data/car_images_metadata.csv"  # Path to your CSV file
    IMAGES_DIR = "/mnt/damian/Projects/minRF/data/car_images"  # Directory with car images
    OUTPUT_DIR = "output"  # Output directory
    CHECKPOINT_DIR = "checkpoints"  # Checkpoint directory
    
    # Model selection (you can change based on your resources)
    # Options:
    # - "unsloth/gemma-3-27b-it-bnb-4bit" (large, quantized)
    # - "google/gemma-2-2b-it" (smaller, faster)
    # - "Qwen/Qwen2-VL-2B-Instruct" (alternative VLM)
    MODEL_NAME = "unsloth/gemma-3-27b-it-bnb-4bit"  # Using smaller model for efficiency
    
    resume = True
    
    # Initialize processor
    processor = CarImageProcessor(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        model_name=MODEL_NAME,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Check if resuming from checkpoint
    if processor.processed_files:
        print(f"Found checkpoint with {len(processor.processed_files)} processed files")
        if resume:
            processor.resume_from_checkpoint()
        else:
            processor.process_dataset()
    else:
        processor.process_dataset()


if __name__ == "__main__":
    main()