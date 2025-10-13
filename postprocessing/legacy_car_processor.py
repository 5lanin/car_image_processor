import os
import requests
import csv
import pandas as pd
import torch
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import shutil
import uuid
import tempfile
from PIL import Image
import io
import time
import json
import random
import argparse
from typing import List, Tuple, Optional, Dict
import logging

# SAM 2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MODEL PATHS - Update these to your paths
YOLO_MODEL_PATH = '/mnt/damian/Projects/Diffus3D/00_model_checkpoints/yolo11x.pt'
SAM2_CONFIG = "sam2_hiera_l.yaml"  # or sam2_hiera_b+.yaml for smaller model
SAM2_CHECKPOINT = "sam2_hiera_large.pt"  # Download from SAM 2 repo

category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class EnhancedCarProcessor:
    def __init__(self, device="cuda", yolo_conf_threshold=0.25, car_conf_threshold=0.5):
        """
        Enhanced car image processor using YOLO + SAM 2
        
        Args:
            device: Device to run models on
            yolo_conf_threshold: YOLO detection confidence threshold
            car_conf_threshold: Minimum confidence to consider a car detection valid
        """
        self.device = device
        self.yolo_conf_threshold = yolo_conf_threshold
        self.car_conf_threshold = car_conf_threshold
        
        # Initialize YOLO
        logger.info("Loading YOLO model...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
        self.yolo_model.eval()
        
        # Initialize SAM 2
        logger.info("Loading SAM 2 model...")
        self.sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Car-related class IDs (car, bus, truck, motorcycle)
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        logger.info("Models loaded successfully!")
    
    def detect_vehicles_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Detect vehicles in a batch of images using YOLO
        
        Returns list of detection dictionaries with boxes, confidences, and class_ids
        """
        results = self.yolo_model(source=images, conf=self.yolo_conf_threshold, verbose=False)
        
        batch_detections = []
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            
            # Filter for vehicle classes only
            vehicle_mask = np.isin(detections.class_id, list(self.vehicle_classes.keys()))
            
            if vehicle_mask.any():
                filtered_detections = {
                    'boxes': detections.xyxy[vehicle_mask],
                    'confidences': detections.confidence[vehicle_mask],
                    'class_ids': detections.class_id[vehicle_mask],
                    'labels': [category_dict[class_id] for class_id in detections.class_id[vehicle_mask]]
                }
            else:
                filtered_detections = {
                    'boxes': np.array([]),
                    'confidences': np.array([]),
                    'class_ids': np.array([]),
                    'labels': []
                }
            
            batch_detections.append(filtered_detections)
        
        return batch_detections
    
    def segment_cars_with_sam2(self, image: np.ndarray, detections: Dict) -> Tuple[List[np.ndarray], List[float]]:
        """
        Use SAM 2 to segment cars based on YOLO detections
        
        Returns:
            masks: List of binary masks for each detected car
            scores: List of segmentation quality scores
        """
        if len(detections['boxes']) == 0:
            return [], []
        
        # Set image in SAM 2 predictor
        self.sam2_predictor.set_image(image)
        
        masks = []
        scores = []
        
        # Process each detected vehicle
        for i, (box, confidence, class_id) in enumerate(zip(
            detections['boxes'], 
            detections['confidences'], 
            detections['class_ids']
        )):
            # Only process high-confidence car detections
            if confidence < self.car_conf_threshold:
                continue
            
            # Convert box to SAM 2 format (xyxy)
            box_prompt = box.reshape(1, 4)  # SAM expects (N, 4)
            
            # Get segmentation mask
            mask_result, quality_scores, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_prompt,
                multimask_output=False,
                return_logits=False,
            )
            
            if len(mask_result) > 0:
                masks.append(mask_result[0])  # Take the first (and only) mask
                scores.append(quality_scores[0])
        
        return masks, scores
    
    def process_image_batch(
        self, 
        pil_images: List[Image.Image], 
        metadata_batch: List[Dict],
        original_urls: List[str],
        min_segmentation_score: float = 0.8,
        target_size: int = 512
    ) -> List[Dict]:
        """
        Process a batch of images with YOLO + SAM 2
        
        Returns list of processed car data with masks and metadata
        """
        # Convert PIL to numpy arrays for processing
        cv2_images = [cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) for pil_img in pil_images]
        
        # 1. Detect vehicles with YOLO
        logger.info(f"Running YOLO detection on {len(cv2_images)} images...")
        batch_detections = self.detect_vehicles_batch(cv2_images)
        
        processed_results = []
        
        # 2. Process each image with SAM 2
        for i, (image, pil_image, detections, metadata, url) in enumerate(
            zip(cv2_images, pil_images, batch_detections, metadata_batch, original_urls)
        ):
            if len(detections['boxes']) == 0:
                logger.debug(f"No vehicles detected in image {i}")
                continue
            
            logger.info(f"Processing image {i+1}/{len(cv2_images)} - found {len(detections['boxes'])} vehicles")
            
            # Get SAM 2 segmentation masks
            masks, mask_scores = self.segment_cars_with_sam2(image, detections)
            
            if not masks:
                logger.debug(f"No valid masks generated for image {i}")
                continue
            
            # Process each mask
            for j, (mask, score, confidence, label) in enumerate(
                zip(masks, mask_scores, detections['confidences'], detections['labels'])
            ):
                if score < min_segmentation_score:
                    logger.debug(f"Skipping mask {j} with low score: {score:.3f}")
                    continue
                
                # Apply mask to create clean car image
                processed_image = self.apply_mask_and_rescale(
                    pil_image, mask, target_size
                )
                
                if processed_image is not None:
                    result = {
                        'processed_image': processed_image,
                        'metadata': metadata.copy(),
                        'original_url': url,
                        'detection_label': label,
                        'detection_confidence': float(confidence),
                        'segmentation_score': float(score),
                        'mask_index': j
                    }
                    processed_results.append(result)
                    logger.debug(f"Successfully processed car {j} from image {i}")
        
        return processed_results
    
    def apply_mask_and_rescale(
        self, 
        pil_image: Image.Image, 
        mask: np.ndarray, 
        target_size: int = 512
    ) -> Optional[Image.Image]:
        """
        Apply SAM 2 mask to image and rescale to target size
        """
        try:
            # Convert PIL to numpy
            img_array = np.array(pil_image)
            
            # Create RGBA image
            if img_array.shape[2] == 3:
                rgba_array = np.concatenate([
                    img_array, 
                    np.ones((img_array.shape[0], img_array.shape[1], 1), dtype=np.uint8) * 255
                ], axis=2)
            else:
                rgba_array = img_array.copy()
            
            # Apply mask to alpha channel
            rgba_array[:, :, 3] = (mask * 255).astype(np.uint8)
            
            # Convert back to PIL
            masked_image = Image.fromarray(rgba_array, 'RGBA')
            
            # Find bounding box of non-transparent pixels
            alpha = np.array(masked_image.getchannel('A'))
            coords = np.argwhere(alpha > 0)
            
            if len(coords) == 0:
                return None
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Crop to content
            cropped = masked_image.crop((x_min, y_min, x_max, y_max))
            
            # Rescale to target size maintaining aspect ratio
            width, height = cropped.size
            scale_factor = min(target_size / width, target_size / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create final image with transparent background
            final_img = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
            paste_x = (target_size - new_width) // 2
            paste_y = (target_size - new_height) // 2
            final_img.paste(resized, (paste_x, paste_y), resized)
            
            return final_img
            
        except Exception as e:
            logger.error(f"Error applying mask: {e}")
            return None


def download_image(url: str, save_path: str = None, max_retries: int = 3) -> Optional[Image.Image]:
    """Download image with retry logic"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.autoevolution.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                if save_path:
                    img.save(save_path)
                return img
            elif response.status_code == 403:
                logger.warning(f"Access forbidden (403) for {url}, retry {attempt+1}/{max_retries}")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed to download image, status code: {response.status_code}")
                break
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            time.sleep(2 ** attempt)
    
    return None


def process_image_batch_enhanced(
    image_urls: List[str], 
    metadata_rows: List[Dict], 
    processor: EnhancedCarProcessor,
    temp_dir: str, 
    output_dir: str, 
    metadata_file: str, 
    batch_size: int = 16,
    metadata_row_file_limit: int = 25
):
    """
    Process a batch of image URLs with the enhanced pipeline
    """
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    all_images = []
    all_metadata = []
    all_urls = []
    
    logger.info(f"Starting to download images from {len(image_urls)} metadata rows")
    
    # Download images
    for i, (url_list, metadata) in enumerate(zip(image_urls, metadata_rows)):
        urls = [url.strip() for url in url_list.split(',')]
        random.shuffle(urls)
        
        successful_downloads = 0
        
        for j, url in enumerate(urls):
            if successful_downloads >= metadata_row_file_limit:
                break
                
            img = download_image(url)
            
            if img is not None:
                all_images.append(img)
                all_metadata.append(metadata)
                all_urls.append(url)
                successful_downloads += 1
            
            time.sleep(0.5)
        
        if successful_downloads > 0:
            logger.info(f"Row {i}: Downloaded {successful_downloads} images successfully")
        else:
            logger.info(f"Row {i}: Failed to download any images")
            
        time.sleep(1)
    
    if not all_images:
        logger.warning("No images were successfully downloaded in this batch")
        return
    
    logger.info(f"Successfully downloaded {len(all_images)} images in total")
    
    car_metadata = []
    
    # Process images in batches
    for batch_start in range(0, len(all_images), batch_size):
        batch_end = min(batch_start + batch_size, len(all_images))
        batch_images = all_images[batch_start:batch_end]
        batch_metadata = all_metadata[batch_start:batch_end]
        batch_urls = all_urls[batch_start:batch_end]
        
        logger.info(f"Processing enhanced batch {batch_start//batch_size + 1}/{(len(all_images)+batch_size-1)//batch_size} with {len(batch_images)} images")
        
        # Process with enhanced pipeline
        processed_results = processor.process_image_batch(
            batch_images, batch_metadata, batch_urls
        )
        
        # Save processed cars
        for result in processed_results:
            unique_id = str(uuid.uuid4())[:8]
            brand = result['metadata']['brand'].replace(' ', '_')
            model = result['metadata']['model'].replace(' ', '_')
            year = result['metadata']['from_year']
            mask_idx = result['mask_index']
            
            filename = f"{brand}_{model}_{year}_{unique_id}_m{mask_idx}.png"
            output_path = os.path.join(output_dir, filename)
            
            result['processed_image'].save(output_path)
            
            # Prepare metadata entry
            metadata_entry = result['metadata'].copy()
            metadata_entry.update({
                'original_url': result['original_url'],
                'saved_filename': filename,
                'detection_label': result['detection_label'],
                'detection_confidence': result['detection_confidence'],
                'segmentation_score': result['segmentation_score'],
                'processing_method': 'yolo_sam2'
            })
            
            car_metadata.append(metadata_entry)
            logger.info(f"Saved car image: {filename} (conf: {result['detection_confidence']:.3f}, seg: {result['segmentation_score']:.3f})")
    
    # Save metadata
    if car_metadata:
        with open(metadata_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=car_metadata[0].keys())
            if os.stat(metadata_file).st_size == 0:
                writer.writeheader()
            writer.writerows(car_metadata)
        logger.info(f"Saved metadata for {len(car_metadata)} car images")
    else:
        logger.warning("No car images were found in the downloaded images")
    
    logger.info(f"Enhanced batch processing complete. Found {len(car_metadata)} cars from {len(all_images)} images.")


def main():
    parser = argparse.ArgumentParser(description='Enhanced car image processing with YOLO + SAM 2.')
    parser.add_argument('--csv_file', type=str, default='./car_models_3887/data_full.csv', help='Path to the input CSV file.')
    parser.add_argument('--output_dir', type=str, default='./car_images_enhanced', help='Directory to save the processed images.')
    parser.add_argument('--metadata_file', type=str, default='./car_images_enhanced_metadata.csv', help='Path to the output metadata file.')
    parser.add_argument('--temp_dir', type=str, default='./temp_downloads', help='Directory to store temporary downloads.')
    parser.add_argument('--index_batch_size', type=int, default=3, help='Number of metadata rows to process at once.')
    parser.add_argument('--gpu_batch_size', type=int, default=16, help='Number of images to process with models at once.')
    parser.add_argument('--car_conf_threshold', type=float, default=0.5, help='Minimum confidence for car detection.')
    parser.add_argument('--seg_score_threshold', type=float, default=0.8, help='Minimum segmentation quality score.')
    parser.add_argument('--test_batch', action='store_true', help='Process only the first batch as a test.')
    parser.add_argument('--target_size', type=int, default=512, help='Target size for output images.')

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.csv_file)

    # Prepare metadata
    fields = ['brand', 'model', 'from_year', 'to_year', 'body_style', 'segment', 'title', 'description', 'model_url']
    metadata_fields = {field: data[field].tolist() for field in fields if field in data.columns}

    metadata_rows = []
    for i in range(len(data)):
        row = {field: metadata_fields[field][i] for field in fields if field in metadata_fields}
        metadata_rows.append(row)

    image_urls = data['image_urls'].tolist()

    # Initialize enhanced processor
    logger.info("Initializing enhanced car processor...")
    processor = EnhancedCarProcessor(
        car_conf_threshold=args.car_conf_threshold
    )

    # Create directories
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize metadata file
    if not os.path.exists(args.metadata_file):
        with open(args.metadata_file, 'w', newline='') as f:
            pass

    # Check for already processed items
    processed_urls = set()
    if os.path.exists(args.metadata_file) and os.stat(args.metadata_file).st_size > 0:
        processed_data = pd.read_csv(args.metadata_file)
        if 'original_url' in processed_data.columns:
            processed_urls = set(processed_data['original_url'])
            logger.info(f"Found {len(processed_urls)} already processed items")

    # Filter remaining items
    remaining_indices = [
        i for i, url_list in enumerate(image_urls) 
        if not any(url.strip() in processed_urls for url in url_list.split(','))
    ]
    logger.info(f"Processing {len(remaining_indices)} remaining items")

    # Process in batches
    for start_idx in range(0, len(remaining_indices), args.index_batch_size):
        end_idx = min(start_idx + args.index_batch_size, len(remaining_indices))
        batch_indices = remaining_indices[start_idx:end_idx]
        logger.info(f"Processing metadata batch {start_idx//args.index_batch_size + 1}/{(len(remaining_indices)+args.index_batch_size-1)//args.index_batch_size} (rows {batch_indices[0]}-{batch_indices[-1]})")
        
        batch_urls = [image_urls[i] for i in batch_indices]
        batch_metadata = [metadata_rows[i] for i in batch_indices]
        
        process_image_batch_enhanced(
            batch_urls, 
            batch_metadata, 
            processor,
            args.temp_dir, 
            args.output_dir, 
            args.metadata_file, 
            args.gpu_batch_size
        )
        
        if args.test_batch:
            logger.info("Test batch completed. Remove --test_batch to process all images.")
            break
        
        time.sleep(5)

    # Cleanup
    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)

    logger.info(f"Enhanced processing complete. Car images saved to {args.output_dir} with metadata in {args.metadata_file}")


if __name__ == "__main__":
    main()
