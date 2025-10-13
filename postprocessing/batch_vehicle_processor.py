#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import logging
from ultralytics import YOLO
import supervision as sv
from typing import List, Tuple, Optional, Dict, Union
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Real-ESRGAN import failed: {e}")
    print("Super resolution will use fallback cv2 upscaling")
    REALESRGAN_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLO Vehicle classes
VEHICLE_CLASSES = {2: 'car', 5: 'bus', 7: 'truck'}

class BatchVehicleProcessor:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        segmentation_method="sam2",
        yolo_conf_threshold=0.25,
        car_conf_threshold=0.5,
        enable_super_resolution=True,
        sr_threshold_size=256,
        bbox_expansion=0.2,
        region_size=800,  # Standardized region size
        final_size=512,   # Final output size
        batch_size=4,     # Default batch size for SAM2 processing
        yolo_batch_size=8 # Default batch size for YOLO processing
    ):
        """
        Batch vehicle processor with uniform region scaling and dual segmentation method support
        
        Args:
            device: Device to run models on
            segmentation_method: Segmentation method ('sam2' or 'yolo'). Default 'sam2' for backward compatibility
            yolo_conf_threshold: YOLO detection confidence threshold
            car_conf_threshold: Minimum confidence for vehicle detections
            enable_super_resolution: Whether to use super resolution for small regions
            sr_threshold_size: Apply SR if region smaller than this
            bbox_expansion: Factor to expand bounding boxes
            region_size: Standardized size for extracted regions (800x800)
            final_size: Final output size (512x512)
            batch_size: Number of regions to process simultaneously (SAM2)
            yolo_batch_size: Number of images to process simultaneously (YOLO), use -1 for auto-sizing
        """
        # Validate segmentation method
        if segmentation_method not in ['sam2', 'yolo']:
            raise ValueError(f"Invalid segmentation_method: {segmentation_method}. Must be 'sam2' or 'yolo'")
        
        self.device = device
        self.segmentation_method = segmentation_method
        self.yolo_conf_threshold = yolo_conf_threshold
        self.car_conf_threshold = car_conf_threshold
        self.enable_super_resolution = enable_super_resolution
        self.sr_threshold_size = sr_threshold_size
        self.bbox_expansion = bbox_expansion
        self.region_size = region_size
        self.final_size = final_size
        self.batch_size = batch_size
        self.yolo_batch_size = yolo_batch_size
        
        print(f"Initializing batch vehicle processor on device: {device}")
        print(f"Segmentation method: {segmentation_method}")
        print(f"Region standardization: {region_size}x{region_size}")
        print(f"Final output: {final_size}x{final_size}")
        if segmentation_method == 'sam2':
            print(f"SAM2 batch size: {batch_size}")
        print(f"YOLO batch size: {yolo_batch_size}")
        
        # Load models
        self._load_models()
        
        print("Batch vehicle processor initialized successfully!")
    
    def _load_models(self):
        """Load models based on segmentation method"""
        if self.segmentation_method == 'sam2':
            # Load YOLO detection model for SAM2 method
            print("Loading YOLO detection model...")
            self.yolo_model = YOLO('yolo11x.pt').to(self.device)
            self.yolo_model.eval()
            
            # Load SAM2 model
            print("Loading SAM2 model...")
            checkpoint_path = "/mnt/damian/Projects/sam2/checkpoints/sam2.1_hiera_large.pt"
            sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
            self.sam2_model = build_sam2(sam2_config, checkpoint_path, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
            
        elif self.segmentation_method == 'yolo':
            # Load YOLO segmentation model for YOLO method
            print("Loading YOLO segmentation model...")
            self.yolo_model = YOLO('yolo11x-seg.pt').to(self.device)
            self.yolo_model.eval()
            
            # No SAM2 model needed for YOLO segmentation
            self.sam2_model = None
            self.sam2_predictor = None
        
        # Load Real-ESRGAN model if enabled
        if self.enable_super_resolution and REALESRGAN_AVAILABLE:
            print("Loading Real-ESRGAN model...")
            self._setup_super_resolution()
        else:
            print("Using fallback super resolution")
    
    def _setup_super_resolution(self):
        """Initialize Real-ESRGAN for super resolution"""
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'postprocessing/weights/RealESRGAN_x4plus.pth'
        
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            gpu_id=0 if self.device == 'cuda' else None
        )
        print("Real-ESRGAN initialized successfully!")
    
    def expand_bbox(self, box: np.ndarray, image_shape: tuple) -> np.ndarray:
        """Expand bounding box by expansion factor while keeping within bounds"""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        expand_w = width * self.bbox_expansion
        expand_h = height * self.bbox_expansion
        
        x1 = max(0, x1 - expand_w / 2)
        y1 = max(0, y1 - expand_h / 2)
        x2 = min(image_shape[1], x2 + expand_w / 2)
        y2 = min(image_shape[0], y2 + expand_h / 2)
        
        return np.array([x1, y1, x2, y2])
    
    def _parse_batch_detection_results(self, batch_results, images: List[np.ndarray]) -> List[Dict]:
        """
        Parse YOLO batch detection results back to per-image format
        
        Args:
            batch_results: Results from YOLO batch processing
            images: Original list of images for shape reference
            
        Returns:
            List of detection dictionaries (same format as original batch_detect_vehicles)
        """
        all_detections = []
        
        # Process each image's results
        for i, (result, image) in enumerate(zip(batch_results, images)):
            detections = sv.Detections.from_ultralytics(result)
            
            # Filter for vehicle classes only
            vehicle_mask = np.isin(detections.class_id, list(VEHICLE_CLASSES.keys()))
            
            if vehicle_mask.any():
                # Expand bounding boxes
                expanded_boxes = []
                for box in detections.xyxy[vehicle_mask]:
                    expanded_box = self.expand_bbox(box, image.shape[:2])
                    expanded_boxes.append(expanded_box)
                
                image_detections = {
                    'boxes': np.array(expanded_boxes),
                    'original_boxes': detections.xyxy[vehicle_mask],
                    'confidences': detections.confidence[vehicle_mask],
                    'class_ids': detections.class_id[vehicle_mask],
                    'labels': [VEHICLE_CLASSES[cid] for cid in detections.class_id[vehicle_mask]]
                }
            else:
                image_detections = {
                    'boxes': np.array([]), 'original_boxes': np.array([]),
                    'confidences': np.array([]), 'class_ids': np.array([]), 'labels': []
                }
            
            all_detections.append(image_detections)
        
        return all_detections
    
    def apply_super_resolution(self, image: np.ndarray, input_format='BGR') -> np.ndarray:
        """Apply super resolution to small regions"""
        if not self.enable_super_resolution:
            return image
        
        try:
            if REALESRGAN_AVAILABLE and hasattr(self, 'upsampler'):
                # Use Real-ESRGAN
                if input_format == 'RGB' and len(image.shape) == 3 and image.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif input_format == 'BGR':
                    image_bgr = image
                else:
                    image_bgr = image
                
                output_bgr, _ = self.upsampler.enhance(image_bgr, outscale=4)
                
                if input_format == 'RGB' and len(image.shape) == 3 and image.shape[2] == 3:
                    output = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
                else:
                    output = output_bgr
                
                return output
            else:
                # Fallback: cv2 upscaling
                h, w = image.shape[:2]
                new_h, new_w = h * 2, w * 2
                
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                
                filtered = cv2.bilateralFilter(image, 9, 75, 75)
                upscaled = cv2.resize(filtered, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(upscaled, -1, kernel)
                output = cv2.addWeighted(upscaled, 0.7, sharpened, 0.3, 0)
                
                return output
                
        except Exception as e:
            logger.error(f"Super resolution failed: {e}")
            return image
    
    def load_images_batch(self, image_paths: List[str]) -> List[Tuple[np.ndarray, str]]:
        """Load multiple images efficiently"""
        loaded_images = []
        
        def load_single_image(path: str) -> Optional[Tuple[np.ndarray, str]]:
            try:
                pil_image = Image.open(path).convert('RGB')
                cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                return cv2_image, path
            except Exception as e:
                logger.error(f"Failed to load image {path}: {e}")
                return None
        
        # Use ThreadPoolExecutor for I/O optimization
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_path = {executor.submit(load_single_image, path): path for path in image_paths}
            
            for future in as_completed(future_to_path):
                result = future.result()
                if result is not None:
                    loaded_images.append(result)
        
        print(f"Successfully loaded {len(loaded_images)}/{len(image_paths)} images")
        return loaded_images
    
    def batch_detect_vehicles(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Detect vehicles in multiple images using TRUE batch processing
        
        Args:
            images: List of images to process
            
        Returns:
            List of detection dictionaries (same format as before, but faster!)
        """
        if not images:
            return []
        
        # Get optimal batch size for this set of images
        effective_batch_size = self._get_adaptive_yolo_batch_size(images)
        
        # Handle large batches by chunking
        all_detections = []
        
        for i in range(0, len(images), effective_batch_size):
            batch_images = images[i:i + effective_batch_size]
            
            try:
                # TRUE BATCH PROCESSING: Process all images in this chunk simultaneously
                batch_results = self.yolo_model(
                    source=batch_images,
                    conf=self.yolo_conf_threshold,
                    verbose=False,
                    device=self.device
                )
                
                # Parse batch results back to per-image format
                chunk_detections = self._parse_batch_detection_results(batch_results, batch_images)
                all_detections.extend(chunk_detections)
                
            except Exception as e:
                # Fallback to sequential processing if batch fails (e.g., memory issues)
                logger.warning(f"Batch processing failed for chunk {i//effective_batch_size + 1}, falling back to sequential: {e}")
                
                for image in batch_images:
                    try:
                        results = self.yolo_model(source=image, conf=self.yolo_conf_threshold, verbose=False)
                        chunk_detections = self._parse_batch_detection_results([results[0]], [image])
                        all_detections.extend(chunk_detections)
                    except Exception as e2:
                        logger.error(f"Failed to process image in fallback mode: {e2}")
                        # Add empty detection for failed image
                        all_detections.append({
                            'boxes': np.array([]), 'original_boxes': np.array([]),
                            'confidences': np.array([]), 'class_ids': np.array([]), 'labels': []
                        })
        
        return all_detections
    
    def _parse_yolo_segmentation_results(self, batch_results, images: List[np.ndarray]) -> List[Dict]:
        """
        Parse YOLO segmentation results to extract both detection and segmentation data
        
        Args:
            batch_results: Results from YOLO segmentation batch processing  
            images: Original list of images for shape reference
            
        Returns:
            List of detection and segmentation dictionaries with masks included
        """
        all_results = []
        
        # Process each image's results
        for i, (result, image) in enumerate(zip(batch_results, images)):
            if result.boxes is not None and len(result.boxes) > 0:
                # Filter for vehicle classes only
                vehicle_mask = np.isin(result.boxes.cls.cpu().numpy(), list(VEHICLE_CLASSES.keys()))
                
                if vehicle_mask.any():
                    # Extract vehicle detections
                    vehicle_boxes = result.boxes.xyxy[vehicle_mask].cpu().numpy()
                    vehicle_confidences = result.boxes.conf[vehicle_mask].cpu().numpy()
                    vehicle_class_ids = result.boxes.cls[vehicle_mask].cpu().numpy()
                    vehicle_labels = [VEHICLE_CLASSES[int(cid)] for cid in vehicle_class_ids]
                    
                    # Extract vehicle masks if available
                    vehicle_masks = None
                    if result.masks is not None and len(result.masks.data) > 0:
                        vehicle_masks = result.masks.data[vehicle_mask].cpu().numpy()
                    
                    # Filter by confidence threshold
                    conf_mask = vehicle_confidences >= self.car_conf_threshold
                    if conf_mask.any():
                        image_results = {
                            'boxes': vehicle_boxes[conf_mask],
                            'confidences': vehicle_confidences[conf_mask],
                            'class_ids': vehicle_class_ids[conf_mask],
                            'labels': [vehicle_labels[j] for j, keep in enumerate(conf_mask) if keep],
                            'masks': vehicle_masks[conf_mask] if vehicle_masks is not None else None
                        }
                    else:
                        image_results = {
                            'boxes': np.array([]), 'confidences': np.array([]),
                            'class_ids': np.array([]), 'labels': [], 'masks': None
                        }
                else:
                    image_results = {
                        'boxes': np.array([]), 'confidences': np.array([]),
                        'class_ids': np.array([]), 'labels': [], 'masks': None
                    }
            else:
                image_results = {
                    'boxes': np.array([]), 'confidences': np.array([]),
                    'class_ids': np.array([]), 'labels': [], 'masks': None
                }
            
            all_results.append(image_results)
        
        return all_results
    
    def batch_detect_and_segment_vehicles_yolo(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Detect and segment vehicles in multiple images using YOLO segmentation model
        
        Args:
            images: List of images to process
            
        Returns:
            List of detection and segmentation dictionaries with masks included
        """
        if not images:
            return []
        
        # Get optimal batch size for this set of images
        effective_batch_size = self._get_adaptive_yolo_batch_size(images)
        
        # Handle large batches by chunking
        all_results = []
        
        for i in range(0, len(images), effective_batch_size):
            batch_images = images[i:i + effective_batch_size]
            
            try:
                # TRUE BATCH PROCESSING: Process all images in this chunk simultaneously
                # YOLO segmentation model provides both detection and segmentation in one pass
                
                # Use mixed precision for better performance if available
                if self.device == 'cuda':
                    with torch.amp.autocast():
                        batch_results = self.yolo_model(
                            source=batch_images,
                            conf=self.yolo_conf_threshold,
                            verbose=False,
                            device=self.device
                        )
                else:
                    batch_results = self.yolo_model(
                        source=batch_images,
                        conf=self.yolo_conf_threshold,
                        verbose=False,
                        device=self.device
                    )
                
                # Parse batch results to extract both detection and segmentation data
                chunk_results = self._parse_yolo_segmentation_results(batch_results, batch_images)
                all_results.extend(chunk_results)
                
                # Clear GPU cache between batch operations to prevent memory accumulation
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                # Fallback to sequential processing if batch fails (e.g., memory issues)
                logger.warning(f"Batch YOLO segmentation failed for chunk {i//effective_batch_size + 1}, falling back to sequential: {e}")
                
                for image in batch_images:
                    try:
                        results = self.yolo_model(source=image, conf=self.yolo_conf_threshold, verbose=False)
                        chunk_results = self._parse_yolo_segmentation_results([results[0]], [image])
                        all_results.extend(chunk_results)
                    except Exception as e2:
                        logger.error(f"Failed to process image in fallback mode: {e2}")
                        # Add empty result for failed image
                        all_results.append({
                            'boxes': np.array([]), 'confidences': np.array([]),
                            'class_ids': np.array([]), 'labels': [], 'masks': None
                        })
        
        return all_results
    
    def _get_adaptive_yolo_batch_size(self, images: List[np.ndarray]) -> int:
        """
        Determine optimal YOLO batch size based on image dimensions, available GPU memory, and segmentation method
        
        Args:
            images: List of images to analyze
            
        Returns:
            Optimal batch size for YOLO processing
        """
        if self.yolo_batch_size == -1:
            # Auto-batch sizing based on image sizes and available memory
            if not images:
                return 8 if self.segmentation_method == 'sam2' else 4  # Default fallback
            
            # Estimate memory usage based on image dimensions
            avg_pixels = np.mean([img.shape[0] * img.shape[1] for img in images])
            
            # Segmentation models require more memory than detection-only models
            if self.segmentation_method == 'yolo':
                # More conservative batch sizes for segmentation models
                if avg_pixels < 640 * 480:  # Small images
                    return min(8, len(images))
                elif avg_pixels < 1280 * 720:  # Medium images
                    return min(4, len(images))
                else:  # Large images
                    return min(2, len(images))
            else:
                # Standard batch sizes for detection-only models (current behavior)
                if avg_pixels < 640 * 480:  # Small images
                    return min(16, len(images))
                elif avg_pixels < 1280 * 720:  # Medium images
                    return min(8, len(images))
                else:  # Large images
                    return min(4, len(images))
        else:
            # Use user-specified batch size, but don't exceed number of images
            return min(self.yolo_batch_size, len(images))
    
    def extract_context_region(self, image: np.ndarray, bbox: np.ndarray, context_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a larger context region around bbox for smart padding
        
        Args:
            image: Full original image
            bbox: Bounding box [x1, y1, x2, y2]
            context_factor: Additional context as fraction of bbox size
            
        Returns:
            context_region: Extracted context region
            context_bbox: Coordinates of context region in original image
        """
        x1, y1, x2, y2 = bbox.astype(int)
        width = x2 - x1
        height = y2 - y1
        
        # Add context padding
        context_w = int(width * context_factor)
        context_h = int(height * context_factor)
        
        # Calculate context region bounds
        ctx_x1 = max(0, x1 - context_w)
        ctx_y1 = max(0, y1 - context_h)
        ctx_x2 = min(image.shape[1], x2 + context_w)
        ctx_y2 = min(image.shape[0], y2 + context_h)
        
        context_region = image[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
        context_bbox = np.array([ctx_x1, ctx_y1, ctx_x2, ctx_y2])
        
        return context_region, context_bbox
    
    def standardize_region_with_smart_padding(
        self,
        image: np.ndarray,           # Full original image
        original_bbox: np.ndarray,   # Tight YOLO detection
        expanded_bbox: np.ndarray,   # Expanded for context
        target_size: int = 800,
        apply_sr: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        ENHANCED: Standardize region with smart padding and dual bbox tracking
        
        Key improvements:
        1. Smart padding using original image content instead of black borders
        2. Track both original and expanded bbox coordinates in 800x800 space
        3. Enable precise SAM2 prompting with original bbox only
        
        Args:
            image: Full original image
            original_bbox: Tight YOLO detection bbox
            expanded_bbox: Expanded bbox for extraction context
            target_size: Target standardized size (800x800)
            apply_sr: Whether to apply super resolution
            
        Returns:
            standardized_region: 800x800 image with smart padding
            transform_info: Comprehensive coordinate mapping information
        """
        
        # Extract the expanded region (for context)
        ex_x1, ex_y1, ex_x2, ex_y2 = expanded_bbox.astype(int)
        expanded_region = image[ex_y1:ex_y2, ex_x1:ex_x2]
        
        # Try to get larger context for smart padding
        context_region, context_bbox = self.extract_context_region(image, expanded_bbox, context_factor=0.5)
        
        # Calculate scale for expanded region to fit in target_size
        eh, ew = expanded_region.shape[:2]
        original_size = max(eh, ew)
        
        # Apply super resolution if needed
        if apply_sr and original_size < self.sr_threshold_size:
            expanded_region = self.apply_super_resolution(expanded_region, input_format='BGR')
            eh, ew = expanded_region.shape[:2]
            # Also scale context if SR was applied
            if context_region.shape[0] > 0 and context_region.shape[1] > 0:
                context_region = self.apply_super_resolution(context_region, input_format='BGR')
        
        scale = target_size / max(eh, ew)
        new_eh = int(eh * scale)
        new_ew = int(ew * scale)
        
        # Resize expanded region
        resized_expanded = cv2.resize(expanded_region, (new_ew, new_eh), interpolation=cv2.INTER_LANCZOS4)
        
        # Create target_size x target_size canvas
        standardized = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calculate centering offsets for expanded region
        y_offset = (target_size - new_eh) // 2
        x_offset = (target_size - new_ew) // 2
        
        # SMART PADDING: Fill borders with context when possible
        if context_region.shape[0] > expanded_region.shape[0] or context_region.shape[1] > expanded_region.shape[1]:
            # Scale context region to same scale
            ch, cw = context_region.shape[:2]
            new_ch = int(ch * scale)
            new_cw = int(cw * scale)
            
            if new_ch <= target_size and new_cw <= target_size:
                resized_context = cv2.resize(context_region, (new_cw, new_ch), interpolation=cv2.INTER_LANCZOS4)
                
                # Center context region
                ctx_y_offset = (target_size - new_ch) // 2
                ctx_x_offset = (target_size - new_cw) // 2
                
                # Place context first (as background)
                if ctx_y_offset >= 0 and ctx_x_offset >= 0:
                    standardized[ctx_y_offset:ctx_y_offset+new_ch, ctx_x_offset:ctx_x_offset+new_cw] = resized_context
        
        # Place resized expanded region on top (this is the main content)
        standardized[y_offset:y_offset+new_eh, x_offset:x_offset+new_ew] = resized_expanded
        
        # CRITICAL: Calculate where original tight bbox maps to in 800x800 space
        orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox.astype(int)
        
        # Relative position within expanded region
        rel_x1 = orig_x1 - ex_x1
        rel_y1 = orig_y1 - ex_y1
        rel_x2 = orig_x2 - ex_x1
        rel_y2 = orig_y2 - ex_y1
        
        # Scale and offset to 800x800 space
        orig_800_x1 = x_offset + int(rel_x1 * scale)
        orig_800_y1 = y_offset + int(rel_y1 * scale)
        orig_800_x2 = x_offset + int(rel_x2 * scale)
        orig_800_y2 = y_offset + int(rel_y2 * scale)
        
        # Ensure bounds are within target_size
        orig_800_x1 = max(0, min(target_size, orig_800_x1))
        orig_800_y1 = max(0, min(target_size, orig_800_y1))
        orig_800_x2 = max(0, min(target_size, orig_800_x2))
        orig_800_y2 = max(0, min(target_size, orig_800_y2))
        
        original_bbox_800 = np.array([orig_800_x1, orig_800_y1, orig_800_x2, orig_800_y2])
        
        # Store comprehensive transformation info
        transform_info = {
            # Expanded region info (for backward compatibility)
            'expanded_size': (new_ew, new_eh),  # (width, height) after scaling
            'scale': scale,
            'offset': (x_offset, y_offset),  # (x, y) - where expanded region is placed
            
            # NEW: Original bbox tracking
            'original_bbox_800': original_bbox_800,  # Original bbox coordinates in 800x800 space
            'original_bbox_orig': original_bbox,     # Original bbox in source image
            'expanded_bbox_orig': expanded_bbox,     # Expanded bbox in source image
            
            # Canvas info
            'letterbox_size': target_size,
            'applied_sr': apply_sr and original_size < self.sr_threshold_size,
            'used_smart_padding': context_region.shape[0] > expanded_region.shape[0] or context_region.shape[1] > expanded_region.shape[1]
        }
        
        return standardized, transform_info
    
    def extract_and_standardize_regions(
        self,
        image: np.ndarray,
        detections: Dict
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """Extract vehicle regions and standardize them to uniform size with dual bbox tracking"""
        if len(detections['boxes']) == 0:
            return [], []
        
        standardized_regions = []
        region_metadata = []
        
        # We need both expanded and original boxes for proper processing
        for i, (expanded_box, original_box, confidence, label) in enumerate(zip(
            detections['boxes'],           # Expanded boxes
            detections['original_boxes'],  # Original tight YOLO detections
            detections['confidences'],
            detections['labels']
        )):
            if confidence < self.car_conf_threshold:
                continue
            
            # Check if region is valid
            ex_x1, ex_y1, ex_x2, ex_y2 = expanded_box.astype(int)
            if ex_x2 <= ex_x1 or ex_y2 <= ex_y1:
                continue
            
            # Determine if super resolution should be applied based on expanded region size
            region_size = max(ex_x2 - ex_x1, ex_y2 - ex_y1)
            apply_sr = region_size < self.sr_threshold_size
            
            # Use enhanced standardization with smart padding and dual bbox tracking
            standardized_region, transform_info = self.standardize_region_with_smart_padding(
                image, original_box, expanded_box, self.region_size, apply_sr=apply_sr
            )
            
            # Store metadata with both bbox references
            metadata = {
                'expanded_bbox': expanded_box,
                'original_bbox': original_box,
                'confidence': confidence,
                'label': label,
                'region_index': i,
                'original_region_size': region_size,
                'transform_info': transform_info
            }
            
            standardized_regions.append(standardized_region)
            region_metadata.append(metadata)
        
        return standardized_regions, region_metadata
    
    def batch_process_standardized_regions(
        self,
        regions: List[np.ndarray],
        metadata_list: List[Dict]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Process standardized regions through SAM2 in batches"""
        if not regions:
            return [], []
        
        all_masks = []
        all_scores = []
        
        # Process regions in batches
        for i in range(0, len(regions), self.batch_size):
            batch_regions = regions[i:i + self.batch_size]
            batch_metadata = metadata_list[i:i + self.batch_size]
            
            # Process each region in the batch (SAM2 processes one image at a time)
            for region, metadata in zip(batch_regions, batch_metadata):
                # Convert BGR to RGB for SAM2
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                
                # Set image for SAM2
                self.sam2_predictor.set_image(region_rgb)
                
                # CORRECTED: Use original bbox coordinates for SAM2 prompt, not expanded area
                # This is the key fix - SAM2 should only segment the vehicle, not the background
                transform_info = metadata['transform_info']
                
                # Use the original bbox coordinates in 800x800 space
                original_bbox_800 = transform_info['original_bbox_800']
                bbox_prompt = np.array([original_bbox_800])
                
                try:
                    mask_result, quality_scores, _ = self.sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bbox_prompt,
                        multimask_output=False,
                        return_logits=False,
                    )
                    
                    if len(mask_result) > 0:
                        all_masks.append(mask_result[0])
                        all_scores.append(quality_scores[0])
                    else:
                        # Create empty mask if SAM2 fails
                        all_masks.append(np.zeros((self.region_size, self.region_size), dtype=bool))
                        all_scores.append(0.0)
                        
                except Exception as e:
                    logger.error(f"SAM2 processing failed for region: {e}")
                    all_masks.append(np.zeros((self.region_size, self.region_size), dtype=bool))
                    all_scores.append(0.0)
        
        return all_masks, all_scores
    
    def create_final_outputs(
        self,
        regions: List[np.ndarray],
        masks: List[np.ndarray],
        metadata_list: List[Dict],
        scores: List[float]
    ) -> List[Dict]:
        """Create final 512x512 outputs from processed regions and masks"""
        final_results = []
        
        for region, mask, metadata, score in zip(regions, masks, metadata_list, scores):
            try:
                # Extract the content area from the standardized region
                transform_info = metadata['transform_info']
                x_offset, y_offset = transform_info['offset']
                new_ew, new_eh = transform_info['expanded_size']
                
                # Extract content region (removing letterbox padding)
                content_region = region[y_offset:y_offset+new_eh, x_offset:x_offset+new_ew]
                content_mask = mask[y_offset:y_offset+new_eh, x_offset:x_offset+new_ew]
                
                # Convert content region back to RGB for final processing
                content_rgb = cv2.cvtColor(content_region, cv2.COLOR_BGR2RGB)
                
                # Create RGBA image
                rgba_array = np.concatenate([
                    content_rgb,
                    np.ones((content_rgb.shape[0], content_rgb.shape[1], 1), dtype=np.uint8) * 255
                ], axis=2)
                
                # Apply mask to alpha channel
                rgba_array[:, :, 3] = (content_mask * 255).astype(np.uint8)
                
                # Convert to PIL image
                masked_image = Image.fromarray(rgba_array, 'RGBA')
                
                # Find content bounding box
                alpha = np.array(masked_image.getchannel('A'))
                coords = np.argwhere(alpha > 0)
                
                if len(coords) == 0:
                    continue
                
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Crop to content
                cropped = masked_image.crop((x_min, y_min, x_max, y_max))
                
                # Scale to final size while preserving aspect ratio
                width, height = cropped.size
                scale_factor = min(self.final_size / width, self.final_size / height)
                
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create final image with transparent background
                final_img = Image.new('RGBA', (self.final_size, self.final_size), (0, 0, 0, 0))
                paste_x = (self.final_size - new_width) // 2
                paste_y = (self.final_size - new_height) // 2
                final_img.paste(resized, (paste_x, paste_y), resized)
                
                # Compile result
                result = {
                    'processed_image': final_img,
                    'detection_label': metadata['label'],
                    'detection_confidence': float(metadata['confidence']),
                    'segmentation_score': float(score),
                    'original_region_size': metadata['original_region_size'],
                    'used_super_resolution': metadata['transform_info']['applied_sr'],
                    'processing_metadata': metadata
                }
                
                final_results.append(result)
                
            except Exception as e:
                logger.error(f"Error creating final output: {e}")
                continue
        
        return final_results
    
    def create_final_outputs_from_yolo_masks(
        self,
        image: np.ndarray,
        detections: Dict
    ) -> List[Dict]:
        """
        Create final outputs directly from YOLO segmentation masks (no region extraction needed)
        
        Args:
            image: Original image (BGR format)
            detections: YOLO detection results with masks
            
        Returns:
            List of final result dictionaries
        """
        final_results = []
        
        if detections['masks'] is None or len(detections['boxes']) == 0:
            return final_results
        
        # Convert image to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for i, (box, confidence, label, mask) in enumerate(zip(
            detections['boxes'],
            detections['confidences'], 
            detections['labels'],
            detections['masks']
        )):
            try:
                # KEY FIX: YOLO masks may or may not be in original image dimensions
                # Check dimensions and resize only if necessary
                mask_h, mask_w = mask.shape
                img_h, img_w = image.shape[:2]
                
                if mask_h == img_h and mask_w == img_w:
                    # Mask is already in correct dimensions
                    final_mask = mask
                    logger.debug(f"Mask {i} already in correct dimensions: {mask_w}x{mask_h}")
                else:
                    # Mask needs resizing to match original image
                    final_mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    logger.debug(f"Mask {i} resized from {mask_w}x{mask_h} to {img_w}x{img_h}")
                
                # Convert to boolean mask
                bool_mask = final_mask > 0.5
                
                # Find bounding box of the mask
                coords = np.argwhere(bool_mask)
                if len(coords) == 0:
                    continue
                    
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Add some padding around the mask
                padding = 10
                y_min = max(0, y_min - padding)
                x_min = max(0, x_min - padding) 
                y_max = min(image.shape[0], y_max + padding)
                x_max = min(image.shape[1], x_max + padding)
                
                # Extract region from original image and mask
                region = image_rgb[y_min:y_max, x_min:x_max]
                region_mask = bool_mask[y_min:y_max, x_min:x_max]
                
                # Create RGBA image
                rgba_array = np.concatenate([
                    region,
                    np.ones((region.shape[0], region.shape[1], 1), dtype=np.uint8) * 255
                ], axis=2)
                
                # Apply mask to alpha channel
                rgba_array[:, :, 3] = (region_mask * 255).astype(np.uint8)
                
                # Convert to PIL image
                masked_image = Image.fromarray(rgba_array, 'RGBA')
                
                # Find content bounding box in the masked region
                alpha = np.array(masked_image.getchannel('A'))
                content_coords = np.argwhere(alpha > 0)
                
                if len(content_coords) == 0:
                    continue
                
                cy_min, cx_min = content_coords.min(axis=0)
                cy_max, cx_max = content_coords.max(axis=0)
                
                # Crop to content
                cropped = masked_image.crop((cx_min, cy_min, cx_max, cy_max))
                
                # Scale to final size while preserving aspect ratio
                width, height = cropped.size
                if width == 0 or height == 0:
                    continue
                    
                scale_factor = min(self.final_size / width, self.final_size / height)
                
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create final image with transparent background
                final_img = Image.new('RGBA', (self.final_size, self.final_size), (0, 0, 0, 0))
                paste_x = (self.final_size - new_width) // 2
                paste_y = (self.final_size - new_height) // 2
                final_img.paste(resized, (paste_x, paste_y), resized)
                
                # Calculate original region size for compatibility
                original_region_size = max(x_max - x_min, y_max - y_min)
                
                # Compile result (same format as SAM2 pipeline)
                result = {
                    'processed_image': final_img,
                    'detection_label': label,
                    'detection_confidence': float(confidence),
                    'segmentation_score': 1.0,  # YOLO doesn't provide separate segmentation score
                    'original_region_size': original_region_size,
                    'used_super_resolution': False,  # YOLO doesn't use super resolution in this pipeline
                    'processing_metadata': {
                        'method': 'yolo',
                        'mask_area': int(bool_mask.sum()),
                        'bbox': box.tolist(),
                        'region_bounds': [x_min, y_min, x_max, y_max],
                        'mask_original_shape': [mask_h, mask_w],
                        'image_shape': [img_h, img_w],
                        'mask_resized': mask_h != img_h or mask_w != img_w
                    }
                }
                
                final_results.append(result)
                
            except Exception as e:
                logger.error(f"Error creating final output from YOLO mask {i}: {e}")
                continue
        
        return final_results
    
    def process_image_batch(
        self,
        image_paths: List[str],
        min_segmentation_score: float = 0.7,
        save_results: bool = False,
        output_dir: str = "./batch_output"
    ) -> List[List[Dict]]:
        """
        Process multiple images in batch with uniform region standardization
        
        Args:
            image_paths: List of image file paths
            min_segmentation_score: Minimum SAM2 quality score
            save_results: Whether to save processed images
            output_dir: Directory to save results
            
        Returns:
            List of results for each image
        """
        start_time = time.time()
        print(f"Starting batch processing of {len(image_paths)} images...")
        
        # Load images
        print("Loading images...")
        loaded_images = self.load_images_batch(image_paths)
        
        if not loaded_images:
            print("No images loaded successfully!")
            return []
        
        # Extract images and paths
        images = [img for img, _ in loaded_images]
        paths = [path for _, path in loaded_images]
        
        # Process each image based on segmentation method
        all_results = []
        total_regions_processed = 0
        
        if self.segmentation_method == 'sam2':
            # SAM2 Processing Pipeline: YOLO detection → region extraction → SAM2 segmentation
            print("Detecting vehicles (SAM2 method)...")
            all_detections = self.batch_detect_vehicles(images)
            
            for image, image_path, detections in zip(images, paths, all_detections):
                if len(detections['boxes']) == 0:
                    print(f"No vehicles detected in {Path(image_path).name}")
                    all_results.append([])
                    continue
                
                print(f"Processing {Path(image_path).name}: {len(detections['boxes'])} vehicles (SAM2)")
                
                # Extract and standardize regions
                standardized_regions, region_metadata = self.extract_and_standardize_regions(
                    image, detections
                )
                
                if not standardized_regions:
                    print(f"  No valid regions extracted")
                    all_results.append([])
                    continue
                
                print(f"  Extracted {len(standardized_regions)} standardized regions")
                
                # Process through SAM2
                masks, scores = self.batch_process_standardized_regions(
                    standardized_regions, region_metadata
                )
                
                # Create final outputs
                results = self.create_final_outputs(
                    standardized_regions, masks, region_metadata, scores
                )
                
                # Filter by segmentation score
                filtered_results = [
                    r for r in results 
                    if r['segmentation_score'] >= min_segmentation_score
                ]
                
                print(f"  Generated {len(filtered_results)} final outputs (score ≥ {min_segmentation_score})")
                total_regions_processed += len(standardized_regions)
                all_results.append(filtered_results)
                
        elif self.segmentation_method == 'yolo':
            # YOLO Processing Pipeline: YOLO segmentation (detection + segmentation in one pass)
            print("Using YOLO segmentation method...")
            
            # Get YOLO segmentation results (includes both detection and segmentation)
            all_yolo_results = self.batch_detect_and_segment_vehicles_yolo(images)
            
            for image, image_path, yolo_results in zip(images, paths, all_yolo_results):
                if len(yolo_results['boxes']) == 0:
                    print(f"No vehicles detected in {Path(image_path).name}")
                    all_results.append([])
                    continue
                
                print(f"Processing {Path(image_path).name}: {len(yolo_results['boxes'])} vehicles (YOLO)")
                
                # Create final outputs directly from YOLO masks
                results = self.create_final_outputs_from_yolo_masks(image, yolo_results)
                
                # Filter by segmentation score (YOLO doesn't have separate segmentation scores, but we can use confidence)
                filtered_results = [
                    r for r in results 
                    if r['detection_confidence'] >= min_segmentation_score
                ]
                
                print(f"  Generated {len(filtered_results)} final outputs (conf ≥ {min_segmentation_score})")
                total_regions_processed += len(yolo_results['boxes'])
                all_results.append(filtered_results)
        
        # Save results if requested
        if save_results and any(all_results):
            self._save_batch_results(all_results, paths, output_dir)
        
        # Print summary
        total_time = time.time() - start_time
        total_outputs = sum(len(results) for results in all_results)
        
        print(f"\n=== BATCH PROCESSING COMPLETE ===")
        print(f"Processed: {len(image_paths)} images")
        print(f"Total regions processed: {total_regions_processed}")
        print(f"Total outputs generated: {total_outputs}")
        print(f"Processing time: {total_time:.2f}s")
        print(f"Average time per region: {total_time/max(1, total_regions_processed):.2f}s")
        
        return all_results
    
    def _save_batch_results(self, all_results: List[List[Dict]], image_paths: List[str], output_dir: str):
        """Save batch processing results"""
        os.makedirs(output_dir, exist_ok=True)
        
        total_saved = 0
        for results, image_path in zip(all_results, image_paths):
            if not results:
                continue
            
            image_name = Path(image_path).stem
            
            for i, result in enumerate(results):
                filename = f"{image_name}_vehicle_{i+1}_{result['detection_label']}.png"
                output_path = os.path.join(output_dir, filename)
                result['processed_image'].save(output_path)
                total_saved += 1
        
        print(f"Saved {total_saved} processed vehicle images to {output_dir}")


def main():
    """Test the batch vehicle processor with both segmentation methods"""
    # Test with sample directory
    image_dir = "/mnt/damian/Projects/car_data_scraper/images/autoevolution_renderings/article_230605"
    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ][:8]  # Test with up to 8 images
    
    print(f"Testing with {len(image_paths)} images from {image_dir}")
    print("=" * 80)
    
    # Test SAM2 method (original)
    print("\n🔬 TESTING SAM2 SEGMENTATION METHOD")
    print("-" * 50)
    sam2_processor = BatchVehicleProcessor(
        segmentation_method='sam2',  # Traditional YOLO detection + SAM2 segmentation
        enable_super_resolution=True,
        sr_threshold_size=300,
        bbox_expansion=0.25,
        region_size=800,
        final_size=512,
        batch_size=4,
        yolo_batch_size=8
    )
    
    sam2_results = sam2_processor.process_image_batch(
        image_paths,
        min_segmentation_score=0.6,
        save_results=True,
        output_dir="./batch_test_output_sam2"
    )
    
    print("\n⚡ TESTING YOLO SEGMENTATION METHOD")
    print("-" * 50)
    yolo_processor = BatchVehicleProcessor(
        segmentation_method='yolo',  # YOLO detection + segmentation in one pass
        enable_super_resolution=False,  # Not needed for YOLO method
        bbox_expansion=0.25,
        final_size=512,
        yolo_batch_size=4  # More conservative for segmentation model
    )
    
    yolo_results = yolo_processor.process_image_batch(
        image_paths,
        min_segmentation_score=0.6,
        save_results=True,
        output_dir="./batch_test_output_yolo"
    )
    
    # Compare results
    print("\n📊 COMPARISON RESULTS")
    print("=" * 80)
    
    sam2_total = sum(len(image_results) for image_results in sam2_results)
    yolo_total = sum(len(image_results) for image_results in yolo_results)
    
    print(f"SAM2 Method: {sam2_total} vehicles processed")
    print(f"YOLO Method: {yolo_total} vehicles processed")
    
    # Detailed comparison per image
    print(f"\n{'Image':<30} {'SAM2':<10} {'YOLO':<10} {'Difference':<10}")
    print("-" * 60)
    
    for i, (sam2_image_results, yolo_image_results, image_path) in enumerate(zip(sam2_results, yolo_results, image_paths)):
        image_name = Path(image_path).name[:25] + "..." if len(Path(image_path).name) > 25 else Path(image_path).name
        sam2_count = len(sam2_image_results)
        yolo_count = len(yolo_image_results)
        diff = yolo_count - sam2_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{image_name:<30} {sam2_count:<10} {yolo_count:<10} {diff_str:<10}")
    
    print(f"\n💡 PERFORMANCE BENEFITS OF YOLO METHOD:")
    print("   • Single-pass detection + segmentation (faster)")
    print("   • True batch processing throughout pipeline")
    print("   • Reduced memory fragmentation")
    print("   • No region extraction overhead")
    print("   • Simplified processing pipeline")


if __name__ == "__main__":
    main()