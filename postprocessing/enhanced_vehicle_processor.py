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
from typing import List, Tuple, Optional, Dict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Real-ESRGAN import failed: {e}")
    print("Super resolution will be disabled")
    REALESRGAN_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLO Vehicle classes
VEHICLE_CLASSES = {2: 'car', 5: 'bus', 7: 'truck'}

class PreSAMSuperResolutionProcessor:
    def __init__(
        self, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        yolo_conf_threshold=0.25,
        car_conf_threshold=0.5,
        enable_super_resolution=True,
        sr_threshold_size=256,  # Apply SR if vehicle bbox is smaller than this
        max_upscale_factor=2.0,  # Maximum upscaling without SR
        bbox_expansion=0.1,  # Expand bounding box by 10%
        bbox_padding=0.2  # Extra padding for extraction (20%)
    ):
        """
        Enhanced vehicle processing with pre-SAM super resolution
        
        Args:
            device: Device to run models on
            yolo_conf_threshold: YOLO detection confidence threshold
            car_conf_threshold: Minimum confidence for vehicle detections
            enable_super_resolution: Whether to use Real-ESRGAN for small vehicles
            sr_threshold_size: Apply super resolution if bbox smaller than this
            max_upscale_factor: Maximum upscaling factor without super resolution
            bbox_expansion: Factor to expand bounding boxes for detection
            bbox_padding: Extra padding when extracting regions for SR
        """
        self.device = device
        self.yolo_conf_threshold = yolo_conf_threshold
        self.car_conf_threshold = car_conf_threshold
        self.enable_super_resolution = enable_super_resolution
        self.sr_threshold_size = sr_threshold_size
        self.max_upscale_factor = max_upscale_factor
        self.bbox_expansion = bbox_expansion
        self.bbox_padding = bbox_padding
        
        print(f"Initializing pre-SAM SR models on device: {device}")
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.yolo_model = YOLO('yolo11x.pt').to(device)
        self.yolo_model.eval()
        
        # Load SAM2 model
        print("Loading SAM2 model...")
        checkpoint_path = "/mnt/damian/Projects/sam2/checkpoints/sam2.1_hiera_large.pt"
        sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam2_model = build_sam2(sam2_config, checkpoint_path, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Load Real-ESRGAN model if enabled
        if self.enable_super_resolution:
            if REALESRGAN_AVAILABLE:
                print("Loading Real-ESRGAN model...")
                self.setup_super_resolution()
            else:
                print("Real-ESRGAN not available, using fallback upscaling")
        
        print("Models loaded successfully!")
    
    def setup_super_resolution(self):
        """Initialize Real-ESRGAN for super resolution"""
        # Define model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'postprocessing/weights/RealESRGAN_x4plus.pth'
        
        # Initialize upsampler
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
        """
        Expand bounding box by a factor while keeping it within image bounds
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Calculate expansion
        expand_w = width * self.bbox_expansion
        expand_h = height * self.bbox_expansion
        
        # Apply expansion
        x1 = max(0, x1 - expand_w / 2)
        y1 = max(0, y1 - expand_h / 2)
        x2 = min(image_shape[1], x2 + expand_w / 2)
        y2 = min(image_shape[0], y2 + expand_h / 2)
        
        return np.array([x1, y1, x2, y2])
    
    def extract_bbox_region(self, image: np.ndarray, box: np.ndarray, padding: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Extract region around bounding box with padding for super resolution
        
        Args:
            image: Full image
            box: Bounding box [x1, y1, x2, y2]
            padding: Padding factor (uses self.bbox_padding if None)
            
        Returns:
            Tuple of (extracted_region, region_info)
        """
        if padding is None:
            padding = self.bbox_padding
            
        x1, y1, x2, y2 = box.astype(int)
        width = x2 - x1
        height = y2 - y1
        
        # Calculate padding
        pad_w = int(width * padding)
        pad_h = int(height * padding)
        
        # Apply padding while staying within image bounds
        img_h, img_w = image.shape[:2]
        x1_padded = max(0, x1 - pad_w)
        y1_padded = max(0, y1 - pad_h)
        x2_padded = min(img_w, x2 + pad_w)
        y2_padded = min(img_h, y2 + pad_h)
        
        # Extract region
        region = image[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Calculate relative box coordinates within the extracted region
        rel_x1 = x1 - x1_padded
        rel_y1 = y1 - y1_padded
        rel_x2 = x2 - x1_padded
        rel_y2 = y2 - y1_padded
        
        region_info = {
            'padded_bbox': [x1_padded, y1_padded, x2_padded, y2_padded],
            'relative_bbox': [rel_x1, rel_y1, rel_x2, rel_y2],
            'original_bbox': box,
            'region_size': region.shape[:2],
            'vehicle_size': max(width, height)
        }
        
        return region, region_info
    
    def apply_super_resolution(self, image: np.ndarray, input_format='BGR') -> np.ndarray:
        """
        Apply Real-ESRGAN super resolution to image or fallback to cv2 upscaling
        
        Args:
            image: Input image array
            input_format: 'BGR' or 'RGB' - format of input image
        """
        if not self.enable_super_resolution:
            return image
        
        try:
            if REALESRGAN_AVAILABLE and hasattr(self, 'upsampler'):
                # Use Real-ESRGAN (expects BGR format)
                if input_format == 'RGB' and len(image.shape) == 3 and image.shape[2] == 3:
                    # Convert RGB to BGR for Real-ESRGAN
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif input_format == 'BGR':
                    # Already in BGR format
                    image_bgr = image
                else:
                    image_bgr = image
                
                # Apply super resolution
                output_bgr, _ = self.upsampler.enhance(image_bgr, outscale=4)
                
                # Convert back to same format as input
                if input_format == 'RGB' and len(image.shape) == 3 and image.shape[2] == 3:
                    output = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
                elif input_format == 'BGR':
                    output = output_bgr
                else:
                    output = output_bgr
                
                return output
            else:
                # Fallback: Use cv2 with INTER_CUBIC for quality upscaling
                h, w = image.shape[:2]
                new_h, new_w = h * 2, w * 2  # 2x upscaling
                
                # Ensure uint8 format
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                
                # Apply bilateral filter for noise reduction
                filtered = cv2.bilateralFilter(image, 9, 75, 75)
                
                # Upscale using INTER_CUBIC
                upscaled = cv2.resize(filtered, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Apply sharpening filter
                kernel = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
                sharpened = cv2.filter2D(upscaled, -1, kernel)
                
                # Blend original upscaled with sharpened
                output = cv2.addWeighted(upscaled, 0.7, sharpened, 0.3, 0)
                
                return output
                
        except Exception as e:
            logger.error(f"Super resolution failed: {e}")
            return image
    
    def detect_vehicles(self, image: np.ndarray) -> Dict:
        """
        Detect vehicles in image using YOLO with bbox expansion
        """
        results = self.yolo_model(source=image, conf=self.yolo_conf_threshold, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Filter for vehicle classes only
        vehicle_mask = np.isin(detections.class_id, list(VEHICLE_CLASSES.keys()))
        
        if vehicle_mask.any():
            # Expand bounding boxes
            expanded_boxes = []
            for box in detections.xyxy[vehicle_mask]:
                expanded_box = self.expand_bbox(box, image.shape[:2])
                expanded_boxes.append(expanded_box)
            
            filtered_detections = {
                'boxes': np.array(expanded_boxes),
                'original_boxes': detections.xyxy[vehicle_mask],
                'confidences': detections.confidence[vehicle_mask],
                'class_ids': detections.class_id[vehicle_mask],
                'labels': [VEHICLE_CLASSES[cid] for cid in detections.class_id[vehicle_mask]]
            }
        else:
            filtered_detections = {
                'boxes': np.array([]),
                'original_boxes': np.array([]),
                'confidences': np.array([]),
                'class_ids': np.array([]),
                'labels': []
            }
        
        return filtered_detections
    
    def process_vehicle_bbox(
        self, 
        image: np.ndarray, 
        box: np.ndarray, 
        confidence: float, 
        label: str,
        target_size: int = 512
    ) -> Tuple[Optional[Image.Image], Dict]:
        """
        Process a single vehicle bounding box with pre-SAM super resolution
        
        Args:
            image: Full input image
            box: Vehicle bounding box
            confidence: Detection confidence
            label: Vehicle label
            target_size: Final output size
            
        Returns:
            Tuple of (processed_image, metrics)
        """
        # Extract region around the vehicle
        region, region_info = self.extract_bbox_region(image, box)
        original_vehicle_size = region_info['vehicle_size']
        
        # Determine if super resolution is needed
        needs_sr = original_vehicle_size < self.sr_threshold_size
        
        metrics = {
            'original_vehicle_size': original_vehicle_size,
            'needs_super_resolution': needs_sr,
            'used_super_resolution': False,
            'region_shape': region.shape,
            'detection_confidence': confidence,
            'detection_label': label
        }
        
        # Apply super resolution if needed
        if needs_sr and self.enable_super_resolution:
            print(f"  → Applying SR to {label} region ({original_vehicle_size}px)")
            upscaled_region = self.apply_super_resolution(region, input_format='BGR')
            
            # Calculate scale factor
            sr_scale = upscaled_region.shape[0] / region.shape[0]
            
            # Update relative bbox coordinates for SAM2
            rel_bbox = region_info['relative_bbox']
            scaled_rel_bbox = [coord * sr_scale for coord in rel_bbox]
            
            metrics['used_super_resolution'] = True
            metrics['sr_scale_factor'] = sr_scale
            metrics['upscaled_shape'] = upscaled_region.shape
        else:
            print(f"  → Processing {label} without SR ({original_vehicle_size}px)")
            upscaled_region = region
            scaled_rel_bbox = region_info['relative_bbox']
            metrics['sr_scale_factor'] = 1.0
            metrics['upscaled_shape'] = region.shape
        
        # Convert to RGB for SAM2 (SAM2 expects RGB format)
        if len(upscaled_region.shape) == 3 and upscaled_region.shape[2] == 3:
            upscaled_region_rgb = cv2.cvtColor(upscaled_region, cv2.COLOR_BGR2RGB)
        else:
            upscaled_region_rgb = upscaled_region
        
        # Run SAM2 on the (possibly upscaled) region
        self.sam2_predictor.set_image(upscaled_region_rgb)
        
        # Use the relative bounding box as prompt
        bbox_prompt = np.array([scaled_rel_bbox]).reshape(1, 4)
        
        try:
            mask_result, quality_scores, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox_prompt,
                multimask_output=True,
                return_logits=False,
            )
            
            if len(mask_result) > 0:
                # Select best mask
                best_idx = np.argmax(quality_scores)
                best_mask = mask_result[best_idx]
                segmentation_score = quality_scores[best_idx]
                
                metrics['segmentation_score'] = float(segmentation_score)
                
                # Apply mask to the upscaled region and create final image
                # Use RGB version for final processing
                processed_image = self.apply_mask_and_rescale(
                    upscaled_region_rgb, best_mask, target_size, metrics
                )
                
                return processed_image, metrics
            else:
                print(f"  → No valid masks generated for {label}")
                return None, metrics
                
        except Exception as e:
            logger.error(f"SAM2 processing failed: {e}")
            return None, metrics
    
    def apply_mask_and_rescale(
        self, 
        region: np.ndarray, 
        mask: np.ndarray, 
        target_size: int,
        metrics: Dict
    ) -> Optional[Image.Image]:
        """
        Apply mask to region and rescale to target size
        """
        try:
            # Convert numpy to PIL
            if region.dtype == np.uint8:
                pil_region = Image.fromarray(region)
            else:
                pil_region = Image.fromarray((region * 255).astype(np.uint8))
            
            # Ensure RGB mode
            if pil_region.mode != 'RGB':
                pil_region = pil_region.convert('RGB')
            
            # Convert to RGBA
            img_array = np.array(pil_region)
            rgba_array = np.concatenate([
                img_array,
                np.ones((img_array.shape[0], img_array.shape[1], 1), dtype=np.uint8) * 255
            ], axis=2)
            
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
            
            # Calculate final scaling
            width, height = cropped.size
            scale_factor = min(target_size / width, target_size / height)
            
            # Avoid excessive upscaling
            if scale_factor > self.max_upscale_factor:
                scale_factor = self.max_upscale_factor
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create final image with transparent background
            final_img = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
            paste_x = (target_size - new_width) // 2
            paste_y = (target_size - new_height) // 2
            final_img.paste(resized, (paste_x, paste_y), resized)
            
            # Update metrics
            metrics['final_scale_factor'] = scale_factor
            metrics['final_size'] = (new_width, new_height)
            
            return final_img
            
        except Exception as e:
            logger.error(f"Error in mask and rescale: {e}")
            return None
    
    def process_image(
        self,
        image_path: str,
        min_segmentation_score: float = 0.7,
        target_size: int = 512
    ) -> List[Dict]:
        """
        Complete processing pipeline with pre-SAM super resolution
        """
        print(f"Processing: {image_path}")
        
        # Load image
        pil_image = Image.open(image_path).convert('RGB')
        print(f"Original image size: {pil_image.size}")
        
        # Convert to OpenCV format
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Detect vehicles
        print("Detecting vehicles...")
        detections = self.detect_vehicles(cv2_image)
        
        if len(detections['boxes']) == 0:
            print("No vehicles detected!")
            return []
        
        print(f"Found {len(detections['boxes'])} vehicles")
        
        # Process each vehicle
        processed_results = []
        
        for i, (box, confidence, label) in enumerate(zip(
            detections['boxes'],
            detections['confidences'],
            detections['labels']
        )):
            if confidence < self.car_conf_threshold:
                print(f"Skipping {label} with low confidence: {confidence:.3f}")
                continue
            
            print(f"Processing {label} {i+1}/{len(detections['boxes'])}...")
            
            # Process this vehicle with pre-SAM SR
            processed_image, metrics = self.process_vehicle_bbox(
                cv2_image, box, confidence, label, target_size
            )
            
            if processed_image is not None and metrics.get('segmentation_score', 0) >= min_segmentation_score:
                result = {
                    'processed_image': processed_image,
                    'original_path': image_path,
                    'processing_metrics': metrics
                }
                processed_results.append(result)
                
                # Log processing details
                sr_used = metrics.get('used_super_resolution', False)
                orig_size = metrics.get('original_vehicle_size', 0)
                seg_score = metrics.get('segmentation_score', 0)
                
                print(f"  ✓ {label}: {orig_size}px → SR: {sr_used}, Seg: {seg_score:.3f}")
            else:
                seg_score = metrics.get('segmentation_score', 0)
                print(f"  ✗ {label}: Low segmentation score ({seg_score:.3f})")
        
        print(f"Processing complete. Generated {len(processed_results)} vehicle images.")
        return processed_results


def main():
    """Test the pre-SAM super resolution processor"""
    import matplotlib.pyplot as plt
    
    # Initialize processor
    processor = PreSAMSuperResolutionProcessor(
        enable_super_resolution=True,
        sr_threshold_size=300,  # Test with higher threshold
        max_upscale_factor=2.0,
        bbox_expansion=0.15,
        bbox_padding=0.25  # Extra padding for SR
    )
    
    # Test image
    test_image = "/mnt/damian/Projects/car_data_scraper/images/autoevolution_distributed/seat/ibiza-cupra/seat-ibiza-cupra-2012_SEAT-Ibiza-Cupra-5048_33_29.jpg"
    
    # Process
    results = processor.process_image(
        test_image,
        min_segmentation_score=0.6,
        target_size=512
    )
    
    # Display and save results
    if results:
        print(f"\n=== RESULTS ===")
        
        # Create output directory
        output_dir = "./test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each result
        for i, result in enumerate(results):
            metrics = result['processing_metrics']
            
            print(f"\nVehicle {i+1}:")
            print(f"  Label: {metrics['detection_label']}")
            print(f"  Original size: {metrics['original_vehicle_size']}px")
            print(f"  Used SR: {metrics['used_super_resolution']}")
            print(f"  SR scale: {metrics.get('sr_scale_factor', 1.0):.2f}x")
            print(f"  Segmentation score: {metrics['segmentation_score']:.3f}")
            print(f"  Final size: {metrics['final_size']}")
            
            # Save image
            filename = f"pre_sam_sr_vehicle_{i+1}_{metrics['detection_label']}.png"
            output_path = os.path.join(output_dir, filename)
            result['processed_image'].save(output_path)
            print(f"  Saved: {output_path}")
        
        # Create comparison plot
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
        if len(results) == 1:
            axes = [axes]
        
        for ax, result in zip(axes, results):
            ax.imshow(result['processed_image'])
            metrics = result['processing_metrics']
            title = f"{metrics['detection_label']}\n"
            title += f"Original: {metrics['original_vehicle_size']}px\n"
            title += f"Pre-SAM SR: {'Yes' if metrics['used_super_resolution'] else 'No'}\n"
            title += f"Seg Score: {metrics['segmentation_score']:.3f}"
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pre_sam_sr_comparison.png"), dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("No vehicles processed successfully")


if __name__ == "__main__":
    main()