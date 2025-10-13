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
from rembg import remove
import io
import time
import json
import random
import argparse

# MODEL_PATH = '../00_model_checkpoints/yolov10x.pt'
MODEL_PATH = '/mnt/damian/Projects/Diffus3D/00_model_checkpoints/yolo11x.pt'

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

class YOLO_helper:
    def __init__(self, device="cuda"):
        self.model = YOLO(MODEL_PATH).to(device)
        self.model.eval()

    def label_batch_images(self, pil_images, single_label_per_image=False):
        cv2_images = [cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) for pil_img in pil_images]
        results = self.model(source=cv2_images, conf=0.25, verbose=False)
        labels = []
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            if single_label_per_image and len(detections.class_id) > 0:
                if detections.confidence[0] < 0.89:
                    internal_labels = " "
                else:
                    internal_labels = f"{category_dict[detections.class_id[0]]}"
            else:
                labels_single_image = [
                    f"{category_dict[class_id]}:{confidence:.2f}"
                    for class_id, confidence in zip(detections.class_id, detections.confidence)
                    ]
                internal_labels = ""
                for label in labels_single_image:
                    if len(internal_labels) > 0:
                        internal_labels += ", "
                    internal_labels += label
                if len(internal_labels) == 0:
                    internal_labels = " "
            labels.append(internal_labels)
        return labels

def download_image(url, save_path=None, max_retries=3):
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
                print(f"Access forbidden (403) for {url}, retry {attempt+1}/{max_retries}")
                time.sleep(2 ** attempt)
            else:
                print(f"Failed to download image, status code: {response.status_code}")
                break
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            time.sleep(2 ** attempt)
    return None

def remove_background_and_rescale(img, target_size=512):
    img_no_bg = remove(img)
    alpha = np.array(img_no_bg.getchannel('A'))
    coords = np.argwhere(alpha > 0)
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = img_no_bg.crop((x_min, y_min, x_max, y_max))
    width, height = cropped.size
    scale_factor = min(target_size / width, target_size / height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized = cropped.resize((new_width, new_height), Image.LANCZOS)
    final_img = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    final_img.paste(resized, (paste_x, paste_y), resized)
    return final_img

def process_image_batch(image_urls, metadata_rows, yolo_helper, temp_dir, output_dir, metadata_file, gpu_batch_size=64, metadata_row_file_limit=25):
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    all_images = []
    all_temp_paths = []
    all_metadata_indices = []
    all_original_urls = []
    
    print(f"Starting to download images from {len(image_urls)} metadata rows")
    for i, (url_list, metadata) in enumerate(zip(image_urls, metadata_rows)):
        urls = [url.strip() for url in url_list.split(',')]
        random.shuffle(urls)
        
        successful_downloads = 0
        
        for j, url in enumerate(urls):
            if successful_downloads >= metadata_row_file_limit:
                break
                
            temp_path = os.path.join(temp_dir, f"temp_{i}_{j}_{uuid.uuid4().hex[:6]}.jpg")
            img = download_image(url, temp_path)
            
            if img is not None:
                all_images.append(img)
                all_temp_paths.append(temp_path)
                all_metadata_indices.append(i)
                all_original_urls.append(url)
                successful_downloads += 1
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            time.sleep(0.5)
        
        if successful_downloads > 0:
            print(f"Row {i}: Downloaded {successful_downloads} images successfully")
        else:
            print(f"Row {i}: Failed to download any images")
            
        time.sleep(1)
    
    if not all_images:
        print("No images were successfully downloaded in this batch")
        return
    
    print(f"Successfully downloaded {len(all_images)} images in total")
    
    car_metadata = []
    
    for batch_start in range(0, len(all_images), gpu_batch_size):
        batch_end = min(batch_start + gpu_batch_size, len(all_images))
        batch_images = all_images[batch_start:batch_end]
        batch_indices = all_metadata_indices[batch_start:batch_end]
        batch_urls = all_original_urls[batch_start:batch_end]
        batch_paths = all_temp_paths[batch_start:batch_end]
        
        print(f"Processing YOLO batch {batch_start//gpu_batch_size + 1}/{(len(all_images)+gpu_batch_size-1)//gpu_batch_size} with {len(batch_images)} images")
        
        labels = yolo_helper.label_batch_images(batch_images, single_label_per_image=True)
        
        for i, (img, label, metadata_idx, url, temp_path) in enumerate(zip(batch_images, labels, batch_indices, batch_urls, batch_paths)):
            if 'car' in label.lower() or 'truck' in label.lower():
                print(f"Processing image with label: {label}")
                processed_img = remove_background_and_rescale(img)
                if processed_img is None:
                    print("  Failed to remove background or no foreground detected")
                    continue
                    
                unique_id = str(uuid.uuid4())[:8]
                brand = metadata_rows[metadata_idx]['brand'].replace(' ', '_')
                model = metadata_rows[metadata_idx]['model'].replace(' ', '_')
                year = metadata_rows[metadata_idx]['from_year']
                
                filename = f"{brand}_{model}_{year}_{unique_id}.png"
                output_path = os.path.join(output_dir, filename)
                
                processed_img.save(output_path)
                
                metadata_entry = metadata_rows[metadata_idx].copy()
                metadata_entry['original_url'] = url
                metadata_entry['saved_filename'] = filename
                metadata_entry['detection_label'] = label
                car_metadata.append(metadata_entry)
                
                print(f"Saved car image: {filename}")
            else:
                print(f"Skipping image with label: {label}")
        
        for path in batch_paths:
            if os.path.exists(path):
                os.remove(path)
    
    if car_metadata:
        with open(metadata_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=car_metadata[0].keys())
            if os.stat(metadata_file).st_size == 0:
                writer.writeheader()
            writer.writerows(car_metadata)
        print(f"Saved metadata for {len(car_metadata)} car images")
    else:
        print("No car images were found in the downloaded images")
    
    print(f"Batch processing complete. Processed {len(all_images)} images, found {len(car_metadata)} cars.")

def main():
    parser = argparse.ArgumentParser(description='Download and process car images.')
    parser.add_argument('--csv_file', type=str, default='./car_models_3887/data_full.csv', help='Path to the input CSV file.')
    parser.add_argument('--output_dir', type=str, default='./car_images', help='Directory to save the processed images.')
    parser.add_argument('--metadata_file', type=str, default='./car_images_metadata.csv', help='Path to the output metadata file.')
    parser.add_argument('--temp_dir', type=str, default='./temp_downloads', help='Directory to store temporary downloads.')
    parser.add_argument('--index_batch_size', type=int, default=5, help='Number of metadata rows to process at once.')
    parser.add_argument('--gpu_batch_size', type=int, default=32, help='Number of images to process with YOLO at once.')
    parser.add_argument('--test_batch', action='store_true', help='Process only the first batch as a test.')

    args = parser.parse_args()

    if 'data' not in globals():
        data = pd.read_csv(args.csv_file)

    fields = ['brand', 'model', 'from_year', 'to_year', 'body_style', 'segment', 'title', 'description', 'model_url']
    metadata_fields = {field: data[field].tolist() for field in fields if field in data.columns}

    metadata_rows = []
    for i in range(len(data)):
        row = {field: metadata_fields[field][i] for field in fields if field in metadata_fields}
        metadata_rows.append(row)

    image_urls = data['image_urls'].tolist()

    yolo_helper = YOLO_helper()

    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.metadata_file):
        with open(args.metadata_file, 'w', newline='') as f:
            pass

    processed_urls = set()
    if os.path.exists(args.metadata_file) and os.stat(args.metadata_file).st_size > 0:
        processed_data = pd.read_csv(args.metadata_file)
        if 'original_url' in processed_data.columns:
            processed_urls = set(processed_data['original_url'])
            print(f"Found {len(processed_urls)} already processed items")

    total_images = len(image_urls)
    remaining_indices = [i for i, url_list in enumerate(image_urls) if not any(url.strip() in processed_urls for url in url_list.split(','))]
    print(f"Processing {len(remaining_indices)} remaining items")

    for start_idx in range(0, len(remaining_indices), args.index_batch_size):
        end_idx = min(start_idx + args.index_batch_size, len(remaining_indices))
        batch_indices = remaining_indices[start_idx:end_idx]
        print(f"Processing metadata batch {start_idx//args.index_batch_size + 1}/{(len(remaining_indices)+args.index_batch_size-1)//args.index_batch_size} (rows {batch_indices[0]}-{batch_indices[-1]}) ")
        
        batch_urls = [image_urls[i] for i in batch_indices]
        batch_metadata = [metadata_rows[i] for i in batch_indices]
        
        process_image_batch(batch_urls, batch_metadata, yolo_helper, args.temp_dir, args.output_dir, args.metadata_file, args.gpu_batch_size)
        
        if args.test_batch:
            print("Test batch completed. Set test_batch = False to process all images.")
            break
        
        time.sleep(5)

    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)

    print(f"Processing complete. Car images saved to {args.output_dir} with metadata in {args.metadata_file}")

if __name__ == "__main__":
    main()
