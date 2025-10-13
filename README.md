# Scripts for car dataset processing

# 1. Install required dependencies

`pip install -r requirements.txt`

# 2. Downloading the data 

You can download the processed car dataset from [car_images](https://huggingface.co/datasets/DamianBoborzi/car_images).

There you should already find the images with metadata with more metadata in the zip file. 

If you want to load the original data, you can use the `process_car_data.py` script. For this you first need to load the `data_full.csv`file from [car_models_3887](https://huggingface.co/datasets/Unit293/car_models_3887) found [here](https://huggingface.co/datasets/Unit293/car_models_3887/resolve/main/data_full.csv?download=true). 

The `process_car_data.py`script uses yolo11x from ultralytics to detect cars and rembg to remove the background. You have to download the yolo11x weights from [here](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt). You can find more information on these model [here](https://docs.ultralytics.com/de/models/yolo11/#supported-tasks-and-modes). If you prefer to use jupyter notebooks, you can also use the `process_combined_car_models.ipynb`notebook. It should do the same thing. 

# 3. Processing the images and generating captions

To create aesthetic scores and captions for the images, run the following command:
```sh
# Update these paths to match your setup
CSV_PATH = "car_models_3887.csv"  # Your CSV file
IMAGES_DIR = "car_images"  # Your images directory
OUTPUT_DIR = "output"
CHECKPOINT_DIR = "checkpoints"

python main.py
```

# 4. Postprocessing 

The `process_car_data.py` and `process_combined_car_models.ipynb`script already perform some processing steps like removing the background using rembg. 

Further postprocessing like removing the background using sam2 and increasing the image size using SRGAN can be done using the scripts in the `postprocessing` folder. 
