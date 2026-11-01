import numpy as np
import os
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Load dataset metadata
print("Loading dataset metadata...")
ds = load_dataset("harvardairobotics/FairVLMed", "default", split='train')

# Create output folder
os.makedirs("all_images", exist_ok=True)

# Data directory
data_dir = "./data/FairVLMed/Dataset/dataset/Training"

print(f"Extracting {len(ds)} images to 'all_images/' folder...")
count = 0

for item in tqdm(ds, desc="Converting Images"):
    filename = item['filename']
    npz_path = os.path.join(data_dir, filename)
    
    # Create the output path (change .npz to .png)
    output_filename = filename.replace('.npz', '.png')
    output_path = os.path.join("all_images", output_filename)
    
    # Skip if already exists
    if os.path.exists(output_path):
        continue
    
    if os.path.exists(npz_path):
        try:
            npz_data = np.load(npz_path)
            
            # Get the image array
            if 'slo_fundus' in npz_data.files:
                image_array = npz_data['slo_fundus']
            elif 'image' in npz_data.files:
                image_array = npz_data['image']
            else:
                image_array = npz_data[npz_data.files[0]]
            
            # Convert to PIL Image
            if len(image_array.shape) == 2:
                img = Image.fromarray(image_array).convert('RGB')
            else:
                img = Image.fromarray(image_array)
                
            img.save(output_path)
            count += 1
        except Exception as e:
            pass # Skip errors silently to keep it fast

print(f"\nDone! Successfully converted {count} images to 'all_images/' folder.")