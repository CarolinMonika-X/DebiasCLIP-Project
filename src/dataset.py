import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets import load_dataset

class FairVLMedDataset(Dataset):
    def __init__(self, split='train', transform=None):
        # Load metadata
        self.ds = load_dataset("harvardairobotics/FairVLMed", "default", split=split)
        self.transform = transform
        
        # --- PATH FIX: Split-a vechu folder-a select pannuvom ---
        # HuggingFace split names: 'train', 'validation', 'test'
        # Unga folder names: 'Training', 'Validation', 'Test'
        split_folder_map = {
            'train': 'Training',
            'validation': 'Validation',
            'test': 'Test'
        }
        folder_name = split_folder_map.get(split, 'Training')
        
        self.data_dir = os.path.join(".", "data", "FairVLMed", "Dataset", "dataset", folder_name)
        print(f"Data directory set to: {self.data_dir}")

        # Label mappings
        self.glaucoma_map = {'no': 0, 'yes': 1}
        self.race_map = {'asian': 0, 'black': 1, 'white': 2}
        self.gender_map = {'male': 0, 'female': 1}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        
        # 1. Load Image
        image_path = os.path.join(self.data_dir, item['filename'])
        
        try:
            npz_data = np.load(image_path)
            
            # Key check (slo_fundus or image)
            if 'slo_fundus' in npz_data.files:
                image_data = npz_data['slo_fundus']
            elif 'image' in npz_data.files:
                image_data = npz_data['image']
            else:
                image_data = npz_data[npz_data.files[0]]

            if len(image_data.shape) == 2:
                image = Image.fromarray(image_data).convert('RGB')
            else:
                image = Image.fromarray(image_data)
                
        except Exception as e:
            # Ippo error-a print pannuvom, path correct-a irundha idhu varakoodadhu
            print(f"Error loading {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        # Transform apply
        if self.transform:
            image = self.transform(image)
        else:
            image = image.resize((224, 224))
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        # 2. Text and Labels
        text = item['gpt4_summary']
        label = torch.tensor(self.glaucoma_map.get(item['glaucoma'].lower(), 0), dtype=torch.float32)
        demographic_label = torch.tensor(self.race_map.get(item['race'].lower(), 0), dtype=torch.long)
        
        return image, text, label, demographic_label