from datasets import load_dataset

print("Loading dataset from cache...")
ds = load_dataset("harvardairobotics/FairVLMed", "default")

print("\n--- Dataset Structure ---")
print(ds)

print("\n--- Sample Data (First Row) ---")
sample = ds['train'][0]
print("Filename:", sample['filename'])
print("Glaucoma:", sample['glaucoma'])
print("Gender:", sample['gender'])
print("Race:", sample['race'])