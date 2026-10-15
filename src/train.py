import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
from dataset import FairVLMedDataset
from model import DebiasCLIP

# --- Configuration ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Model Load Pannunga ---
print("Loading DebiasCLIP model...")
model = DebiasCLIP().to(DEVICE)
preprocess = model.preprocess

# --- Dataset Load Pannunga ---
print("Loading dataset...")
full_dataset = FairVLMedDataset(split='train', transform=preprocess)

# Split the dataset into train (80%) and validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Optimizer and Loss ---
optimizer = AdamW(model.adapter.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()

def fairness_loss(logits, demographic_labels):
    unique_demographics = torch.unique(demographic_labels)
    group_preds = []
    for demo in unique_demographics:
        group_mask = (demographic_labels == demo)
        if group_mask.sum() > 0:
            group_preds.append(logits[group_mask].mean())
    if len(group_preds) < 2:
        return torch.tensor(0.0, device=logits.device)
    preds_tensor = torch.stack(group_preds)
    return preds_tensor.var()

# --- Training Loop ---
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    total_fairness_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for images, texts, labels, demographics in progress_bar:
        images, labels, demographics = images.to(DEVICE), labels.to(DEVICE), demographics.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(images, texts)
        
        cls_loss = criterion(logits, labels)
        fair_loss = fairness_loss(logits, demographics)
        loss = cls_loss + 0.1 * fair_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_fairness_loss += fair_loss.item()
        progress_bar.set_postfix({'loss': loss.item(), 'fair_loss': fair_loss.item()})

    avg_loss = total_loss / len(train_loader)
    avg_fair_loss = total_fairness_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Fairness Loss: {avg_fair_loss:.4f}")

    # --- Validation Loop ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, texts, labels, demographics in val_loader:
            images, labels, demographics = images.to(DEVICE), labels.to(DEVICE), demographics.to(DEVICE)
            logits = model(images, texts)
            loss = criterion(logits, labels)
            val_loss += loss.item()
    
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

print("Training complete! 🎉")