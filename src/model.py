import torch
import torch.nn as nn
import open_clip

class DebiasCLIP(nn.Module):
    def __init__(self, clip_model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        super().__init__()
        
        # 1. Load the pre-trained CLIP model
        print(f"Loading CLIP model: {clip_model_name}...")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        
        # 2. Freeze the entire CLIP model (We don't train this part)
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 3. Get the embedding dimension
        embedding_dim = self.clip_model.text_projection.shape[1]
        print(f"CLIP embedding dimension: {embedding_dim}")
        
        # 4. Define the trainable adapter (Dual-Modal)
        # Input is now 1024 (512 image + 512 text)
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256), # 512 * 2 = 1024
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1) # Single output for binary classification
        )

    def forward(self, images, texts):
        # Get image and text features from the frozen CLIP model
        with torch.no_grad():
            # Image features
            image_features = self.clip_model.encode_image(images)
            
            # Text features
            text_tokens = open_clip.tokenize(texts).to(images.device)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # --- Dual-Modal Implementation ---
        # Combine image and text features along the feature dimension
        combined_features = torch.cat([image_features, text_features], dim=-1)
        
        # Pass the combined features through our trainable adapter
        logits = self.adapter(combined_features)
        
        return logits.squeeze(-1) # Remove the last dimension