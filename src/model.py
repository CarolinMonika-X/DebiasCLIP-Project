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
        
        # 4. Define the UPGRADED trainable adapter (Dual-Modal)
        # Input is now 1024 (512 image + 512 text)
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),   # Increased from 256 to 512
            nn.LayerNorm(512),                   # Added LayerNorm for stability
            nn.ReLU(),
            nn.Dropout(0.3),                     # Increased dropout to prevent overfitting
            nn.Linear(512, 1)                    # Single output for binary classification
        )
        
        # 5. Learnable residual weight (Like CLIP-Adapter)
        self.alpha = nn.Parameter(torch.tensor(0.1))

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
        adapter_out = self.adapter(combined_features)
        
        # Residual Connection: Add a small portion of the image feature mean to the logits
        # This helps the model retain CLIP's original knowledge
        logits = adapter_out + self.alpha * image_features.mean(dim=-1, keepdim=True)
        
        return logits.squeeze(-1) # Remove the last dimension