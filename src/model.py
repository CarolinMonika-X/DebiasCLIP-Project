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
        
        # 2. Freeze the entire CLIP model first
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 3. Selective Fine-Tuning: Unfreeze ONLY the last 2 transformer blocks
        print("Unfreezing last 2 transformer blocks of CLIP Image Encoder...")
        
        vision_blocks = self.clip_model.visual.transformer.resblocks
        num_blocks = len(vision_blocks)
        
        # Last 2 blocks-a unfreeze pannu
        for block in vision_blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
                
        # Final LayerNorm-a kooda unfreeze pannalam (stability ku)
        if hasattr(self.clip_model.visual, 'ln_post'):
            for param in self.clip_model.visual.ln_post.parameters():
                param.requires_grad = True
                
        print(f"Total vision blocks: {num_blocks}. Unfroze blocks {num_blocks-2} and {num_blocks-1}.")
            
        # 4. Get the embedding dimension
        embedding_dim = self.clip_model.text_projection.shape[1]
        print(f"CLIP embedding dimension: {embedding_dim}")
        
        # 5. Define the trainable adapter (Dual-Modal)
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),   
            nn.LayerNorm(512),                   
            nn.ReLU(),
            nn.Dropout(0.3),                     
            nn.Linear(512, 1)                    
        )
        
        # 6. Learnable residual weight
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, images, texts):
        # Image features (Last 2 blocks train aagum)
        image_features = self.clip_model.encode_image(images)
        
        # Text features (Fully frozen)
        with torch.no_grad():
            text_tokens = open_clip.tokenize(texts).to(images.device)
            text_features = self.clip_model.encode_text(text_tokens)
            
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # --- Dual-Modal Implementation ---
        combined_features = torch.cat([image_features, text_features], dim=-1)
        
        # Pass through adapter
        adapter_out = self.adapter(combined_features)
        
        # Residual Connection
        logits = adapter_out + self.alpha * image_features.mean(dim=-1, keepdim=True)
        
        return logits.squeeze(-1)