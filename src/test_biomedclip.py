import time, numpy as np, torch, open_clip
from PIL import Image
name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
model, _, pre = open_clip.create_model_and_transforms(name)
tok = open_clip.get_tokenizer(name)
model.eval()
imgs = [Image.fromarray(np.random.randint(0,255,(512,512,3),dtype=np.uint8)) for _ in range(8)]
x = torch.stack([pre(i) for i in imgs])
t0 = time.time()
with torch.no_grad():
    f = model.encode_image(x)
    ft = model.encode_text(tok(["glaucoma suspect"]*2, context_length=256))
dt = time.time()-t0
print("image", tuple(f.shape), "text", tuple(ft.shape))
print("sec per image:", round(dt/8,2), "-> hours for 10000:", round(dt/8*10000/3600,1))
