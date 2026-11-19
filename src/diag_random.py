import torch, numpy as np
from torch.utils.data import random_split
from datasets import load_dataset
G={"no":0,"yes":1}; R={"asian":0,"black":1,"white":2}
ds=load_dataset("harvardairobotics/FairVLMed","default",split="train")
y=np.array([G[str(v).lower()] for v in ds["glaucoma"]]); r=np.array([R[str(v).lower()] for v in ds["race"]])
n=len(ds); n_tr,n_va=int(0.7*n),int(0.1*n); n_te=n-n_tr-n_va
parts=random_split(range(n),[n_tr,n_va,n_te],generator=torch.Generator().manual_seed(42))
c=torch.load("cached_features.pt",weights_only=True)
for s,p in zip(("train","val","test"),parts):
    ids=list(p.indices)
    eq=(c[s]["labels"].numpy()==y[ids])&(c[s]["races"].numpy()==r[ids])
    print(s,"n=",len(ids),"position-match with random_split(seed 42):",round(float(eq.mean()),4))
