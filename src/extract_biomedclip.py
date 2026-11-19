import os, sys, time, argparse
import numpy as np, torch, open_clip
from PIL import Image
from datasets import load_dataset

ROOT = os.path.join("data", "FairVLMed", "Dataset", "dataset")
FOLDERS = {"train": "Training", "val": "Validation", "test": "Test"}
NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
GMAP = {"no": 0, "yes": 1}
RMAP = {"asian": 0, "black": 1, "white": 2}
CHUNK = 256

def load_img(path):
    z = np.load(path)
    key = "slo_fundus" if "slo_fundus" in z.files else ("image" if "image" in z.files else z.files[0])
    a = z[key]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if a.max() <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a).convert("RGB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-field", default="gpt4_summary", choices=["gpt4_summary", "note"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N rows per split")
    args = ap.parse_args()
    out_dir = "features_biomedclip" + ("_smoke" if args.limit else "")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
    fns = ds["filename"]; texts = ds[args.text_field]
    labels_all = np.array([GMAP[str(v).lower()] for v in ds["glaucoma"]])
    races_all = np.array([RMAP[str(v).lower()] for v in ds["race"]])

    from torch.utils.data import random_split
    n = len(ds); n_tr, n_va = int(0.7 * n), int(0.1 * n); n_te = n - n_tr - n_va
    parts = random_split(range(n), [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42))
    idx = {s: list(p.indices) for s, p in zip(("train", "val", "test"), parts)}
    print({s: len(v) for s, v in idx.items()})
    cached = torch.load("cached_features.pt", weights_only=True)
    for s in FOLDERS:
        ok = (np.array_equal(cached[s]["labels"].numpy(), labels_all[idx[s]]) and
              np.array_equal(cached[s]["races"].numpy(), races_all[idx[s]]))
        print("alignment with cached_features.pt for", s, ":", ok)
        if not ok:
            sys.exit("STOP: alignment failed. Send me this output.")
    where = {}
    for f in FOLDERS.values():
        for fn in os.listdir(os.path.join(ROOT, f)):
            if fn.endswith(".npz"):
                where[fn] = f

    model, _, pre = open_clip.create_model_and_transforms(NAME)
    tok = open_clip.get_tokenizer(NAME)
    model.eval()
    norm = lambda f: torch.nn.functional.normalize(f, dim=-1).numpy()

    t_start = time.time(); done = 0
    for s in FOLDERS:
        rows = idx[s][:args.limit] if args.limit else idx[s]
        for start in range(0, len(rows), CHUNK):
            path = os.path.join(out_dir, f"{s}_{start:05d}.npz")
            if os.path.exists(path):
                continue
            sub = rows[start:start + CHUNK]
            imgs_out, txt_out = [], []
            for b in range(0, len(sub), args.bs):
                bi = sub[b:b + args.bs]
                x = torch.stack([pre(load_img(os.path.join(ROOT, where[fns[i]], fns[i]))) for i in bi])
                t = tok([str(texts[i]) for i in bi], context_length=256)
                with torch.no_grad():
                    imgs_out.append(norm(model.encode_image(x)))
                    txt_out.append(norm(model.encode_text(t)))
            np.savez(path, img=np.concatenate(imgs_out), txt=np.concatenate(txt_out))
            done += len(sub)
            el = time.time() - t_start
            print(f"{s} {start + len(sub)}/{len(rows)}  elapsed {el/60:.1f} min  ({el/max(done,1):.2f}s/case)")

    if args.limit:
        print("Smoke test OK. Chunks in", out_dir); return
    out = {}
    for s in FOLDERS:
        parts = sorted(f for f in os.listdir(out_dir) if f.startswith(s + "_"))
        arr = [np.load(os.path.join(out_dir, p)) for p in parts]
        img = np.concatenate([a["img"] for a in arr]); txt = np.concatenate([a["txt"] for a in arr])
        assert len(img) == len(idx[s]), (s, len(img))
        out[s] = {"img": torch.tensor(img), "txt": torch.tensor(txt),
                  "labels": torch.tensor(labels_all[idx[s]]).long(),
                  "races": torch.tensor(races_all[idx[s]]).long()}
    torch.save(out, "cached_features_biomedclip.pt")
    print("Saved cached_features_biomedclip.pt", {s: tuple(out[s]["img"].shape) for s in out})

if __name__ == "__main__":
    main()
