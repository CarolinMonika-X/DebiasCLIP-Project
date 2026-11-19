p = "src/extract_biomedclip.py"
t = open(p, encoding="utf-8").read()
a = t.index("    where = {}")
b = t.index("    idx = chosen\n") + len("    idx = chosen\n")
new = '''    from torch.utils.data import random_split
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
'''
t = t[:a] + new + t[b:]
t = t.replace("os.path.join(ROOT, FOLDERS[s], fns[i])", "os.path.join(ROOT, where[fns[i]], fns[i])")
open(p, "w", encoding="utf-8").write(t)
print("patched")
