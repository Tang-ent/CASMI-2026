# %% [markdown]
# # casmi26-k44-bundle（M8、2026-10-19、計画 1-3-deep-storm.md）
# 自作エンジン v12 の入力（k01 cache、k05 pool、k05b、k15-s1 の fpnet2、k20 fwd、k21 frag、k25、k26、k35 の ranker7、fprich、g2、
# rdkit 2026.9.1 の wheel）を 1 つの tar（casmi26_engine_bundle.tar、非圧縮）にまとめる CPU カーネル。
# 融合 Notebook（ユーザーの fork）はこの出力を kernel source として付け、実行時に /tmp に展開して own_runner.py（k44_own）に渡す。
# tar にするのは、土台の Notebook が /kaggle/input/** をファイル名で探す（最短のパス）ので、自作のファイルを並べると名前が
# 衝突しうるため。tar の中は /kaggle/input と同じ形（<slug>/<元の相対パス>）で、casmi26.paths.find がそのまま使える。
# 各入力のカーネルのログ（<slug>.log / .txt）と ipynb は入れない。manifest（casmi26_engine_bundle.json）に全ファイルの
# 大きさと sha256 を残す（tar の中と外の両方）。

# %%
import glob
import hashlib
import io
import json
import os
import tarfile
import time

T0 = time.time()
SOURCES = ["casmi26-k01-cache", "casmi26-k05-pool", "casmi26-k05b-massfix", "casmi26-k15-fpnet2-s1", "casmi26-k20-fwd",
           "casmi26-k21-frag", "casmi26-k25-fwd2", "casmi26-k26-fp2", "casmi26-k35-eval7", "casmi26-fprich", "casmi26-g2",
           "casmi26-wheels"]
KEEP_WHEELS = ("-cp312-", "-cp313-")  # 土台の Notebook は Python 3.12 に固定、念のため 3.13 も
OUT = "/kaggle/working/casmi26_engine_bundle.tar"


def source_dir(slug):
    """/kaggle/input/<slug> or /kaggle/input/{datasets,notebooks}/<owner>/<slug> (shortest)."""
    hits = [d for d in glob.glob(f"/kaggle/input/**/{slug}", recursive=True) if os.path.isdir(d)]
    assert hits, f"input {slug} is not attached"
    return sorted(hits, key=len)[0]


def keep(slug, rel):
    name = os.path.basename(rel)
    if name in (f"{slug}.log", f"{slug}.txt", "__results__.html", "__notebook__.ipynb", "dataset-metadata.json"):
        return False
    if name.endswith((".ipynb", ".html")) or name.startswith("__"):
        return False
    if slug == "casmi26-wheels":
        return name.startswith("rdkit-2026.9.1-") and any(t in name for t in KEEP_WHEELS)
    if slug == "casmi26-k35-eval7":
        return name.startswith("ranker7_both_fprich_g2_grp_") or name in ("chosen7.json", "rule7.csv")
    return True


files = []
for slug in SOURCES:
    d = source_dir(slug)
    for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
        rel = os.path.relpath(p, d)
        if os.path.isfile(p) and keep(slug, rel):
            files.append((slug, rel, p))
    print(slug, d, sum(f[0] == slug for f in files), "files", flush=True)


def sha256(path, n=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), sources=SOURCES,
           files=[dict(path=f"{slug}/{rel}", bytes=os.path.getsize(p), sha256=sha256(p)) for slug, rel, p in files])
man["total_bytes"] = sum(f["bytes"] for f in man["files"])
print(f"{len(files)} files, {man['total_bytes'] / 1e9:.2f} GB, hashed in {time.time() - T0:.0f}s", flush=True)
blob = json.dumps(man, indent=1).encode()
with tarfile.open(OUT, "w") as tf:
    for slug, rel, p in files:
        tf.add(p, arcname=f"{slug}/{rel}")
    ti = tarfile.TarInfo("casmi26_engine_bundle.json")
    ti.size = len(blob)
    tf.addfile(ti, io.BytesIO(blob))
open("/kaggle/working/casmi26_engine_bundle.json", "wb").write(blob)
print(f"wrote {OUT} {os.path.getsize(OUT) / 1e9:.2f} GB ({time.time() - T0:.0f}s)")
for f in man["files"]:
    print(f"  {f['bytes']:>12,d}  {f['sha256'][:12]}  {f['path']}")
