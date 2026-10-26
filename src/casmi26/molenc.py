"""Molecule-side inputs of the MSAlign-type model (M9, plan 10.2; kernels k49_dreams_emb / k50_msalign).

Two molecule towers read from here:
  cb  frozen DeepChem/ChemBERTa-100M-MLM (MIT, rev f5c45f44), mean of the last hidden state over the non-padding tokens
      (attention_mask = 1, so <s> and </s> are included; the model card gives no pooling recommendation, MSAlign v2 uses
      CLS), 768-d, fp16 on the GPU.
  fp  Morgan r=2 2048 of the pool (pool_fp.npy; RDKit's default includeChirality=False, so no stereo bits) as floats.

Text canonicalisation (plan 10.2, 10.1: pool.smiles is not uniform; train-only strings carry 0% '@', COCONUT strings
51-63%, so a text model would read the provenance off the string). Truth and candidates go through one function on one
RDKit version (casmi26-wheels 2026.9.1): MolFromSmiles -> largest fragment (rdMolStandardize.LargestFragmentChooser) ->
Uncharger -> RemoveStereochemistry -> MolToSmiles(isomericSmiles=False, canonical=True). isomericSmiles=False also drops
isotope labels (isotopologues collapse onto the protium form). Tautomers: canon_text(taut=True), the default, adds chem's TautomerEnumerator (maxTautomers 250, as in
k05) after the stereo strip (the official metric's order: stereo strip -> tautomer). pool.smiles already went through
chem.canonical in k05 (pool.py:95-96; raw train inserts of pool.py:106-113 did not), but that ran on the molecule as
written (stereo, charges, salts) and is not idempotent under the steps above: on the k05 pool the pass changes 1.3% of
the COCONUT-only strings, 0.57% of the shared ones and 0.03% of the train-only ones (tools/smoke/check_molenc.py), a
provenance signal, so it stays on (cost: ~21-24 ms per molecule and process on the pool, heavy tail at high mass, vs
~0.6 ms without it); taut=False only for the comparison.

Token cap: the checkpoint has max_position_embeddings 512 and pad_token_id 1. RoBERTa numbers the positions from
padding_idx + 1 = 2, so at most 510 tokens (including <s> and </s>) fit; max_length=512 (plan text) indexes past the
position table. MAX_TOKENS = 510 (checked in check_molenc.py).
"""
import functools
import time

import numpy as np

CB_ID = "DeepChem/ChemBERTa-100M-MLM"
CB_REV = "f5c45f44d3061f0346888f5c09db17ec1146d29d"
# sha256 of model.safetensors at CB_REV (the copy in out/m9_msalign_in/chemberta, 2026-10-24); k49 asserts it
CB_SAFETENSORS_SHA256 = "cb04edf30124fbd44817ba7715b174aa80b1981b423f721bc8a6007a0c798a99"
CB_DIM = 768
MAX_TOKENS = 510  # = max_position_embeddings (512) - padding_idx (1) - 1

_R = {}


def _rdkit():
    """RDKit objects, lazily per process (kept apart from chem._rdkit so its defaults stay as they are)."""
    if not _R:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog("rdApp.*")
        _R.update(Chem=Chem, largest=rdMolStandardize.LargestFragmentChooser(),
                  uncharger=rdMolStandardize.Uncharger())
    return _R


def canon_text(smiles, taut=True):
    """The one SMILES form fed to the text tower (plan 10.2) -> str, or None when RDKit fails.
    largest fragment -> uncharge -> remove stereo -> tautomer canonicalisation (taut) -> canonical SMILES without
    stereo / isotopes."""
    r = _rdkit()
    try:
        m = r["Chem"].MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if m is None:
            return None
        m = r["largest"].choose(m)
        m = r["uncharger"].uncharge(m)
        r["Chem"].RemoveStereochemistry(m)
        if taut:
            from . import chem
            m = chem._rdkit()["taut"].Canonicalize(m)
        s = r["Chem"].MolToSmiles(m, isomericSmiles=False, canonical=True)
        return s or None
    except Exception:
        return None


def canon_pool(smiles_list, workers=4, taut=True, chunksize=100, log=print, tag="canon"):
    """canon_text over a list with a multiprocessing pool -> list of str | None (same order; no fallback string).
    A small chunksize keeps the workers balanced: the tautomer pass has a heavy tail (large polyphenols)."""
    from multiprocessing import Pool as MPool
    t0 = time.time()
    fn = functools.partial(canon_text, taut=taut)
    out = []
    n = len(smiles_list)
    with MPool(workers) as p:
        for i, s in enumerate(p.imap(fn, smiles_list, chunksize=chunksize)):
            out.append(s)
            if (i + 1) % 100_000 == 0:
                log(f"  {tag}: {i + 1:,}/{n:,} ({time.time() - t0:.0f}s)")
    nf = sum(s is None for s in out)
    log(f"  {tag}: {n:,} done, failed {nf:,} ({nf / max(n, 1):.4%}), {time.time() - t0:.0f}s, taut={taut}")
    return out


# ---------------------------------------------------------------------------------------------- ChemBERTa tower (cb)
def load_chemberta(path=None, device="cuda", fp16=True):
    """(tokenizer, model) of ChemBERTa-100M-MLM in eval mode. path: a local snapshot directory (Kaggle dataset); None
    -> the HF id at the pinned revision (needs internet). Only from_pretrained(path | id, revision=...) and
    add_pooling_layer=False are used (the same in transformers 5.0 and 5.16); the dtype is set by .half() afterwards,
    not by a from_pretrained argument (renamed across versions)."""
    from transformers import AutoModel, AutoTokenizer
    src, kw = (path, {}) if path else (CB_ID, dict(revision=CB_REV))
    tok = AutoTokenizer.from_pretrained(src, **kw)
    model = AutoModel.from_pretrained(src, add_pooling_layer=False, **kw)
    model.eval()
    if fp16:
        model.half()
    return tok, model.to(device)


def token_lengths(tok, texts, chunk=50_000):
    """Number of tokens of each text including <s> and </s>, without truncation (int32)."""
    out = []
    for i in range(0, len(texts), chunk):
        enc = tok(list(texts[i:i + chunk]), add_special_tokens=True, truncation=False)["input_ids"]
        out.append(np.fromiter((len(e) for e in enc), np.int32, len(enc)))
    return np.concatenate(out) if out else np.zeros(0, np.int32)


def embed_smiles(texts, tok, model, batch=256, device="cuda", max_length=MAX_TOKENS, log=None):
    """Mean-pooled last hidden state (over attention_mask = 1) of each text -> (n, 768) float16 numpy, in input order.
    texts should already be canon_text strings (None is not allowed). Batches are built on the length-sorted order
    (the character length; padding per batch, right side); the masked sum and the division run in fp32."""
    import torch
    n = len(texts)
    out = np.zeros((n, CB_DIM), np.float16)
    order = np.argsort(np.fromiter((len(t) for t in texts), np.int64, n), kind="stable")
    t0 = time.time()
    with torch.inference_mode():
        for bi, i in enumerate(range(0, n, batch)):
            idx = order[i:i + batch]
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            out[idx] = e.cpu().numpy().astype(np.float16)
            if log and (bi + 1) % 200 == 0:
                log(f"  embed: {i + len(idx):,}/{n:,} ({time.time() - t0:.0f}s)")
    return out


# ---------------------------------------------------------------------------------------------- fingerprint tower (fp)
def fp_bits(P, pids, dtype=np.float32):
    """Morgan r=2 2048 bits of pool ids (pool_fp.npy, packed 256 bytes per row) -> (len(pids), 2048) in {0, 1}."""
    return np.unpackbits(P.fp[np.asarray(pids, np.int64)], axis=1).astype(dtype)
