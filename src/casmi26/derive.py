"""Class-3 candidate generation (M5 P0-2, k19_gen_probe): derivatives of parent structures by up to two reaction
steps whose summed mass change matches target mass - parent mass.

Steps (own list, natural-product style modifications):
- additions as reaction SMARTS on a site (one product per matching site, i.e. regioisomers are separate candidates):
  hydroxylation, methylation, (de)hydrogenation, hydration, alcohol <-> ketone, glycosylation (hexose, deoxyhexose,
  pentose, glucuronic acid, two disaccharides), acylation (acetyl, malonyl, coumaroyl, caffeoyl, feruloyl, galloyl),
  sulfation, prenylation, dehydroxylation
- removals by one generic cleavage: cut one acyclic single bond between heavy atoms, keep either side (implicit H on the
  cut atom) and accept it when the removed mass equals the step's group (methyl, sugars, acyls, sulfate, prenyl).
  This covers demethylation, deglycosylation and deacylation with one function.
The planner only applies step sequences whose summed delta matches the needed mass change (tolerance in Da), so most
reactions are never run for a given parent. Stereo is dropped (the metric key ignores it).
"""
import itertools

import numpy as np

from .chem import formula_mass

# name, added formula (sign +1) or removed formula (sign -1), reaction SMARTS (None = generic removal)
_ADD = [
    ("OH_arom", "O", "[cH:1]>>[c:1]O"),
    ("OH_aliph", "O", "[CX4;!H0:1]>>[C:1]O"),
    ("Me_O", "CH2", "[OX2H1:1]>>[O:1]C"),
    ("Me_N", "CH2", "[NX3;H1,H2;!$(N-C=O):1]>>[N:1]C"),
    ("Me_C_arom", "CH2", "[cH:1]>>[c:1]C"),
    ("H2_add", "H2", "[C:1]=[C:2]>>[C:1]-[C:2]"),
    ("H2_ketone", "H2", "[#6:3][CX3:1](=[OX1:2])[#6:4]>>[#6:3][C:1](-[O:2])[#6:4]"),
    ("hydration", "H2O", "[C:1]=[C:2]>>[C:1](O)-[C:2]"),
    ("hexose", "C6H10O5", "[OX2H1:1]>>[O:1]C1OC(CO)C(O)C(O)C1O"),
    ("deoxyhexose", "C6H10O4", "[OX2H1:1]>>[O:1]C1OC(C)C(O)C(O)C1O"),
    ("pentose", "C5H8O4", "[OX2H1:1]>>[O:1]C1OCC(O)C(O)C1O"),
    ("glucuronide", "C6H8O6", "[OX2H1:1]>>[O:1]C1OC(C(=O)O)C(O)C(O)C1O"),
    ("rutinose", "C12H20O9", "[OX2H1:1]>>[O:1]C1OC(COC2OC(C)C(O)C(O)C2O)C(O)C(O)C1O"),
    ("dihexose", "C12H20O10", "[OX2H1:1]>>[O:1]C1OC(COC2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O"),
    ("acetyl", "C2H2O", "[OX2H1:1]>>[O:1]C(C)=O"),
    ("malonyl", "C3H2O3", "[OX2H1:1]>>[O:1]C(=O)CC(=O)O"),
    ("sulfate", "SO3", "[OX2H1:1]>>[O:1]S(=O)(=O)O"),
    ("prenyl_O", "C5H8", "[OX2H1:1]>>[O:1]CC=C(C)C"),
    ("prenyl_C", "C5H8", "[cH:1]>>[c:1]CC=C(C)C"),
    ("coumaroyl", "C9H6O2", "[OX2H1:1]>>[O:1]C(=O)C=Cc1ccc(O)cc1"),
    ("caffeoyl", "C9H6O3", "[OX2H1:1]>>[O:1]C(=O)C=Cc1ccc(O)c(O)c1"),
    ("feruloyl", "C10H8O3", "[OX2H1:1]>>[O:1]C(=O)C=Cc1ccc(O)c(OC)c1"),
    ("galloyl", "C7H4O4", "[OX2H1:1]>>[O:1]C(=O)c1cc(O)c(O)c(O)c1"),
]
_SUB = [  # mass decreases by a SMARTS
    ("H2_remove", "H2", "[CX4;!H0:1]-[CX4;!H0:2]>>[C:1]=[C:2]"),
    ("H2_alcohol", "H2", "[CX4;H1,H2:1]-[OX2H1:2]>>[C:1]=[O:2]"),
    ("dehydroxy", "O", "[#6:1]-[OX2H1]>>[#6:1]"),
]
_REMOVE = [  # generic cleavage: removed group formula (= side piece + H - H on the kept atom)
    ("de_Me", "CH2"), ("de_hexose", "C6H10O5"), ("de_deoxyhexose", "C6H10O4"), ("de_pentose", "C5H8O4"),
    ("de_glucuronide", "C6H8O6"), ("de_rutinose", "C12H20O9"), ("de_dihexose", "C12H20O10"),
    ("de_acetyl", "C2H2O"), ("de_malonyl", "C3H2O3"), ("de_sulfate", "SO3"), ("de_prenyl", "C5H8"),
    ("de_coumaroyl", "C9H6O2"), ("de_caffeoyl", "C9H6O3"), ("de_feruloyl", "C10H8O3"), ("de_galloyl", "C7H4O4"),
]


def _steps():
    out = []
    for name, f, sm in _ADD:
        out.append(dict(name=name, delta=formula_mass(f), smarts=sm))
    for name, f, sm in _SUB:
        out.append(dict(name=name, delta=-formula_mass(f), smarts=sm))
    for name, f in _REMOVE:
        out.append(dict(name=name, delta=-formula_mass(f), smarts=None))
    return out


STEPS = _steps()
STEP_OF = {s["name"]: s for s in STEPS}
_DELTA = np.array([s["delta"] for s in STEPS])
_PAIRS = [(i, j) for i, j in itertools.product(range(len(STEPS)), repeat=2)]
_PAIR_DELTA = np.array([_DELTA[i] + _DELTA[j] for i, j in _PAIRS])
_R = {}


def _rd():
    if not _R:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, Descriptors
        RDLogger.DisableLog("rdApp.*")
        _R.update(Chem=Chem, mw=Descriptors.ExactMolWt,
                  rxn={s["name"]: AllChem.ReactionFromSmarts(s["smarts"]) for s in STEPS if s["smarts"]})
    return _R


def sequences(delta, tol):
    """Step-name tuples (1 or 2 steps) whose summed mass change is within tol (Da) of delta. Pairs are ordered
    (both orders are kept: a removal before or after an addition can give different products). Excludes delta ~ 0."""
    if abs(delta) <= tol:
        return []
    out = [(STEPS[i]["name"],) for i in np.flatnonzero(np.abs(_DELTA - delta) <= tol)]
    out += [(STEPS[_PAIRS[k][0]]["name"], STEPS[_PAIRS[k][1]]["name"])
            for k in np.flatnonzero(np.abs(_PAIR_DELTA - delta) <= tol)]
    return out


def n_steps(delta, tol):
    """Fewest steps (0, 1 or 2) whose summed mass change is within tol of delta; None when no sequence fits."""
    if abs(delta) <= tol:
        return 0
    if (np.abs(_DELTA - delta) <= tol).any():
        return 1
    if (np.abs(_PAIR_DELTA - delta) <= tol).any():
        return 2
    return None


def all_deltas():
    """Every 1-step and 2-step summed mass change (for the parent windows of the pool source) with its sequences."""
    d = {}
    for i, s in enumerate(STEPS):
        d.setdefault(round(s["delta"], 5), []).append((s["name"],))
    for k, (i, j) in enumerate(_PAIRS):
        d.setdefault(round(_PAIR_DELTA[k], 5), []).append((STEPS[i]["name"], STEPS[j]["name"]))
    return d


def _smiles(m):
    Chem = _rd()["Chem"]
    try:
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m, isomericSmiles=False)
    except Exception:
        return None


def _apply(mol, step, max_out):
    """Products (non-isomeric canonical SMILES -> mol) of one step on mol."""
    r = _rd()
    Chem = r["Chem"]
    out = {}
    st = STEP_OF[step]
    if st["smarts"] is not None:
        for prods in r["rxn"][step].RunReactants((mol,), maxProducts=4 * max_out):
            p = prods[0]
            smi = _smiles(p)
            if smi and smi not in out:
                m2 = Chem.MolFromSmiles(smi)
                if m2 is not None:
                    out[smi] = m2
                    if len(out) >= max_out:
                        break
        return out
    # generic removal: cut one acyclic single bond between heavy atoms, keep the side whose loss matches
    want = -st["delta"]
    mw = r["mw"](mol)
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        rw = Chem.RWMol(mol)
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        rw.RemoveBond(a1, a2)
        for a in (a1, a2):
            at = rw.GetAtomWithIdx(a)
            at.SetNoImplicit(False)
            at.SetNumExplicitHs(0)
        try:
            frags = Chem.GetMolFrags(rw.GetMol(), asMols=True, sanitizeFrags=True)
        except Exception:
            continue
        for f in frags:
            if abs((mw - r["mw"](f)) - want) < 1e-3:
                smi = Chem.MolToSmiles(f, isomericSmiles=False)
                if smi not in out:
                    m2 = Chem.MolFromSmiles(smi)
                    if m2 is not None:
                        out[smi] = m2
        if len(out) >= max_out:
            break
    return out


def generate(parent_smiles, seqs, max_per_parent=100):
    """Products of a parent under the step sequences -> dict(smiles -> sequence tuple), at most max_per_parent.
    Products are non-isomeric canonical SMILES (stereo dropped). Charged / unparsable parents give nothing."""
    Chem = _rd()["Chem"]
    mol = Chem.MolFromSmiles(parent_smiles) if isinstance(parent_smiles, str) else None
    if mol is None:
        return {}
    out = {}
    for seq in seqs:
        level = {Chem.MolToSmiles(mol, isomericSmiles=False): mol}
        for step in seq:
            nxt = {}
            for m in level.values():
                for smi, m2 in _apply(m, step, max_per_parent).items():
                    nxt.setdefault(smi, m2)
                if len(nxt) >= max_per_parent:
                    break
            level = nxt
            if not level:
                break
        for smi in level:
            if smi not in out:
                out[smi] = seq
                if len(out) >= max_per_parent:
                    return out
    return out


def exact_mass(smiles):
    Chem = _rd()["Chem"]
    m = Chem.MolFromSmiles(smiles)
    return _rd()["mw"](m) if m is not None else np.nan


def noniso(smiles):
    """Non-isomeric canonical SMILES (the matching key for recall; stereo dropped, tautomers not unified)."""
    Chem = _rd()["Chem"]
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


def ik14(smiles):
    """Standard InChIKey first block (no tautomer canonicalisation; fast). None on failure."""
    Chem = _rd()["Chem"]
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    try:
        k = Chem.MolToInchiKey(m)
        return k[:14] if k else None
    except Exception:
        return None
