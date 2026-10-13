"""Masses, adducts, fingerprints and the metric's structure key."""
import re

import numpy as np

E_MASS = 0.000548579909
MONO = {
    "H": 1.00782503207, "D": 2.0141017778, "B": 11.0093054, "C": 12.0, "N": 14.0030740048,
    "O": 15.99491461956, "F": 18.99840322, "Na": 22.9897692809, "Mg": 23.98504170, "Al": 26.98153863,
    "Si": 27.9769265325, "P": 30.97376163, "S": 31.97207100, "Cl": 34.96885268, "K": 38.96370668,
    "Ca": 39.96259098, "Fe": 55.9349375, "Ni": 57.9353429, "Co": 58.9331950, "Cu": 62.9295975,
    "Zn": 63.9291422, "As": 74.9215965, "Br": 78.9183371, "Se": 79.9165213, "Mo": 97.9054082,
    "Sn": 119.9021947, "I": 126.904473,
}
PROTON = MONO["H"] - E_MASS
H2O = 2 * MONO["H"] + MONO["O"]
NH4 = MONO["N"] + 4 * MONO["H"]
HCOOH = MONO["C"] + 2 * MONO["H"] + 2 * MONO["O"]
CH3COOH = 2 * MONO["C"] + 4 * MONO["H"] + 2 * MONO["O"]

# name -> (n_M, |charge|, delta):  m/z = (n_M * M + delta) / |charge|
ADDUCTS = {
    "[M+H]+": (1, 1, PROTON),
    "[M+NH4]+": (1, 1, NH4 - E_MASS),
    "[M+Na]+": (1, 1, MONO["Na"] - E_MASS),
    "[M+K]+": (1, 1, MONO["K"] - E_MASS),
    "[M-H2O+H]+": (1, 1, PROTON - H2O),
    "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M]+": (1, 1, -E_MASS),
    "[M+2H]2+": (1, 2, 2 * PROTON),
    "[2M+H]+": (2, 1, PROTON),
    "[2M+Na]+": (2, 1, MONO["Na"] - E_MASS),
    "[M-H]-": (1, 1, -PROTON),
    "[M-H2O-H]-": (1, 1, -PROTON - H2O),
    "[M+CH2O2-H]-": (1, 1, HCOOH - PROTON),
    "[M+C2H4O2-H]-": (1, 1, CH3COOH - PROTON),
    "[M+Cl]-": (1, 1, MONO["Cl"] + E_MASS),
    "[M]-": (1, 1, E_MASS),
    "[2M-H]-": (2, 1, -PROTON),
}
TEST_ADDUCTS = [
    "[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
    "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-",
]

_FORMULA = re.compile(r"^((?:[A-Z][a-z]?\d*)+)([+-]\d*)?$")
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula):
    """Monoisotopic mass of a formula such as 'C21H28O5' or 'C5H14NO+' (charged: electron mass removed).
    NaN when it does not parse or contains an untabulated element."""
    if not isinstance(formula, str):
        return np.nan
    m = _FORMULA.match(formula.strip())
    if m is None:
        return np.nan
    mass = 0.0
    for el, n in _TOKEN.findall(m.group(1)):
        if el not in MONO:
            return np.nan
        mass += MONO[el] * (int(n) if n else 1)
    charge = m.group(2)
    if charge:
        z = int(charge[1:]) if len(charge) > 1 else 1
        mass -= (z if charge[0] == "+" else -z) * E_MASS
    return mass


def neutral_mass(precursor_mz, adduct):
    """Neutral mass M from the precursor m/z; NaN for an adduct outside ADDUCTS."""
    a = ADDUCTS.get(str(adduct).replace(" ", ""))
    if a is None or not np.isfinite(precursor_mz):
        return np.nan
    n, z, delta = a
    return (float(precursor_mz) * z - delta) / n


# ---------------------------------------------------------------- RDKit (imported lazily)
FP_BITS = 2048
_R = {}


MAX_TAUTOMERS = 250  # 巨大なポリフェノールで正規化が数十秒かかるのを防ぐ上限（既定は 1000）


def _rdkit():
    if not _R:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        RDLogger.DisableLog("rdApp.*")
        params = rdMolStandardize.CleanupParameters()
        params.maxTautomers = MAX_TAUTOMERS
        params.maxTransforms = MAX_TAUTOMERS
        _R.update(Chem=Chem, gen=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS),
                  taut=rdMolStandardize.TautomerEnumerator(params), formula=CalcMolFormula)
    return _R


def _fp_of_mol(m):
    return np.packbits(_rdkit()["gen"].GetFingerprintAsNumPy(m).astype(np.uint8))


def morgan_packed(smiles):
    """Packed Morgan r=2 fingerprint (FP_BITS // 8 bytes); zeros when RDKit cannot parse."""
    r = _rdkit()
    m = r["Chem"].MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return np.zeros(FP_BITS // 8, np.uint8)
    return _fp_of_mol(m)


def canonical(smiles):
    """Tautomer-canonical form of a SMILES -> dict(smiles, key, formula, fp, fp_raw) or None.
    key = InChIKey14 of the canonical tautomer (== score_key); fp is computed on the canonical tautomer so that
    the same molecule written in different styles (train vs COCONUT) gets one fingerprint; fp_raw is the
    fingerprint of the input as written (for the before/after check)."""
    r = _rdkit()
    try:
        m = r["Chem"].MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if m is None:
            return None
        fp_raw = _fp_of_mol(m)
        c = r["taut"].Canonicalize(m)
        key = r["Chem"].MolToInchiKey(c)[:14]
        if not key:
            return None
        return dict(smiles=r["Chem"].MolToSmiles(c), key=key, formula=r["formula"](c), fp=_fp_of_mol(c),
                    fp_raw=fp_raw)
    except Exception:
        return None


def parent_smiles(smiles):
    """Submission hygiene (M7): a multi-component SMILES (salt / mixture, e.g. '...Cl' or '[Na+]...') can never match,
    because the official metric compares heavy-atom compositions and the answers are single-component. -> the largest
    component, uncharged where possible (None if RDKit cannot parse it); single-component SMILES are returned as is."""
    if not isinstance(smiles, str) or "." not in smiles:
        return smiles
    from rdkit.Chem.MolStandardize import rdMolStandardize
    r = _rdkit()
    try:
        m = r["Chem"].MolFromSmiles(smiles)
        if m is None:
            return None
        m = rdMolStandardize.LargestFragmentChooser().choose(m)
        m = rdMolStandardize.Uncharger().uncharge(m)
        return r["Chem"].MolToSmiles(m)
    except Exception:
        return None


def score_key(smiles):
    """Metric key: InChIKey14 after RDKit tautomer canonicalisation (None on failure)."""
    c = canonical(smiles)
    return c["key"] if c else None


ALT_FPS = ("morgan3_4096", "morgan2+maccs+ap")


def alt_fps(smiles):
    """Alternative packed fingerprints of a SMILES (M4 fingerprint-ceiling check; zeros when RDKit cannot parse):
    Morgan r=3 4096 bits, and Morgan r=2 2048 + MACCS 167 + atom pairs 2048 concatenated (4263 bits)."""
    r = _rdkit()
    if "g3" not in r:
        from rdkit import DataStructs
        from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
        r.update(g3=rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096),
                 ap=rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048), maccs=MACCSkeys.GenMACCSKeys,
                 ds=DataStructs)
    m = r["Chem"].MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return np.zeros(512, np.uint8), np.zeros((2048 + 167 + 2048 + 7) // 8, np.uint8)
    mac = np.zeros(167, np.uint8)
    r["ds"].ConvertToNumpyArray(r["maccs"](m), mac)
    multi = np.concatenate([r["gen"].GetFingerprintAsNumPy(m).astype(np.uint8), mac,
                            r["ap"].GetFingerprintAsNumPy(m).astype(np.uint8)])
    return np.packbits(r["g3"].GetFingerprintAsNumPy(m).astype(np.uint8)), np.packbits(multi)


# M6 P0-2 (k26_fp2 / k28_fprich): multi-block fingerprint targets. Block 0 is the pool's Morgan r=2 2048 (P.fp), so the
# first 2048 bits of P.fp2 equal P.fp; the other blocks come from the pool's canonical SMILES.
RICH_FP_BLOCKS = [("morgan2", 2048), ("fcfp4", 2048), ("atompair", 2048), ("torsion", 2048), ("rdkit5", 2048),
                  ("maccs", 167)]
RICH_FP_BITS = sum(n for _, n in RICH_FP_BLOCKS)


def rich_fp_tail(smiles):
    """Unpacked bits (uint8) of RICH_FP_BLOCKS[1:] for one SMILES; zeros when RDKit cannot parse."""
    r = _rdkit()
    if "rich" not in r:
        from rdkit import DataStructs
        from rdkit.Chem import MACCSkeys, rdFingerprintGenerator as G
        r["rich"] = [G.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen()),
                     G.GetAtomPairGenerator(fpSize=2048), G.GetTopologicalTorsionGenerator(fpSize=2048),
                     G.GetRDKitFPGenerator(maxPath=5, fpSize=2048)]
        r["maccs"], r["ds"] = MACCSkeys.GenMACCSKeys, DataStructs
    n_tail = RICH_FP_BITS - RICH_FP_BLOCKS[0][1]
    m = r["Chem"].MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return np.zeros(n_tail, np.uint8)
    mac = np.zeros(167, np.uint8)
    r["ds"].ConvertToNumpyArray(r["maccs"](m), mac)
    return np.concatenate([g.GetFingerprintAsNumPy(m).astype(np.uint8) for g in r["rich"]] + [mac])
