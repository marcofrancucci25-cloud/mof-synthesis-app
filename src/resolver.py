from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import quote

import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

USER_AGENT = "MOF-Synthesis-Assistant/9.0 (chemical resolver; academic research)"
DEFAULT_TIMEOUT = 5

# Small curated alias layer: it accelerates common MOF linkers but is not a ligand database.
ALIASES: dict[str, str] = {
    "bdc": "terephthalic acid",
    "h2bdc": "terephthalic acid",
    "1,4-bdc": "terephthalic acid",
    "btc": "benzene-1,3,5-tricarboxylic acid",
    "h3btc": "benzene-1,3,5-tricarboxylic acid",
    "trimesic acid": "benzene-1,3,5-tricarboxylic acid",
    "bpdc": "biphenyl-4,4'-dicarboxylic acid",
    "dobdc": "2,5-dihydroxyterephthalic acid",
    "h4dobdc": "2,5-dihydroxyterephthalic acid",
    "bpz": "4,4'-bipyrazole",
    "h2bpz": "4,4'-bipyrazole",
    "hmim": "2-methylimidazole",
    "mim": "2-methylimidazole",
    "bpy": "2,2'-bipyridine",
    "4,4-bpy": "4,4'-bipyridine",
}

CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*){2,}(?:[+-]\d*)?$")


@dataclass
class ResolutionResult:
    success: bool
    query: str
    normalized_query: str
    input_type: str
    source: str | None = None
    title: str | None = None
    iupac_name: str | None = None
    molecular_formula: str | None = None
    smiles: str | None = None
    connectivity_smiles: str | None = None
    inchikey: str | None = None
    molecular_weight: float | None = None
    message: str | None = None
    ambiguity_warning: str | None = None
    descriptors: dict[str, float | int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_query(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    replacements = {
        "’": "'", "‘": "'", "′": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "−": "-", "·": ".",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = " ".join(text.strip().split())
    return text


def _alias(query: str) -> str:
    lowered = query.casefold()
    if lowered in ALIASES:
        return ALIASES[lowered]
    match = re.search(r"\(([^)]+)\)", query)
    if match:
        abbreviation = match.group(1).strip().casefold()
        stripped = re.sub(r"^h\d+", "", abbreviation)
        if abbreviation in ALIASES:
            return ALIASES[abbreviation]
        if stripped in ALIASES:
            return ALIASES[stripped]
    return query


def detect_input_type(query: str) -> str:
    if CAS_RE.fullmatch(query):
        return "CAS"
    mol = Chem.MolFromSmiles(query)
    if mol is not None and any(token in query for token in "[]=#()@+-\\/"):
        return "SMILES"
    compact = query.replace(" ", "")
    if FORMULA_RE.fullmatch(compact):
        return "molecular formula"
    return "chemical name or abbreviation"


def _descriptors(smiles: str) -> dict[str, float | int] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolecularWeight_RDKit": round(float(Descriptors.MolWt(mol)), 4),
        "ExactMass": round(float(Descriptors.ExactMolWt(mol)), 4),
        "TPSA": round(float(rdMolDescriptors.CalcTPSA(mol)), 4),
        "LogP": round(float(Crippen.MolLogP(mol)), 4),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "RotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
        "AromaticRings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "HeavyAtoms": int(mol.GetNumHeavyAtoms()),
        "FormalCharge": int(Chem.GetFormalCharge(mol)),
    }


def _canonicalize_smiles(smiles: str) -> tuple[str, str] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    full = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return full, connectivity


def _request_json(url: str, timeout: int) -> dict[str, Any] | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(2):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                time.sleep(0.8)
                continue
            return None
        except requests.RequestException:
            if attempt == 0:
                time.sleep(0.5)
                continue
            return None
    return None


def _cactus_smiles(query: str, timeout: int) -> str | None:
    url = f"https://cactus.nci.nih.gov/chemical/structure/{quote(query, safe='')}/smiles"
    headers = {"User-Agent": USER_AGENT, "Accept": "text/plain"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return None
        candidate = response.text.strip()
        if not candidate or "page not found" in candidate.casefold():
            return None
        canonical = _canonicalize_smiles(candidate)
        return canonical[0] if canonical else None
    except requests.RequestException:
        return None


def _pubchem_properties(namespace: str, identifier: str, timeout: int) -> dict[str, Any] | None:
    # PubChem renamed the historical CanonicalSMILES/IsomericSMILES properties.
    properties = "Title,IUPACName,MolecularFormula,SMILES,ConnectivitySMILES,InChIKey,MolecularWeight"
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
        f"{namespace}/{quote(identifier, safe='')}/property/{properties}/JSON"
    )
    payload = _request_json(url, timeout)
    if not payload:
        return None
    rows = payload.get("PropertyTable", {}).get("Properties", [])
    return rows[0] if rows else None


def _pubchem_formula_candidate(formula: str, timeout: int) -> tuple[dict[str, Any] | None, str | None]:
    # Formula searches are inherently ambiguous. Retrieve a short CID list and use the
    # first PubChem result only as an editable candidate, while surfacing the ambiguity.
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/formula/"
        f"{quote(formula, safe='')}/cids/JSON?MaxRecords=10"
    )
    payload = _request_json(url, timeout)
    cids = (payload or {}).get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None, None
    row = _pubchem_properties("cid", str(cids[0]), timeout)
    warning = (
        f"The molecular formula matched {len(cids)} PubChem candidate(s) in the first result page. "
        "The proposed structure must be checked manually or replaced with a name, CAS number or SMILES."
    )
    return row, warning


def _result_from_pubchem(query: str, normalized: str, input_type: str, row: dict[str, Any], source: str,
                         warning: str | None = None) -> ResolutionResult:
    smiles = row.get("SMILES") or row.get("ConnectivitySMILES")
    canonical = _canonicalize_smiles(smiles) if smiles else None
    full = canonical[0] if canonical else smiles
    connectivity = canonical[1] if canonical else row.get("ConnectivitySMILES")
    return ResolutionResult(
        success=bool(full), query=query, normalized_query=normalized, input_type=input_type,
        source=source, title=row.get("Title"), iupac_name=row.get("IUPACName"),
        molecular_formula=row.get("MolecularFormula"), smiles=full,
        connectivity_smiles=connectivity, inchikey=row.get("InChIKey"),
        molecular_weight=float(row["MolecularWeight"]) if row.get("MolecularWeight") is not None else None,
        ambiguity_warning=warning, descriptors=_descriptors(full) if full else None,
        message="Structure resolved and validated with RDKit." if full else "The service returned no valid SMILES."
    )


@lru_cache(maxsize=512)
def resolve_ligand(query: str, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    original = str(query or "")
    normalized = normalize_query(original)
    if not normalized:
        return ResolutionResult(False, original, normalized, "empty", message="Enter a ligand identifier.").to_dict()

    aliased = _alias(normalized)
    input_type = detect_input_type(aliased)

    if input_type == "SMILES":
        canonical = _canonicalize_smiles(aliased)
        if canonical:
            full, connectivity = canonical
            return ResolutionResult(
                True, original, normalized, input_type, source="direct SMILES / RDKit",
                title=normalized, smiles=full, connectivity_smiles=connectivity,
                descriptors=_descriptors(full), message="SMILES parsed and canonicalized locally."
            ).to_dict()

    # PubChem is queried first for names/CAS because it returns identity metadata in one call.
    if input_type == "molecular formula":
        row, warning = _pubchem_formula_candidate(aliased.replace(" ", ""), timeout)
        if row:
            return _result_from_pubchem(original, normalized, input_type, row, "PubChem formula search", warning).to_dict()
    else:
        row = _pubchem_properties("name", aliased, timeout)
        if row:
            return _result_from_pubchem(original, normalized, input_type, row, "PubChem PUG REST").to_dict()

    # NCI Cactus is an independent fallback and often recognizes alternative names.
    cactus = _cactus_smiles(aliased, timeout)
    if cactus:
        row = _pubchem_properties("smiles", cactus, timeout)
        if row:
            result = _result_from_pubchem(original, normalized, input_type, row, "NCI Cactus + PubChem")
            result.message = "Resolved by NCI Cactus, enriched through PubChem and validated with RDKit."
            return result.to_dict()
        full, connectivity = _canonicalize_smiles(cactus) or (cactus, cactus)
        return ResolutionResult(
            True, original, normalized, input_type, source="NCI Cactus",
            title=aliased, smiles=full, connectivity_smiles=connectivity,
            descriptors=_descriptors(full), message="Resolved by NCI Cactus and validated with RDKit."
        ).to_dict()

    return ResolutionResult(
        False, original, normalized, input_type, source=None,
        message=("No valid structure was found. Enter a SMILES, CAS number, a less abbreviated chemical name, "
                 "or continue with the text input and select the ligand family manually.")
    ).to_dict()
