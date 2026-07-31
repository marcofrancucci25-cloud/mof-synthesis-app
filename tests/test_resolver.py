from src.resolver import normalize_query, detect_input_type, resolve_ligand


def test_normalization():
    assert normalize_query(" 4,4′-bipyrazole ") == "4,4'-bipyrazole"


def test_direct_smiles():
    result = resolve_ligand("O=C(O)c1ccc(C(=O)O)cc1")
    assert result["success"]
    assert result["source"] == "direct SMILES / RDKit"
    assert result["descriptors"]["HBA"] >= 2


def test_input_detection():
    assert detect_input_type("100-21-0") == "CAS"
    assert detect_input_type("C8H6O4") == "molecular formula"
