from src.engine import predict, optimize
from src.resolver import resolve_ligand

sample = {
    "Legante": "terephthalic acid",
    "Famiglia_Legante": "Carboxylate",
    "Metallo": "Zn",
    "Sale_Metallico": "Zn(NO3)2·6H2O",
    "Counterion_Class": "nitrate",
    "Hydration_Number": 6,
    "Oxidation_State": 2,
    "Solvente": "DMF",
    "Additivo_Colinker": "Nessuno",
    "Temperatura_C": 120,
    "Tempo_ore": 24,
    "mmol_Legante": 0.1,
    "mmol_Sale": 0.1,
    "Rapporto_LM": 1.0,
    "Volume solvente": 10,
}

resolution = resolve_ligand("O=C(O)c1ccc(C(=O)O)cc1")
assert resolution["success"]
_, probabilities, predicted = predict(sample)
assert abs(float(sum(probabilities)) - 1.0) < 1e-6
assert predicted in (0, 1, 2)
assert len(optimize(sample, 3)) == 3
print("v9 smoke test passed")
