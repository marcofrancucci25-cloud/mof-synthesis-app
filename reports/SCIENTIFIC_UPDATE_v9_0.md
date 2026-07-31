# MOF Synthesis Assistant v9.0 — chemical resolver update

Version 9.0 replaces the single-endpoint PubChem resolver with a fault-tolerant chemical identity pipeline:

1. Unicode and punctuation normalization.
2. Direct local SMILES parsing and canonicalization with RDKit.
3. A small curated alias layer for frequent MOF linker abbreviations.
4. NCI/CADD Chemical Identifier Resolver (Cactus).
5. PubChem PUG REST enrichment using the current `SMILES` and `ConnectivitySMILES` property names.
6. RDKit validation and calculation of basic molecular descriptors.

Names, common abbreviations, CAS registry numbers, formulas and SMILES are accepted. Formula-only searches are explicitly marked as ambiguous because multiple constitutional isomers can share one formula.

## Validation boundary

The v9.0 change concerns the input/resolution layer. The frozen predictive core remains the v8.0 ensemble, so the existing v8.0 external-test metrics remain the applicable model-validation results. A chemical identity successfully resolved by an API may still be outside the model applicability domain.
