# Scientific update v8.0

## Objective
Extend the model beyond closed categorical menus by introducing chemistry-aware representations for ligand identity, metal identity and metal precursor composition.

## Added representations

- Ligand free text: character n-gram TF-IDF representation of ligand name/abbreviation/formula together with ligand family.
- Metal descriptors: atomic number, atomic weight, group, period, block and Pauling electronegativity, alongside categorical metal identity.
- Precursor descriptors: full salt string, counterion class, hydration number and inferred/entered oxidation state.
- Experimental descriptors: solvent, additive, temperature, time, reagent amounts, ligand/metal ratio and solvent volume.

## Model architecture

Chemistry-aware soft ensemble:

- 70% calibrated structured Random Forest
- 30% calibrated ligand-text Logistic Regression

Both components were calibrated by grouped cross-validation using ligand identity as the grouping variable.

## External ligand-group test

See `external_metrics_v8_0.json` for the final locked results. The external test contains ligand groups excluded from model development.

## Interpretation limitation

Textual recognition is not molecular-graph recognition. An arbitrary ligand can be entered and compared by name/formula fragments, but reliable structural extrapolation requires curated SMILES or molecular structures for all training ligands. The application therefore exposes applicability-domain warnings for unseen ligands, metals and salts.
