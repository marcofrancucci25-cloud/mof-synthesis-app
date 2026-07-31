# MOF Synthesis Assistant v9.2

Interactive Streamlit application for literature-aware MOF synthesis prediction and condition prioritization.

## New in v9.2

- ligand input by name, abbreviation, CAS number, molecular formula or SMILES;
- robust resolver chain: local RDKit → MOF aliases → NCI Cactus → PubChem;
- corrected PubChem properties (`SMILES` and `ConnectivitySMILES`);
- RDKit validation and molecular descriptors;
- explicit warning for ambiguous formula-only queries;
- metal precursor formula, counterion, oxidation state and hydration fields;
- predictive, knowledge, applicability-domain and optimizer modules.

## Deploy on Streamlit Community Cloud

- Repository branch: `main`
- Main file: `app.py`
- Recommended Python: 3.12 or 3.13

The application requires outbound HTTPS access to resolve chemical names. Direct SMILES input and the predictive core remain available without the external resolver.

## Local run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Scientific boundary

The v9.2 release improves chemical identity resolution but retains the frozen v8.0 predictive core. Successful structure resolution does not imply that a ligand is inside the model applicability domain.


## v9.2 interface workflow
Prediction and optimization are now integrated in one page. After a prediction, the app displays a traffic-light assessment, a local sensitivity explanation of favorable and limiting experimental parameters, and a contextual button for generating improved conditions while keeping the selected metal–ligand identity fixed.


## Tavily literature search

Version 9.3 removes the API-key field from the user interface. The key is read only from Streamlit Secrets or the local private secrets file.

The sidebar includes a recent-literature search restricted to selected scientific domains. Configure the API key in Streamlit Cloud under **App settings → Secrets**:

```toml
TAVILY_API_KEY = "tvly-your-key"
```

The application does not store keys entered in the temporary local password field. Search results remain retrieval outputs and should be checked against the original publisher record.


## Tavily configuration in v9.4

The literature-search interface no longer displays an API-key field. The temporary deployment key is defined once in `src/literature.py` as `TAVILY_API_KEY`; replace that single value when rotating the key.
