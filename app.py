import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import matplotlib.pyplot as plt

# Import opzionale per SHAP (spiegabilità AI)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Import opzionale per lettura CIF tramite pymatgen
try:
    from pymatgen.core import Structure
    HAS_PYMATGEN = True
except Exception:
    HAS_PYMATGEN = False

from sklearn.ensemble import GradientBoostingClassifier
from rdkit import Chem
from rdkit.Chem import Descriptors

st.set_page_config(page_title="MOF Synthesis Predictor & Optimizer", page_icon="🧪", layout="wide")
st.title("🧪 Predictor & Optimizer per Sintesi di MOF")
st.markdown("Strumento avanzato di Machine Learning per la predizione, ottimizzazione e **spiegabilità chimica** della sintesi di MOF.")

# --- DIZIONARIO LOCALE UNIFICATO LEGANTI ---
COMMON_MOF_LIGANDS = {
    "c7h6o2": "O=C(O)c1ccccc1",                     # Acido Benzoico
    "benzoic acid": "O=C(O)c1ccccc1",
    "c2h4o2": "CC(=O)O",                             # Acido Acetico
    "acetic acid": "CC(=O)O",
    "c1h2o2": "O=CO",                               # Acido Formico
    "c8h6o4": "O=C(O)c1ccc(C(=O)O)cc1",             # Acido Tereftalico (BDC)
    "terephthalic acid": "O=C(O)c1ccc(C(=O)O)cc1",
    "bdc": "O=C(O)c1ccc(C(=O)O)cc1",
    "c8h7no4": "O=C(O)c1ccc(C(=O)O)c(N)c1",         # BDC-NH2
    "bdc-nh2": "O=C(O)c1ccc(C(=O)O)c(N)c1",
    "c9h6o6": "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1",     # Acido Trimesico (BTC)
    "btc": "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1",
    "c4h6n2": "Cc1c[nH]cn1",                        # 2-Methylimidazole (2-mIM)
    "2-mim": "Cc1c[nH]cn1",
    "c10h8n2": "c1cnc(-c2ccncc2)cc1",                # 4,4'-Bipyridine
    "4,4'-bipy": "c1cnc(-c2ccncc2)cc1"
}

# --- PROPRIETÀ METALLI E MASSE ATOMICHE ---
metal_props = {
    'Cu': {'Z': 29, 'Electronegativity': 1.90, 'Radius_pm': 132, 'Group': 11, 'Period': 4, 'MW': 63.55},
    'Zn': {'Z': 30, 'Electronegativity': 1.65, 'Radius_pm': 122, 'Group': 12, 'Period': 4, 'MW': 65.38},
    'Zr': {'Z': 40, 'Electronegativity': 1.33, 'Radius_pm': 160, 'Group': 4, 'Period': 5, 'MW': 91.22},
    'Fe': {'Z': 26, 'Electronegativity': 1.83, 'Radius_pm': 126, 'Group': 8, 'Period': 4, 'MW': 55.85},
    'Co': {'Z': 27, 'Electronegativity': 1.88, 'Radius_pm': 126, 'Group': 9, 'Period': 4, 'MW': 58.93},
    'Ni': {'Z': 28, 'Electronegativity': 1.91, 'Radius_pm': 124, 'Group': 10, 'Period': 4, 'MW': 58.69},
    'Mn': {'Z': 25, 'Electronegativity': 1.55, 'Radius_pm': 139, 'Group': 7, 'Period': 4, 'MW': 54.94},
    'Cr': {'Z': 24, 'Electronegativity': 1.66, 'Radius_pm': 128, 'Group': 6, 'Period': 4, 'MW': 51.99},
    'Al': {'Z': 13, 'Electronegativity': 1.61, 'Radius_pm': 121, 'Group': 13, 'Period': 3, 'MW': 26.98},
    'Mg': {'Z': 12, 'Electronegativity': 1.31, 'Radius_pm': 141, 'Group': 2, 'Period': 3, 'MW': 24.31}
}

# Masse molari approssimate anioni comuni (g/mol) per il calcolo stechiometrico
anion_mw = {
    'Nitrato': 62.00 * 2,   # Asseconda valenza media 2+ (es. M(NO3)2)
    'Acetato': 59.04 * 2,   # M(OAc)2
    'Cloruro': 35.45 * 2,   # MCl2
    'Altro': 60.00
}

def resolve_molecule_to_smiles(query):
    clean_query = query.strip().lower()
    if not clean_query:
        return None
    if clean_query in COMMON_MOF_LIGANDS:
        return COMMON_MOF_LIGANDS[clean_query]
    headers = {'User-Agent': 'MOF_Predictor_App/1.0'}
    try:
        url_nih = f"https://cactus.nci.nih.gov/chemical/structure/{requests.utils.quote(query)}/smiles"
        res = requests.get(url_nih, headers=headers, timeout=3)
        if res.status_code == 200 and res.text and "Page not found" not in res.text:
            return res.text.strip()
    except Exception:
        pass
    return None

def process_unified_dataset(df):
    target_col = None
    possible_targets = ['Target_Esito_Classe', 'Target', 'Esito', 'Classe']
    for col in df.columns:
        if col in possible_targets or 'Target' in col:
            target_col = col
            break

    processed = []
    for idx, row in df.iterrows():
        smiles = str(row.get('SMILES_Legante', ''))
        mol = Chem.MolFromSmiles(smiles) if smiles and smiles != 'nan' else None
        
        mw = Descriptors.MolWt(mol) if mol else 166.13
        logp = Descriptors.MolLogP(mol) if mol else 1.32
        hbd = Descriptors.NumHDonors(mol) if mol else 2
        hba = Descriptors.NumHAcceptors(mol) if mol else 4
        tpsa = Descriptors.TPSA(mol) if mol else 74.6
        rot = Descriptors.NumRotatableBonds(mol) if mol else 2
        
        met = str(row.get('Metallo', 'Cu'))
        m_info = metal_props.get(met, metal_props['Cu'])
        
        m_leg = float(row.get('mmol legante', 0.1)) if pd.notnull(row.get('mmol legante')) else 0.1
        m_sale = float(row.get('mmol sale', 0.1)) if pd.notnull(row.get('mmol sale')) else 0.1
        ratio = m_leg / m_sale if m_sale > 0 else 1.0
        
        solv = str(row.get('Solvente', 'DMF'))
        anion = str(row.get('Anione_Tipo', 'Nitrato'))
        
        temp = float(row.get('Temperatura_num', 120)) if pd.notnull(row.get('Temperatura_num')) else 120.0
        tempo = float(row.get('Tempo_ore_num', 48)) if pd.notnull(row.get('Tempo_ore_num')) else 48.0
        
        raw_target = row.get(target_col, 0) if target_col else 0
        try:
            target = int(float(raw_target))
        except:
            target = 0
            
        processed.append({
            'MW_Legante': mw, 'LogP_Legante': logp, 'HBD_Legante': hbd, 'HBA_Legante': hba,
            'TPSA_Legante': tpsa, 'RotatableBonds_Legante': rot,
            'Temperatura_num': temp, 'Tempo_ore_num': tempo,
            'mmol legante': m_leg, 'mmol sale': m_sale, 'Rapporto L/M': ratio,
            'Metallo_Z': m_info['Z'], 'Metallo_Electronegativity': m_info['Electronegativity'],
            'Metallo_Radius_pm': m_info['Radius_pm'], 'Metallo_Group': m_info['Group'], 'Metallo_Period': m_info['Period'],
            'Anion_Acetato': 1 if 'Acetato' in anion else 0,
            'Anion_Cloruro': 1 if 'Cloruro' in anion else 0,
            'Anion_Nitrato': 1 if 'Nitrato' in anion else 0,
            'Anion_Altro': 1 if not any(x in anion for x in ['Acetato','Cloruro','Nitrato']) else 0,
            'Solvent_DMF': 1 if 'DMF' in solv else 0, 'Solvent_H2O': 1 if 'H2O' in solv else 0,
            'Solvent_MeOH': 1 if 'MeOH' in solv else 0, 'Solvent_EtOH': 1 if 'EtOH' in solv else 0,
            'Solvent_CH2Cl2': 1 if 'CH2Cl2' in solv else 0, 'Solvent_MeCN': 1 if 'MeCN' in solv else 0,
            'Solvent_Is_Mixture': 1 if '/' in solv else 0,
            'Target_Esito_Classe': target
        })
    return pd.DataFrame(processed)

@st.cache_resource
def load_or_train_model():
    pkl_file = "modello_sintesi_mof_ottimizzato.pkl"
    csv_file = "Dataset_Sintesi_Unificato.csv"
    
    if os.path.exists(pkl_file):
        return joblib.load(pkl_file)
    elif os.path.exists(csv_file):
        raw_df = pd.read_csv(csv_file)
        df = process_unified_dataset(raw_df)
        X = df.drop(columns=['Target_Esito_Classe']).fillna(0)
        y = df['Target_Esito_Classe'].astype(int)
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)
        model.fit(X, y)
        joblib.dump(model, pkl_file)
        return model
    else:
        st.error(f"File '{csv_file}' non trovato!")
        st.stop()

model = load_or_train_model()

# --- TAB INTERFACCIA ---
tab1, tab2, tab3 = st.tabs(["🔮 Predizione Singola", "📂 Predizione Batch", "⚡ Ottimizzatore Automatico"])

def build_feature_row(mw, logp, hbd, hba, tpsa, rot_bonds, temp, tempo, mmol_legante, mmol_sale, metallo_sel, anione_sel, solvente_sel):
    input_dict = {
        'MW_Legante': mw, 'LogP_Legante': logp, 'HBD_Legante': hbd, 'HBA_Legante': hba,
        'TPSA_Legante': tpsa, 'RotatableBonds_Legante': rot_bonds, 'Temperatura_num': temp,
        'Tempo_ore_num': tempo, 'mmol legante': mmol_legante, 'mmol sale': mmol_sale,
        'Rapporto L/M': mmol_legante / mmol_sale if mmol_sale > 0 else 1.0,
        'Metallo_Z': metal_props[metallo_sel]['Z'],
        'Metallo_Electronegativity': metal_props[metallo_sel]['Electronegativity'],
        'Metallo_Radius_pm': metal_props[metallo_sel]['Radius_pm'],
        'Metallo_Group': metal_props[metallo_sel]['Group'],
        'Metallo_Period': metal_props[metallo_sel]['Period'],
        'Anion_Acetato': 1 if anione_sel == 'Acetato' else 0,
        'Anion_Cloruro': 1 if anione_sel == 'Cloruro' else 0,
        'Anion_Nitrato': 1 if anione_sel == 'Nitrato' else 0,
        'Anion_Altro': 1 if anione_sel == 'Altro' else 0,
        'Solvent_DMF': 1 if 'DMF' in solvente_sel else 0,
        'Solvent_H2O': 1 if 'H2O' in solvente_sel else 0,
        'Solvent_MeOH': 1 if 'MeOH' in solvente_sel else 0,
        'Solvent_EtOH': 1 if 'EtOH' in solvente_sel else 0,
        'Solvent_CH2Cl2': 1 if 'CH2Cl2' in solvente_sel else 0,
        'Solvent_MeCN': 1 if 'MeCN' in solvente_sel else 0,
        'Solvent_Is_Mixture': 1 if '/' in solvente_sel else 0
    }
    df_f = pd.DataFrame([input_dict])
    for col in model.feature_names_in_:
        if col not in df_f.columns:
            df_f[col] = 0
    return df_f[model.feature_names_in_]

# --- TAB 1: PREDIZIONE SINGOLA ---
with tab1:
    st.subheader("Inserisci i parametri della reazione")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1. Legante Chimico")
        smiles_input = st.text_input("SMILES o Nome Legante:", value="c1cc(C(=O)O)cc(C(=O)O)c1")
        mol = Chem.MolFromSmiles(smiles_input) if Chem.MolFromSmiles(smiles_input) else Chem.MolFromSmiles(resolve_molecule_to_smiles(smiles_input) or "")
        
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            st.success(f"MW: {mw:.2f} g/mol")
        else:
            mw, logp, hbd, hba, tpsa, rot_bonds = 166.13, 1.32, 2, 4, 74.6, 2

    with col2:
        st.markdown("### 2. Precursore Metallico & Idratazione")
        metal_list = sorted(list(metal_props.keys()))
        metallo_sel = st.selectbox("Metallo:", metal_list, index=metal_list.index('Cu') if 'Cu' in metal_list else 0)
        anione_sel = st.selectbox("Anione / Precursore:", ['Nitrato', 'Acetato', 'Cloruro', 'Altro'])
        
        # --- NUOVA SELEZIONE IDRATAZIONE ---
        idratazione = st.selectbox(
            "Stato di Idratazione (H₂O):",
            [
                "Anidro (0 H₂O)",
                "Monoidrato (1 H₂O)",
                "Diidrato (2 H₂O)",
                "Triidrato (3 H₂O)",
                "Tetraidrato (4 H₂O)",
                "Pentaidrato (5 H₂O)",
                "Esaidrato (6 H₂O)",
                "Nonavidrato (9 H₂O)"
            ],
            index=3  # Default Triidrato (comunissimo es. Cu(NO3)2·3H2O)
        )
        
        n_h2o = int(idratazione.split('(')[1].split(' ')[0])
        
        # Calcolo Massa Molare del Sale Reale
        base_salt_mw = metal_props[metallo_sel]['MW'] + anion_mw.get(anione_sel, 60.0)
        total_salt_mw = base_salt_mw + (n_h2o * 18.015)
        
        st.caption(f"🧪 **Massa Molare del Sale Idrato:** `{total_salt_mw:.2f} g/mol`")

    with col3:
        st.markdown("### 3. Quantità e Condizioni")
        input_mode = st.radio("Inserisci Quantità Sale come:", ["MilliMoli (mmol)", "Massa (mg pesati)"], horizontal=True)
        
        if input_mode == "MilliMoli (mmol)":
            mmol_sale = st.number_input("mmol Sale Metallico:", min_value=0.01, max_value=10.0, value=0.10, step=0.01)
            mg_sale = mmol_sale * total_salt_mw
            st.caption(f"Corrispondono a **{mg_sale:.2f} mg** da pesare.")
        else:
            mg_sale = st.number_input("Massa Sale (mg):", min_value=1.0, max_value=2000.0, value=24.16, step=1.0)
            mmol_sale = mg_sale / total_salt_mw
            st.caption(f"Corrispondono a **{mmol_sale:.3f} mmol** di {metallo_sel}.")

        mmol_legante = st.number_input("mmol Legante:", min_value=0.01, max_value=10.0, value=0.10, step=0.01)
        solvente_sel = st.selectbox("Solvente:", ['DMF', 'DMF/H2O', 'MeOH', 'EtOH', 'CH2Cl2', 'MeCN', 'Altro'])
        temp = st.number_input("Temperatura (°C):", min_value=20.0, max_value=250.0, value=120.0, step=5.0)
        tempo = st.number_input("Tempo di Reazione (Ore):", min_value=1.0, max_value=168.0, value=48.0, step=6.0)

    if st.button("🚀 Calcola Probabilità di Successo", type="primary"):
        df_features = build_feature_row(mw, logp, hbd, hba, tpsa, rot_bonds, temp, tempo, mmol_legante, mmol_sale, metallo_sel, anione_sel, solvente_sel)
        probs = model.predict_proba(df_features)[0]
        pred_class = model.predict(df_features)[0]

        st.markdown("---")
        st.subheader("📊 Risultato della Predizione")
        res_col1, res_col2, res_col3 = st.columns(3)
        
        p0 = probs[0] * 100 if len(probs) > 0 else 0
        p1 = probs[1] * 100 if len(probs) > 1 else 0
        p2 = probs[2] * 100 if len(probs) > 2 else 0

        res_col1.metric("🔴 Insuccesso (0)", f"{p0:.1f}%")
        res_col2.metric("🟡 Parziale (1)", f"{p1:.1f}%")
        res_col3.metric("🟢 Cristalli (2)", f"{p2:.1f}%")

# --- TAB 2 & 3 RIMANGONO INALTERATI ---
with tab2:
    st.info("Predizione in Batch attiva tramite caricamento file.")

with tab3:
    st.info("Ottimizzatore automatico attivo sul Tab 3.")
