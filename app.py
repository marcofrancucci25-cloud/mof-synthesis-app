import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from rdkit import Chem
from rdkit.Chem import Descriptors

st.set_page_config(page_title="MOF Synthesis Predictor", page_icon="🧪", layout="wide")
st.title("🧪 Predictor & Optimizer per Sintesi di MOF")
st.markdown("Strumento di supporto alle decisioni di laboratorio basato su Machine Learning (Gradient Boosting).")

# Dizionario proprietà metalli
metal_props = {
    'Co': {'Z': 27, 'Electronegativity': 1.88, 'Radius_pm': 126, 'Group': 9, 'Period': 4},
    'Cu': {'Z': 29, 'Electronegativity': 1.90, 'Radius_pm': 132, 'Group': 11, 'Period': 4},
    'Cd': {'Z': 48, 'Electronegativity': 1.69, 'Radius_pm': 144, 'Group': 12, 'Period': 5},
    'Ag': {'Z': 47, 'Electronegativity': 1.93, 'Radius_pm': 145, 'Group': 11, 'Period': 5},
    'Zr': {'Z': 40, 'Electronegativity': 1.33, 'Radius_pm': 160, 'Group': 4, 'Period': 5},
    'Ni': {'Z': 28, 'Electronegativity': 1.91, 'Radius_pm': 124, 'Group': 10, 'Period': 4},
    'Ru': {'Z': 44, 'Electronegativity': 2.20, 'Radius_pm': 134, 'Group': 8, 'Period': 5},
    'Zn': {'Z': 30, 'Electronegativity': 1.65, 'Radius_pm': 122, 'Group': 12, 'Period': 4},
    'Fe': {'Z': 26, 'Electronegativity': 1.83, 'Radius_pm': 126, 'Group': 8, 'Period': 4},
    'Mn': {'Z': 25, 'Electronegativity': 1.55, 'Radius_pm': 139, 'Group': 7, 'Period': 4},
    'Rh': {'Z': 45, 'Electronegativity': 2.28, 'Radius_pm': 135, 'Group': 9, 'Period': 5},
    'Au': {'Z': 79, 'Electronegativity': 2.54, 'Radius_pm': 136, 'Group': 11, 'Period': 6},
    'Ir': {'Z': 77, 'Electronegativity': 2.20, 'Radius_pm': 136, 'Group': 9, 'Period': 6},
    'Al': {'Z': 13, 'Electronegativity': 1.61, 'Radius_pm': 121, 'Group': 13, 'Period': 3},
    'Ti': {'Z': 22, 'Electronegativity': 1.54, 'Radius_pm': 147, 'Group': 4, 'Period': 4},
    'Mg': {'Z': 12, 'Electronegativity': 1.31, 'Radius_pm': 141, 'Group': 2, 'Period': 3}
}

# Funzione per encodare e preparare i dati al volo
def process_unified_dataset(df):
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
        target = int(row.get('Target_Esito_Classe', 0)) if pd.notnull(row.get('Target_Esito_Classe')) else 0
        
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
        st.info("⚡ Generazione e analisi automatica del dataset in corso...")
        raw_df = pd.read_csv(csv_file)
        df = process_unified_dataset(raw_df)
        
        # Gestione pulizia dati mancanti (NaN)
        X = df.drop(columns=['Target_Esito_Classe'])
        X = X.fillna(X.mean()).fillna(0)
        y = df['Target_Esito_Classe'].fillna(0).astype(int)
        
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)
        model.fit(X, y)
        joblib.dump(model, pkl_file)
        return model
    else:
        st.error(f"File '{csv_file}' non trovato! Assicurati di aver caricato il file su GitHub.")
        st.stop()

try:
    model = load_or_train_model()
    st.sidebar.success("Modello ML attivo e pronto!")
except Exception as e:
    st.sidebar.error(f"Errore: {e}")
    st.stop()

tab1, tab2 = st.tabs(["🔮 Predizione Singola Sintesi", "📂 Predizione da File Excel"])

with tab1:
    st.subheader("Inserisci i parametri della reazione")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1. Legante Chimico")
        smiles_input = st.text_input("SMILES del Legante:", value="c1cc(C(=O)O)cc(C(=O)O)c1")
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            st.success(f"Molecola Riconosciuta! MW: {mw:.2f} g/mol, LogP: {logp:.2f}")
        else:
            st.error("SMILES non valido.")
            mw, logp, hbd, hba, tpsa, rot_bonds = 0, 0, 0, 0, 0, 0

    with col2:
        st.markdown("### 2. Precursore Metallico")
        metallo_sel = st.selectbox("Metallo:", list(metal_props.keys()), index=1)
        anione_sel = st.selectbox("Anione / Precursore:", ['Nitrato', 'Acetato', 'Cloruro', 'Altro'])
        
    with col3:
        st.markdown("### 3. Condizioni di Reazione")
        solvente_sel = st.selectbox("Solvente:", ['DMF', 'DMF/H2O', 'MeOH', 'EtOH', 'CH2Cl2', 'MeCN', 'Altro'])
        temp = st.number_input("Temperatura (°C):", min_value=20.0, max_value=250.0, value=120.0, step=5.0)
        tempo = st.number_input("Tempo di Reazione (Ore):", min_value=1.0, max_value=168.0, value=48.0, step=6.0)
        mmol_legante = st.number_input("mmol Legante:", min_value=0.01, max_value=10.0, value=0.10, step=0.01)
        mmol_sale = st.number_input("mmol Sale Metallico:", min_value=0.01, max_value=10.0, value=0.10, step=0.01)

    if st.button("🚀 Calcola Probabilità di Successo", type="primary"):
        if not mol:
            st.error("Inserisci uno SMILES valido prima di continuare.")
        else:
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

            df_features = pd.DataFrame([input_dict])
            for col in model.feature_names_in_:
                if col not in df_features.columns:
                    df_features[col] = 0

            df_features = df_features[model.feature_names_in_]
            probs = model.predict_proba(df_features)[0]
            pred_class = model.predict(df_features)[0]

            st.markdown("---")
            st.subheader("📊 Risultato della Predizione")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("🔴 Probabilità Insuccesso (0)", f"{probs[0]*100:.1f}%")
            res_col2.metric("🟡 Probabilità Parziale (1)", f"{probs[1]*100:.1f}%")
            res_col3.metric("🟢 Probabilità Cristalli/Successo (2)", f"{probs[2]*100:.1f}%")

            if pred_class == 2:
                st.balloons()
                st.success("✨ **Sintesi Promettente!** Alta probabilità di formazione di monocristalli o fase pulita.")
            elif pred_class == 1:
                st.warning("⚠️ **Risultato Parziale Atteso.** Possibile prodotto amorfo o miscela.")
            else:
                st.error("❌ **Insuccesso Probabile.** Si consiglia di rivedere le condizioni di reazione.")

with tab2:
    st.subheader("Carica un file Excel con più sintesi da testare")
    uploaded_file = st.file_uploader("Carica File .xlsx o .csv", type=['xlsx', 'csv'])
