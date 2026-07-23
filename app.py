import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import matplotlib.pyplot as plt

# Try optional shap import gracefully
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from sklearn.ensemble import GradientBoostingClassifier
from rdkit import Chem
from rdkit.Chem import Descriptors

st.set_page_config(page_title="MOF Synthesis Predictor & Optimizer", page_icon="🧪", layout="wide")
st.title("🧪 Predictor & Optimizer per Sintesi di MOF")
st.markdown("Strumento avanzato di Machine Learning per la predizione, ottimizzazione e **spiegabilità chimica** della sintesi di MOF.")

# --- FUNZIONE HELPER: DA NOME/FORMULA A SMILES (PubChem API) ---
def get_smiles_from_pubchem(query):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/IsomericSMILES/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['PropertyTable']['Properties'][0]['IsomericSMILES']
    except Exception:
        pass
    return None

# --- PROPRIETÀ METALLI ---
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

def process_unified_dataset(df):
    target_col = None
    possible_targets = ['Target_Esito_Classe', 'Target', 'Esito', 'Classe', 'Target_Classe', 'Esito_Classe']
    for col in df.columns:
        if col in possible_targets or 'Target' in col or 'Esito' in col:
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
        st.info("⚡ Generazione e analisi automatica del dataset in corso...")
        raw_df = pd.read_csv(csv_file)
        df = process_unified_dataset(raw_df)
        
        X = df.drop(columns=['Target_Esito_Classe'])
        X = X.fillna(X.mean()).fillna(0)
        y = df['Target_Esito_Classe'].astype(int)
        
        if len(y.unique()) < 2:
            X_extra = X.copy().iloc[:3]
            y_extra = pd.Series([0, 1, 2][:len(X_extra)])
            X = pd.concat([X, X_extra], ignore_index=True)
            y = pd.concat([y, y_extra], ignore_index=True)

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)
        model.fit(X, y)
        joblib.dump(model, pkl_file)
        return model
    else:
        st.error(f"File '{csv_file}' non trovato!")
        st.stop()

try:
    model = load_or_train_model()
    st.sidebar.success("Modello ML attivo e pronto!")
except Exception as e:
    st.sidebar.error(f"Errore: {e}")
    st.stop()

# --- SIDEBAR: FEATURE IMPORTANCE GLOBAL ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Importanza Globale Parametri")
if hasattr(model, 'feature_importances_'):
    importances = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values(ascending=True).tail(8)
    st.sidebar.bar_chart(importances)

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
        
        mode_legante = st.radio(
            "Modalità Input Legante:", 
            ["SMILES", "Nome Chimico / Formula", "Carica File (.mol / .sdf)"],
            horizontal=True
        )
        
        mol = None
        
        if mode_legante == "SMILES":
            smiles_input = st.text_input("SMILES del Legante:", value="c1cc(C(=O)O)cc(C(=O)O)c1")
            if smiles_input:
                mol = Chem.MolFromSmiles(smiles_input)
                
        elif mode_legante == "Nome Chimico / Formula":
            query_input = st.text_input("Nome o Formula (es. 'Terephthalic acid' o 'C8H6O4'):", value="Terephthalic acid")
            if query_input:
                with st.spinner("Ricerca struttura su PubChem..."):
                    found_smiles = get_smiles_from_pubchem(query_input)
                    if found_smiles:
                        mol = Chem.MolFromSmiles(found_smiles)
                        st.caption(f"SMILES Ricavato: `{found_smiles}`")
                    else:
                        st.error("Nessuna molecola trovata per questa ricerca. Prova con il nome in inglese.")
                        
        elif mode_legante == "Carica File (.mol / .sdf)":
            uploaded_mol_file = st.file_uploader("Carica file .mol o .sdf", type=['mol', 'sdf'])
            if uploaded_mol_file is not None:
                file_bytes = uploaded_mol_file.getvalue().decode('utf-8')
                mol = Chem.MolFromMolBlock(file_bytes)
                if not mol:
                    st.error("Impossibile interpretare il file strutturale.")

        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            st.success(f"Molecola Valida! MW: {mw:.2f} g/mol, LogP: {logp:.2f}")
        else:
            if mode_legante == "SMILES":
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
            st.error("Inserisci una molecola valida prima di continuare.")
        else:
            df_features = build_feature_row(mw, logp, hbd, hba, tpsa, rot_bonds, temp, tempo, mmol_legante, mmol_sale, metallo_sel, anione_sel, solvente_sel)
            probs = model.predict_proba(df_features)[0]
            pred_class = model.predict(df_features)[0]

            st.markdown("---")
            st.subheader("📊 Risultato della Predizione")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            p0 = probs[0] * 100 if len(probs) > 0 else 0
            p1 = probs[1] * 100 if len(probs) > 1 else 0
            p2 = probs[2] * 100 if len(probs) > 2 else 0

            res_col1.metric("🔴 Probabilità Insuccesso (0)", f"{p0:.1f}%")
            res_col2.metric("🟡 Probabilità Parziale (1)", f"{p1:.1f}%")
            res_col3.metric("🟢 Probabilità Cristalli/Successo (2)", f"{p2:.1f}%")

            if pred_class == 2:
                st.balloons()
                st.success("✨ **Sintesi Promettente!** Alta probabilità di formazione di monocristalli o fase pulita.")
            elif pred_class == 1:
                st.warning("⚠️ **Risultato Parziale Atteso.** Possibile prodotto amorfo o miscela.")
            else:
                st.error("❌ **Insuccesso Probabile.** Si consiglia di rivedere le condizioni di reazione.")

            # --- SPIEGABILITÀ CHIMICA ---
            st.markdown("---")
            st.subheader("🧬 Spiegabilità Chimica della Predizione")
            
            rendered = False
            if HAS_SHAP:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(df_features)
                    target_idx = min(2, len(shap_values) - 1) if isinstance(shap_values, list) else 0
                    
                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    s_vals = shap_values[target_idx][0] if isinstance(shap_values, list) else shap_values[0]
                    shap_series = pd.Series(s_vals, index=df_features.columns).sort_values(key=abs, ascending=True).tail(8)
                    
                    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in shap_series.values]
                    ax.barh(shap_series.index, shap_series.values, color=colors)
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                    ax.set_xlabel("Impatto SHAP sulla Probabilità di Successo")
                    ax.set_title("Analisi SHAP (Verde = Positivo, Rosso = Negativo)")
                    st.pyplot(fig)
                    rendered = True
                except Exception:
                    rendered = False

            if not rendered:
                fig, ax = plt.subplots(figsize=(8, 3.5))
                feats_contrib = (df_features.iloc[0] * model.feature_importances_).sort_values(ascending=True).tail(8)
                ax.barh(feats_contrib.index, feats_contrib.values, color='#3498db')
                ax.set_xlabel("Punto di Impatto Relativo dei Parametri")
                ax.set_title("Contributo dei Parametri Inseriti al Modello")
                st.pyplot(fig)

# --- TAB 2: PREDIZIONE BATCH ---
with tab2:
    st.subheader("Carica un file Excel o CSV con più sintesi da valutare")
    uploaded_file = st.file_uploader("Scegli un file (.xlsx o .csv)", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                input_batch = pd.read_csv(uploaded_file)
            else:
                input_batch = pd.read_excel(uploaded_file)
                
            st.write("📋 **Anteprima dei dati caricati:**", input_batch.head())
            
            if st.button("⚡ Elabora tutte le Sintesi"):
                processed_batch = process_unified_dataset(input_batch)
                X_batch = processed_batch.drop(columns=['Target_Esito_Classe'])
                
                for col in model.feature_names_in_:
                    if col not in X_batch.columns:
                        X_batch[col] = 0
                X_batch = X_batch[model.feature_names_in_]
                
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)
                
                results_df = input_batch.copy()
                results_df['Predizione_Classe'] = preds
                results_df['Prob_Insuccesso_%'] = (probs[:, 0] * 100).round(1) if probs.shape[1] > 0 else 0
                results_df['Prob_Parziale_%'] = (probs[:, 1] * 100).round(1) if probs.shape[1] > 1 else 0
                results_df['Prob_Successo_%'] = (probs[:, 2] * 100).round(1) if probs.shape[1] > 2 else 0
                
                st.success("✅ Predizioni completate con successo!")
                st.dataframe(results_df)
                
                csv_download = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Scarica Risultati in CSV",
                    data=csv_download,
                    file_name="Risultati_Predizione_MOF.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Errore durante l'elaborazione del file: {e}")

# --- TAB 3: OTTIMIZZATORE AUTOMATICO (INVERSE DESIGN) ---
with tab3:
    st.subheader("⚡ Ottimizzatore di Condizioni Sperimentali")
    st.markdown("Inserisci i reagenti di partenza e l'IA cercherà la **combinazione ottimale di temperatura, tempo e solvente** per massimizzare la formazione dei cristalli.")
    
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        opt_smiles = st.text_input("SMILES Legante:", value="c1cc(C(=O)O)cc(C(=O)O)c1", key="opt_smiles")
        opt_mol = Chem.MolFromSmiles(opt_smiles)
    with opt_col2:
        opt_metallo = st.selectbox("Metallo Desiderato:", list(metal_props.keys()), index=1, key="opt_met")
        opt_anione = st.selectbox("Anione:", ['Nitrato', 'Acetato', 'Cloruro', 'Altro'], key="opt_an")

    if st.button("🔍 Trova Ricetta Ottimale"):
        if not opt_mol:
            st.error("SMILES non valido.")
        else:
            opt_mw = Descriptors.MolWt(opt_mol)
            opt_logp = Descriptors.MolLogP(opt_mol)
            opt_hbd = Descriptors.NumHDonors(opt_mol)
            opt_hba = Descriptors.NumHAcceptors(opt_mol)
            opt_tpsa = Descriptors.TPSA(opt_mol)
            opt_rot = Descriptors.NumRotatableBonds(opt_mol)
            
            temperatures = [80, 100, 120, 140, 160]
            times = [12, 24, 48, 72]
            solvents = ['DMF', 'DMF/H2O', 'MeOH', 'EtOH', 'CH2Cl2', 'MeCN']
            ratios = [(0.1, 0.1), (0.2, 0.1), (0.1, 0.2)]
            
            candidates = []
            
            with st.spinner("Generazione e simulazione dello spazio di reazione..."):
                for t in temperatures:
                    for tm in times:
                        for s in solvents:
                            for m_leg, m_sale in ratios:
                                feat = build_feature_row(opt_mw, opt_logp, opt_hbd, opt_hba, opt_tpsa, opt_rot, t, tm, m_leg, m_sale, opt_metallo, opt_anione, s)
                                prob_succ = model.predict_proba(feat)[0]
                                p_success = prob_succ[2] if len(prob_succ) > 2 else 0.0
                                
                                candidates.append({
                                    'Temperatura (°C)': t,
                                    'Tempo (Ore)': tm,
                                    'Solvente': s,
                                    'mmol Legante': m_leg,
                                    'mmol Sale': m_sale,
                                    'Rapporto L/M': m_leg / m_sale,
                                    'Probabilità Successo (%)': round(p_success * 100, 2)
                                })
            
            df_cand = pd.DataFrame(candidates).sort_values(by='Probabilità Successo (%)', ascending=False)
            
            st.markdown("---")
            st.subheader("🥇 Migliori Condizioni Consigliate dall'IA")
            
            best = df_cand.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Temperatura", f"{best['Temperatura (°C)']} °C")
            m2.metric("Tempo", f"{best['Tempo (Ore)']} h")
            m3.metric("Solvente", f"{best['Solvente']}")
            m4.metric("Probabilità Max", f"{best['Probabilità Successo (%)']}%")
            
            st.markdown("### 📋 Classifica delle prime 5 migliori ricette:")
            st.dataframe(df_cand.head(5))
