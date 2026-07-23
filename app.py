import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import matplotlib.pyplot as plt

# Import per LightGBM e Calibrazione ML
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

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

from rdkit import Chem
from rdkit.Chem import Descriptors

st.set_page_config(page_title="MOF Synthesis Predictor & Optimizer", page_icon="🧪", layout="wide")
st.title("🧪 Predictor & Optimizer per Sintesi di MOF")
st.markdown("Strumento avanzato di Machine Learning per la predizione, ottimizzazione e **spiegabilità chimica** della sintesi di MOF.")

# --- DIZIONARIO LOCALE LEGANTE MOF ---
COMMON_MOF_LIGANDS = {
    # 1. ACIDI MONOCARBOSSILICI E MODULANTI
    "c7h6o2": "O=C(O)c1ccccc1",                     # Acido Benzoico
    "benzoic acid": "O=C(O)c1ccccc1",
    "c2h4o2": "CC(=O)O",                             # Acido Acetico
    "acetic acid": "CC(=O)O",
    "c1h2o2": "O=CO",                               # Acido Formico
    "formic acid": "O=CO",
    "c2hf3o2": "O=C(O)C(F)(F)F",                    # Acido Trifluoroacetico (TFA)
    "trifluoroacetic acid": "O=C(O)C(F)(F)F",
    "tfa": "O=C(O)C(F)(F)F",
    "c3h6o2": "CCC(=O)O",                            # Acido Propionico
    "propionic acid": "CCC(=O)O",
    "c5h10o2": "CC(C)(C)C(=O)O",                     # Acido Pivalico
    "pivalic acid": "CC(C)(C)C(=O)O",

    # 2. DICARBOSSILICI AROMATICI (BDC e Derivati)
    "c8h6o4": "O=C(O)c1ccc(C(=O)O)cc1",             # Acido Tereftalico (BDC)
    "terephthalic acid": "O=C(O)c1ccc(C(=O)O)cc1",
    "bdc": "O=C(O)c1ccc(C(=O)O)cc1",
    "isophthalic acid": "O=C(O)c1cccc(C(=O)O)c1",   # Acido Isoftalico
    "phthalic acid": "O=C(O)c1ccccc1C(=O)O",         # Acido Ftalico
    "c8h7no4": "O=C(O)c1ccc(C(=O)O)c(N)c1",         # BDC-NH2
    "bdc-nh2": "O=C(O)c1ccc(C(=O)O)c(N)c1",
    "c8h5no6": "O=C(O)c1ccc(C(=O)O)c([N+](=O)[O-])c1", # BDC-NO2
    "bdc-no2": "O=C(O)c1ccc(C(=O)O)c([N+](=O)[O-])c1",
    "c8h5bro4": "O=C(O)c1ccc(C(=O)O)c(Br)c1",       # BDC-Br
    "bdc-br": "O=C(O)c1ccc(C(=O)O)c(Br)c1",
    "c8h6o5": "O=C(O)c1ccc(C(=O)O)c(O)c1",          # BDC-OH
    "bdc-oh": "O=C(O)c1ccc(C(=O)O)c(O)c1",

    # 3. DICARBOSSILICI ESTESI E ALIFATICI
    "c12h10o4": "O=C(O)c1ccc(-c2ccc(C(=O)O)cc2)cc1", # BPDC
    "bpdc": "O=C(O)c1ccc(-c2ccc(C(=O)O)cc2)cc1",
    "c12h8o4": "O=C(O)c1ccc2ccc(C(=O)O)cc2c1",      # 2,6-NDC
    "ndc": "O=C(O)c1ccc2ccc(C(=O)O)cc2c1",
    "c4h4o4": "O=C(O)/C=C/C(=O)O",                 # Acido Fumarico
    "fumaric acid": "O=C(O)/C=C/C(=O)O",

    # 4. POLICARBOSSILICI
    "c9h6o6": "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1",     # Acido Trimesico (BTC)
    "btc": "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1",
    "c27h18o6": "O=C(O)c1ccc(-c2cc(-c3ccc(C(=O)O)cc3)cc(-c3ccc(C(=O)O)cc3)c2)cc1", # BTB
    "btb": "O=C(O)c1ccc(-c2cc(-c3ccc(C(=O)O)cc3)cc(-c3ccc(C(=O)O)cc3)c2)cc1",

    # 5. IMIDAZOLI E LINKER PIRIDINICI
    "c3h4n2": "c1c[nH]cn1",                        # Imidazolo
    "c4h6n2": "Cc1c[nH]cn1",                        # 2-mIM
    "2-mim": "Cc1c[nH]cn1",
    "c10h8n2": "c1cnc(-c2ccncc2)cc1",                # 4,4'-Bipy
    "4,4'-bipy": "c1cnc(-c2ccncc2)cc1"
}

# --- PROPRIETÀ ADDITIVI E MODULATORI ---
ADDITIVES_DATABASE = {
    'Nessuno': {'type': 'None', 'MW': 0.0, 'pKa': 0.0},
    'Acido Acetico (AcOH)': {'type': 'Acid', 'MW': 60.05, 'pKa': 4.76},
    'Acido Formico (HCOOH)': {'type': 'Acid', 'MW': 46.03, 'pKa': 3.75},
    'Acido Benzoico': {'type': 'Acid', 'MW': 122.12, 'pKa': 4.20},
    'Acido Trifluoroacetico (TFA)': {'type': 'Acid', 'MW': 114.02, 'pKa': 0.23},
    'Acido Cloridrico (HCl)': {'type': 'Acid', 'MW': 36.46, 'pKa': -6.0},
    'Trietilammina (TEA)': {'type': 'Base', 'MW': 101.19, 'pKa': 10.75},
    'Diisopropiletilammina (DIPEA)': {'type': 'Base', 'MW': 129.24, 'pKa': 11.0},
    'N-Metilmorfolina': {'type': 'Base', 'MW': 101.15, 'pKa': 7.38},
    'Piridinetilammina / Piridina': {'type': 'Base', 'MW': 79.10, 'pKa': 5.25},
    'Acqua (H2O Modulatore)': {'type': 'Neutral', 'MW': 18.015, 'pKa': 14.0},
    'HF (Acido Fluoridrico)': {'type': 'Acid', 'MW': 20.01, 'pKa': 3.17}
}

# --- PROPRIETÀ METALLI COMPLETI ---
metal_props = {
    'Cu': {'Z': 29, 'Electronegativity': 1.90, 'Radius_pm': 132, 'Group': 11, 'Period': 4, 'MW': 63.55},
    'Zn': {'Z': 30, 'Electronegativity': 1.65, 'Radius_pm': 122, 'Group': 12, 'Period': 4, 'MW': 65.38},
    'Zr': {'Z': 40, 'Electronegativity': 1.33, 'Radius_pm': 160, 'Group': 4, 'Period': 5, 'MW': 91.22},
    'Fe': {'Z': 26, 'Electronegativity': 1.83, 'Radius_pm': 126, 'Group': 8, 'Period': 4, 'MW': 55.85},
    'Co': {'Z': 27, 'Electronegativity': 1.88, 'Radius_pm': 126, 'Group': 9, 'Period': 4, 'MW': 58.93},
    'Ni': {'Z': 28, 'Electronegativity': 1.91, 'Radius_pm': 124, 'Group': 10, 'Period': 4, 'MW': 58.69},
    'Mn': {'Z': 25, 'Electronegativity': 1.55, 'Radius_pm': 139, 'Group': 7, 'Period': 4, 'MW': 54.94},
    'Cr': {'Z': 24, 'Electronegativity': 1.66, 'Radius_pm': 128, 'Group': 6, 'Period': 4, 'MW': 51.99},
    'Ti': {'Z': 22, 'Electronegativity': 1.54, 'Radius_pm': 147, 'Group': 4, 'Period': 4, 'MW': 47.87},
    'Al': {'Z': 13, 'Electronegativity': 1.61, 'Radius_pm': 121, 'Group': 13, 'Period': 3, 'MW': 26.98},
    'Mg': {'Z': 12, 'Electronegativity': 1.31, 'Radius_pm': 141, 'Group': 2, 'Period': 3, 'MW': 24.31},
    'Ce': {'Z': 58, 'Electronegativity': 1.12, 'Radius_pm': 181, 'Group': 3, 'Period': 6, 'MW': 140.12}
}

anion_mw = {
    'Nitrato': 62.00 * 2,
    'Acetato': 59.04 * 2,
    'Cloruro': 35.45 * 2,
    'Altro': 60.00
}

# --- FUNZIONE RESOLVER UNIVERSALE ---
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
            smiles_candidate = res.text.strip()
            if Chem.MolFromSmiles(smiles_candidate):
                return smiles_candidate
    except Exception:
        pass

    try:
        url_name = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(query)}/property/IsomericSMILES/JSON"
        res = requests.get(url_name, headers=headers, timeout=3)
        if res.status_code == 200:
            return res.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
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
        anione_sel = str(row.get('Anione', 'Nitrato'))
        
        m_leg = float(row.get('mmol legante', 0.1)) if pd.notnull(row.get('mmol legante')) else 0.1
        m_sale = float(row.get('mmol sale', 0.1)) if pd.notnull(row.get('mmol sale')) else 0.1
        ratio = m_leg / m_sale if m_sale > 0 else 1.0
        
        solv_p = str(row.get('Solvente', 'DMF'))
        cosolv = str(row.get('CoSolvente', 'Nessuno'))
        
        ml_solv_p = float(row.get('mL_Solvente_P', 10.0)) if pd.notnull(row.get('mL_Solvente_P')) else 10.0
        ml_cosolv = float(row.get('mL_CoSolvente', 0.0)) if pd.notnull(row.get('mL_CoSolvente')) else 0.0
        total_vol = ml_solv_p + ml_cosolv
        cosolv_pct = (ml_cosolv / total_vol * 100) if total_vol > 0 else 0.0
        
        add_type = str(row.get('Additivo_Tipo', 'None'))
        add_eq = float(row.get('Additivo_Eq', 0.0)) if pd.notnull(row.get('Additivo_Eq')) else 0.0
        
        temp = float(row.get('Temperatura_num', 120)) if pd.notnull(row.get('Temperatura_num')) else 120.0
        tempo = float(row.get('Tempo_ore_num', 48)) if pd.notnull(row.get('Tempo_ore_num')) else 48.0
        
        raw_target = row.get(target_col, 0) if target_col else 0
        try:
            target = int(float(raw_target))
        except:
            target = 0
            
        processed.append({
            'MW_Legante': float(mw), 'LogP_Legante': float(logp), 'HBD_Legante': float(hbd), 'HBA_Legante': float(hba),
            'TPSA_Legante': float(tpsa), 'RotatableBonds_Legante': float(rot),
            'Temperatura_num': float(temp), 'Tempo_ore_num': float(tempo),
            'mmol legante': float(m_leg), 'mmol sale': float(m_sale), 'Rapporto L/M': float(ratio),
            'Metallo_Z': m_info['Z'], 'Metallo_Electronegativity': m_info['Electronegativity'],
            'Metallo_Radius_pm': m_info['Radius_pm'], 'Metallo_Group': m_info['Group'], 'Metallo_Period': m_info['Period'],
            'Anion_Acetato': 1 if anione_sel == 'Acetato' else 0,
            'Anion_Cloruro': 1 if anione_sel == 'Cloruro' else 0,
            'Anion_Nitrato': 1 if anione_sel == 'Nitrato' else 0,
            'Anion_Altro': 1 if anione_sel == 'Altro' else 0,
            'mL_Solvente_P': float(ml_solv_p), 'mL_CoSolvente': float(ml_cosolv), 'Total_Volume_mL': float(total_vol),
            'CoSolvent_Pct': float(cosolv_pct),
            'Additive_Eq': float(add_eq),
            'Additive_Is_Acid': 1 if add_type == 'Acid' else 0,
            'Additive_Is_Base': 1 if add_type == 'Base' else 0,
            'Additive_Is_Neutral': 1 if add_type == 'Neutral' else 0,
            'Solvent_DMF': 1 if 'DMF' in solv_p or 'DMF' in cosolv else 0,
            'Solvent_H2O': 1 if 'H2O' in solv_p or 'H2O' in cosolv else 0,
            'Solvent_MeOH': 1 if 'MeOH' in solv_p or 'MeOH' in cosolv else 0,
            'Solvent_EtOH': 1 if 'EtOH' in solv_p or 'EtOH' in cosolv else 0,
            'Solvent_DEF': 1 if 'DEF' in solv_p or 'DEF' in cosolv else 0,
            'Solvent_MeCN': 1 if 'MeCN' in solv_p or 'MeCN' in cosolv else 0,
            'Target_Esito_Classe': target
        })
    return pd.DataFrame(processed)

# --- ADDESTRAMENTO MODELLO POTENZIATO CON LIGHTGBM E CALIBRAZIONE ---
@st.cache_resource
def load_or_train_model():
    pkl_file = "modello_sintesi_mof_ottimizzato.pkl"
    csv_file = "Dataset_Sintesi_Unificato.csv"
    
    if os.path.exists(pkl_file):
        return joblib.load(pkl_file)
    elif os.path.exists(csv_file):
        st.info("⚡ Addestramento del modello avanzato LightGBM con calibrazione chimica...")
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

        # Base LightGBM con Pesi Bilanciati per evitare bias verso l'insuccesso
        base_model = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        
        # Calibrazione Isotonica per probabilità chimicamente realistiche
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method='isotonic',
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        )
        
        calibrated_model.fit(X, y)
        
        # Preservazione metadati feature
        calibrated_model.feature_names_in_ = X.columns.tolist()
        
        try:
            importances = np.mean([est.estimator.feature_importances_ for est in calibrated_model.calibrated_classifiers_], axis=0)
            calibrated_model.feature_importances_ = importances
        except Exception:
            calibrated_model.feature_importances_ = np.zeros(X.shape[1])

        joblib.dump(calibrated_model, pkl_file)
        return calibrated_model
    else:
        st.error(f"File '{csv_file}' non trovato!")
        st.stop()

try:
    model = load_or_train_model()
    st.sidebar.success("Modello ML LightGBM attivo e pronto!")
except Exception as e:
    st.sidebar.error(f"Errore caricamento modello: {e}")
    st.stop()

# --- SIDEBAR: FEATURE IMPORTANCE ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Importanza Globale Parametri")
if hasattr(model, 'feature_importances_'):
    importances = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values(ascending=True).tail(8)
    st.sidebar.bar_chart(importances)

# --- TAB INTERFACCIA ---
tab1, tab2, tab3 = st.tabs(["🔮 Predizione Singola", "📂 Predizione Batch", "⚡ Ottimizzatore Automatico"])

def build_feature_row(mw, logp, hbd, hba, tpsa, rot_bonds, temp, tempo, mmol_legante, mmol_sale, metallo_sel, anione_sel, solvente_p, ml_solv_p, cosolvente, ml_cosolv, additivo_sel, add_eq):
    add_info = ADDITIVES_DATABASE.get(additivo_sel, ADDITIVES_DATABASE['Nessuno'])
    add_type = add_info['type']
    
    total_vol = float(ml_solv_p) + float(ml_cosolv)
    cosolv_pct = (float(ml_cosolv) / total_vol * 100.0) if total_vol > 0 else 0.0
    
    input_dict = {
        'MW_Legante': float(mw),
        'LogP_Legante': float(logp),
        'HBD_Legante': float(hbd),
        'HBA_Legante': float(hba),
        'TPSA_Legante': float(tpsa),
        'RotatableBonds_Legante': float(rot_bonds),
        'Temperatura_num': float(temp),
        'Tempo_ore_num': float(tempo),
        'mmol legante': float(mmol_legante),
        'mmol sale': float(mmol_sale),
        'Rapporto L/M': float(mmol_legante) / float(mmol_sale) if float(mmol_sale) > 0 else 1.0,
        'Metallo_Z': metal_props[metallo_sel]['Z'],
        'Metallo_Electronegativity': metal_props[metallo_sel]['Electronegativity'],
        'Metallo_Radius_pm': metal_props[metallo_sel]['Radius_pm'],
        'Metallo_Group': metal_props[metallo_sel]['Group'],
        'Metallo_Period': metal_props[metallo_sel]['Period'],
        'Anion_Acetato': 1 if anione_sel == 'Acetato' else 0,
        'Anion_Cloruro': 1 if anione_sel == 'Cloruro' else 0,
        'Anion_Nitrato': 1 if anione_sel == 'Nitrato' else 0,
        'Anion_Altro': 1 if anione_sel == 'Altro' else 0,
        'mL_Solvente_P': float(ml_solv_p),
        'mL_CoSolvente': float(ml_cosolv),
        'Total_Volume_mL': float(total_vol),
        'CoSolvent_Pct': float(cosolv_pct),
        'Additive_Eq': float(add_eq),
        'Additive_Is_Acid': 1 if add_type == 'Acid' else 0,
        'Additive_Is_Base': 1 if add_type == 'Base' else 0,
        'Additive_Is_Neutral': 1 if add_type == 'Neutral' else 0,
        'Solvent_DMF': 1 if 'DMF' in solvente_p or 'DMF' in cosolvente else 0,
        'Solvent_H2O': 1 if 'H2O' in solvente_p or 'H2O' in cosolvente else 0,
        'Solvent_MeOH': 1 if 'MeOH' in solvente_p or 'MeOH' in cosolvente else 0,
        'Solvent_EtOH': 1 if 'EtOH' in solvente_p or 'EtOH' in cosolvente else 0,
        'Solvent_DEF': 1 if 'DEF' in solvente_p or 'DEF' in cosolvente else 0,
        'Solvent_MeCN': 1 if 'MeCN' in solvente_p or 'MeCN' in cosolvente else 0,
    }
    
    df_f = pd.DataFrame([input_dict])
    
    for col in model.feature_names_in_:
        if col not in df_f.columns:
            df_f[col] = 0.0
            
    return df_f[model.feature_names_in_]

# --- TAB 1: PREDIZIONE SINGOLA ---
with tab1:
    st.subheader("Inserisci i parametri della reazione")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1. Legante Chimico")
        mode_legante = st.radio(
            "Modalità Input Legante:", 
            ["SMILES", "Nome / Formula / CAS", "Carica File (.mol / .sdf / .cif)"],
            horizontal=True
        )
        
        mol = None
        if mode_legante == "SMILES":
            smiles_input = st.text_input("SMILES del Legante:", value="c1cc(C(=O)O)cc(C(=O)O)c1")
            if smiles_input:
                mol = Chem.MolFromSmiles(smiles_input)
                
        elif mode_legante == "Nome / Formula / CAS":
            query_input = st.text_input("Nome, Formula o CAS:", value="Benzoic acid")
            if query_input:
                with st.spinner("Ricerca molecola nei database..."):
                    found_smiles = resolve_molecule_to_smiles(query_input)
                    if found_smiles:
                        mol = Chem.MolFromSmiles(found_smiles)
                        st.caption(f"SMILES Identificato: `{found_smiles}`")
                    else:
                        st.error("Nessuna molecola trovata.")
                        
        elif mode_legante == "Carica File (.mol / .sdf / .cif)":
            uploaded_file = st.file_uploader("Carica file .mol, .sdf o .cif", type=['mol', 'sdf', 'cif'])
            if uploaded_file is not None:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                file_bytes = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                
                if file_ext in ['mol', 'sdf']:
                    mol = Chem.MolFromMolBlock(file_bytes)
                elif file_ext == 'cif':
                    if HAS_PYMATGEN:
                        try:
                            with open("temp_upload.cif", "w", encoding="utf-8") as f:
                                f.write(file_bytes)
                            struct = Structure.from_file("temp_upload.cif")
                            red_formula = struct.composition.reduced_formula
                            found_smiles = resolve_molecule_to_smiles(red_formula)
                            if found_smiles:
                                mol = Chem.MolFromSmiles(found_smiles)
                        except Exception as e:
                            st.error(f"Errore lettura CIF: {e}")

        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            st.success(f"Molecola Valida! MW: {mw:.2f} g/mol")
        else:
            mw, logp, hbd, hba, tpsa, rot_bonds = 166.13, 1.32, 2, 4, 74.6, 2

        input_mode_leg = st.radio("Inserisci Legante come:", ["MilliMoli (mmol)", "Massa (mg)"], key="rad_leg", horizontal=True)
        if input_mode_leg == "MilliMoli (mmol)":
            mmol_legante = st.number_input("mmol Legante:", min_value=0.001, max_value=20.0, value=0.10, step=0.01)
            mg_legante = mmol_legante * mw
            st.caption(f"⚖️ Corrispondono a **{mg_legante:.2f} mg** da pesare.")
        else:
            mg_legante = st.number_input("Massa Legante (mg pesati):", min_value=0.1, max_value=5000.0, value=16.61, step=1.0)
            mmol_legante = mg_legante / mw if mw > 0 else 0.1
            st.caption(f"⚖️ Corrispondono a **{mmol_legante:.3f} mmol** di Legante.")

    with col2:
        st.markdown("### 2. Sale Metallico & Idratazione")
        metal_list = sorted(list(metal_props.keys()))
        metallo_sel = st.selectbox("Metallo:", metal_list, index=metal_list.index('Cu') if 'Cu' in metal_list else 0)
        anione_sel = st.selectbox("Anione / Precursore:", ['Nitrato', 'Acetato', 'Cloruro', 'Altro'])
        
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
            index=3
        )
        
        n_h2o = int(idratazione.split('(')[1].split(' ')[0])
        base_salt_mw = metal_props[metallo_sel]['MW'] + anion_mw.get(anione_sel, 60.0)
        total_salt_mw = base_salt_mw + (n_h2o * 18.015)
        
        st.caption(f"🧪 **Massa Molare Sale Idrato:** `{total_salt_mw:.2f} g/mol`")

        input_mode_sale = st.radio("Inserisci Sale come:", ["MilliMoli (mmol)", "Massa (mg)"], key="rad_sale", horizontal=True)
        if input_mode_sale == "MilliMoli (mmol)":
            mmol_sale = st.number_input("mmol Sale Metallico:", min_value=0.001, max_value=20.0, value=0.10, step=0.01)
            mg_sale = mmol_sale * total_salt_mw
            st.caption(f"⚖️ Corrispondono a **{mg_sale:.2f} mg** da pesare.")
        else:
            mg_sale = st.number_input("Massa Sale (mg pesati):", min_value=0.1, max_value=5000.0, value=24.16, step=1.0)
            mmol_sale = mg_sale / total_salt_mw
            st.caption(f"⚖️ Corrispondono a **{mmol_sale:.3f} mmol** di {metallo_sel}.")

    with col3:
        st.markdown("### 3. Miscela Solvente (mL) & Modulatori")
        
        solvente_p = st.selectbox("Solvente Principale:", ['DMF', 'DEF', 'DMSO', 'MeCN', 'H2O', 'MeOH', 'EtOH'])
        ml_solv_p = st.number_input(f"mL di {solvente_p}:", min_value=0.1, max_value=200.0, value=10.0, step=0.5)
        
        co_solvente = st.selectbox("Co-Solvente (Opzionale):", ['Nessuno', 'H2O', 'MeOH', 'EtOH', 'CH2Cl2', 'DEF'])
        
        ml_cosolv = 0.0
        if co_solvente != 'Nessuno':
            ml_cosolv = st.number_input(f"mL di Co-solvente ({co_solvente}):", min_value=0.1, max_value=200.0, value=2.0, step=0.5)

        tot_vol = ml_solv_p + ml_cosolv
        cosolv_pct = (ml_cosolv / tot_vol * 100) if tot_vol > 0 else 0.0
        st.caption(f"🧪 **Volume Totale Miscela:** `{tot_vol:.1f} mL` | **Co-solvente:** `{cosolv_pct:.1f}% v/v`")

        temp = st.number_input("Temperatura (°C):", min_value=20.0, max_value=250.0, value=120.0, step=5.0)
        tempo = st.number_input("Tempo di Reazione (Ore):", min_value=1.0, max_value=168.0, value=48.0, step=6.0)

        st.markdown("---")
        use_add = st.checkbox("➕ Aggiungi Additivo / Modulatore (Base/Acido)")
        
        additivo_sel = 'Nessuno'
        add_eq = 0.0
        if use_add:
            additivo_sel = st.selectbox("Seleziona Additivo:", list(ADDITIVES_DATABASE.keys())[1:])
            add_mode = st.radio("Inserisci quantità additivo come:", ["Equivalenti (vs Legante)", "mmol Additivo"], horizontal=True)
            
            if add_mode == "Equivalenti (vs Legante)":
                add_eq = st.number_input("Equivalenti rispetto al Legante:", min_value=0.1, max_value=100.0, value=2.0, step=0.5)
                add_mmol = add_eq * mmol_legante
                st.caption(f"🧪 Corrispondono a **{add_mmol:.3f} mmol** di additivo.")
            else:
                add_mmol = st.number_input("mmol Additivo:", min_value=0.001, max_value=50.0, value=0.20, step=0.05)
                add_eq = add_mmol / mmol_legante if mmol_legante > 0 else 0.0
                st.caption(f"🧪 Corrispondono a **{add_eq:.2f} eq.** rispetto al Legante.")

    if st.button("🚀 Calcola Probabilità di Successo", type="primary"):
        if not mol:
            st.error("Inserisci una molecola valida prima di continuare.")
        else:
            df_features = build_feature_row(
                mw, logp, hbd, hba, tpsa, rot_bonds, temp, tempo, 
                mmol_legante, mmol_sale, metallo_sel, anione_sel, 
                solvente_p, ml_solv_p, co_solvente, ml_cosolv, additivo_sel, add_eq
            )
            probs = model.predict_proba(df_features)[0]
            pred_class = model.predict(df_features)[0]

            st.markdown("---")
            st.subheader("📊 Risultato della Predizione")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            classes_map = {cls: idx for idx, cls in enumerate(model.classes_)}
            p0 = probs[classes_map[0]] * 100 if 0 in classes_map else 0.0
            p1 = probs[classes_map[1]] * 100 if 1 in classes_map else 0.0
            p2 = probs[classes_map[2]] * 100 if 2 in classes_map else 0.0

            res_col1.metric("🔴 Insuccesso (0)", f"{p0:.1f}%")
            res_col2.metric("🟡 Parziale (1)", f"{p1:.1f}%")
            res_col3.metric("🟢 Cristalli / Successo (2)", f"{p2:.1f}%")

            if pred_class == 2:
                st.balloons()
                st.success("✨ **Sintesi Promettente!** Alta probabilità di formazione di monocristalli o fase pulita.")
            elif pred_class == 1:
                st.warning("⚠️ **Risultato Parziale Atteso.** Possibile prodotto amorfo o miscela.")
            else:
                st.error("❌ **Insuccesso Probabile.** Si consiglia di rivedere le condizioni di reazione.")

            # SPIEGABILITÀ CHIMICA
            st.markdown("---")
            st.subheader("🧬 Spiegabilità Chimica della Predizione")
            
            rendered = False
            if HAS_SHAP:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(df_features)
                    target_idx = classes_map.get(2, len(shap_values) - 1) if isinstance(shap_values, list) else 0
                    
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
                        X_batch[col] = 0.0
                X_batch = X_batch[model.feature_names_in_]
                
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)
                
                classes_list = list(model.classes_)
                idx0 = classes_list.index(0) if 0 in classes_list else None
                idx1 = classes_list.index(1) if 1 in classes_list else None
                idx2 = classes_list.index(2) if 2 in classes_list else None
                
                results_df = input_batch.copy()
                results_df['Predizione_Classe'] = preds
                results_df['Prob_Insuccesso_%'] = (probs[:, idx0] * 100).round(1) if idx0 is not None else 0.0
                results_df['Prob_Parziale_%'] = (probs[:, idx1] * 100).round(1) if idx1 is not None else 0.0
                results_df['Prob_Successo_%'] = (probs[:, idx2] * 100).round(1) if idx2 is not None else 0.0
                
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

# --- TAB 3: OTTIMIZZATORE AUTOMATICO ---
with tab3:
    st.subheader("⚡ Ottimizzatore di Condizioni Sperimentali con Modulatori e Volumi")
    st.markdown("L'IA cercherà la **combinazione ottimale di temperatura, volumi di solventi (mL) e modulatori** per massimizzare la cristalizzazione del MOF.")
    
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        opt_smiles = st.text_input("SMILES Legante:", value="c1cc(C(=O)O)cc(C(=O)O)c1", key="opt_smiles")
        opt_mol = Chem.MolFromSmiles(opt_smiles)
    with opt_col2:
        metal_list_opt = sorted(list(metal_props.keys()))
        opt_metallo = st.selectbox("Metallo Desiderato:", metal_list_opt, index=metal_list_opt.index('Cu') if 'Cu' in metal_list_opt else 0, key="opt_met")
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
            
            temperatures = [100.0, 120.0, 140.0, 160.0]
            times = [24.0, 48.0, 72.0]
            solvents_p = ['DMF', 'DEF', 'DMSO']
            volumes_p = [5.0, 10.0]
            cosolvents = [('Nessuno', 0.0), ('H2O', 1.0), ('MeOH', 2.0)]
            additives = [('Nessuno', 0.0), ('Acido Acetico (AcOH)', 2.0), ('Trietilammina (TEA)', 1.0)]
            
            candidates = []
            
            # Dynamic Target Index Mapping
            classes_list = [int(c) if str(c).isdigit() else c for c in model.classes_]
            
            if 2 in classes_list:
                target_class_idx = classes_list.index(2)
            elif 1 in classes_list:
                target_class_idx = classes_list.index(1)
            else:
                target_class_idx = len(classes_list) - 1

            with st.spinner("Simulazione dello spazio di reazione in corso..."):
                for t in temperatures:
                    for tm in times:
                        for sp in solvents_p:
                            for ml_sp in volumes_p:
                                for cs, ml_cs in cosolvents:
                                    for add_name, add_eq in additives:
                                        feat = build_feature_row(
                                            opt_mw, opt_logp, opt_hbd, opt_hba, opt_tpsa, opt_rot, 
                                            t, tm, 0.1, 0.1, opt_metallo, opt_anione, 
                                            sp, ml_sp, cs, ml_cs, add_name, add_eq
                                        )
                                        
                                        prob_array = model.predict_proba(feat)[0]
                                        p_success = float(prob_array[target_class_idx]) * 100.0
                                        
                                        candidates.append({
                                            'Temperatura (°C)': t,
                                            'Tempo (h)': tm,
                                            'Solvente P. (mL)': f"{sp} ({ml_sp} mL)",
                                            'Co-Solvente (mL)': f"{cs} ({ml_cs} mL)" if cs != 'Nessuno' else 'Nessuno',
                                            'Vol. Totale (mL)': ml_sp + ml_cs,
                                            'Additivo / Modulatore': f"{add_name} ({add_eq} eq)" if add_name != 'Nessuno' else 'Nessuno',
                                            'Prob. Successo (%)': round(p_success, 1)
                                        })
            
            opt_df = pd.DataFrame(candidates).sort_values(by='Prob. Successo (%)', ascending=False).reset_index(drop=True)
            
            st.success("✅ Ottimizzazione completata!")
            st.markdown("### 🏆 Migliori Condizioni Sperimentali Identificate")
            
            best = opt_df.iloc[0]
            st.metric("Top Probabilità di Successo", f"{best['Prob. Successo (%)']}%")
            st.dataframe(opt_df.head(15))
            
            csv_opt = opt_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Scarica tutte le condizioni simulate",
                data=csv_opt,
                file_name="Ottimizzazione_Sintesi_MOF.csv",
                mime="text/csv"
            )
