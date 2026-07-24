import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import requests
import matplotlib.pyplot as plt

# Import opzionale per SHAP (Spiegabilità Chimica)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

st.set_page_config(page_title="MOF Synthesis Predictor & Optimizer", page_icon="🧪", layout="wide")
st.title("🧪 Predictor & Optimizer per Sintesi di MOF")
st.markdown("Strumento avanzato di Machine Learning per la predizione, ottimizzazione e **spiegabilità chimica** della sintesi di MOF.")

# --- CONFIGURAZIONE TAVILY AI ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

with st.sidebar.expander("🌐 Configurazione Agent Web (Tavily)", expanded=False):
    tavily_input_key = st.text_input("Tavily API Key:", value=TAVILY_API_KEY, type="password")
    if tavily_input_key:
        TAVILY_API_KEY = tavily_input_key

def search_tavily_web(query, max_results=3):
    if not TAVILY_API_KEY:
        return None
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "max_results": max_results
        }
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.warning(f"Errore ricerca Tavily: {e}")
    return None

def search_tavily_for_ligand_smiles(query):
    search_prompt = f"chemical SMILES string for {query} MOF ligand"
    res = search_tavily_web(search_prompt)
    if res and "results" in res:
        for item in res["results"]:
            content = item.get("content", "")
            words = content.replace(";", " ").replace("\n", " ").split(" ")
            for w in words:
                w_clean = w.strip(".,()[]{}")
                if len(w_clean) > 3 and Chem.MolFromSmiles(w_clean):
                    return w_clean
    return None

# --- DATABASE SOLVENTI E PROPRIETÀ CHIMICHE ESTESE ---
SOLVENT_PROPERTIES = {
    'DMF':  {'alpha': 0.00, 'beta': 0.69, 'pi_star': 0.88, 'dielectric': 36.7, 'boiling_pt': 153.0, 'viscosity': 0.92, 'dipole_moment': 3.82, 'molar_vol': 77.0},
    'DEF':  {'alpha': 0.00, 'beta': 0.69, 'pi_star': 0.88, 'dielectric': 32.1, 'boiling_pt': 177.0, 'viscosity': 1.15, 'dipole_moment': 3.90, 'molar_vol': 108.0},
    'DMSO': {'alpha': 0.00, 'beta': 0.76, 'pi_star': 1.00, 'dielectric': 46.7, 'boiling_pt': 189.0, 'viscosity': 1.99, 'dipole_moment': 3.96, 'molar_vol': 71.0},
    'MeCN': {'alpha': 0.19, 'beta': 0.31, 'pi_star': 0.75, 'dielectric': 37.5, 'boiling_pt': 82.0,  'viscosity': 0.34, 'dipole_moment': 3.92, 'molar_vol': 52.6},
    'H2O':  {'alpha': 1.17, 'beta': 0.18, 'pi_star': 1.09, 'dielectric': 80.1, 'boiling_pt': 100.0, 'viscosity': 0.89, 'dipole_moment': 1.85, 'molar_vol': 18.0},
    'MeOH': {'alpha': 0.93, 'beta': 0.62, 'pi_star': 0.60, 'dielectric': 32.7, 'boiling_pt': 64.7,  'viscosity': 0.54, 'dipole_moment': 1.70, 'molar_vol': 40.5},
    'EtOH': {'alpha': 0.83, 'beta': 0.77, 'pi_star': 0.54, 'dielectric': 24.5, 'boiling_pt': 78.3,  'viscosity': 1.07, 'dipole_moment': 1.69, 'molar_vol': 58.5},
    'CH2Cl2': {'alpha': 0.13, 'beta': 0.10, 'pi_star': 0.82, 'dielectric': 8.9,  'boiling_pt': 39.6,  'viscosity': 0.41, 'dipole_moment': 1.60, 'molar_vol': 64.0},
    'THF':   {'alpha': 0.00, 'beta': 0.55, 'pi_star': 0.58, 'dielectric': 7.58, 'boiling_pt': 66.0,  'viscosity': 0.48, 'dipole_moment': 1.75, 'molar_vol': 81.0},
    'Acetone': {'alpha': 0.08, 'beta': 0.48, 'pi_star': 0.71, 'dielectric': 20.7, 'boiling_pt': 56.0, 'viscosity': 0.32, 'dipole_moment': 2.88, 'molar_vol': 74.0},
    'Nessuno': {'alpha': 0.00, 'beta': 0.00, 'pi_star': 0.00, 'dielectric': 0.0,  'boiling_pt': 0.0,   'viscosity': 0.00, 'dipole_moment': 0.00, 'molar_vol': 0.0}
}

COMMON_MOF_LIGANDS = {
    "c7h6o2": "O=C(O)c1ccccc1", "benzoic acid": "O=C(O)c1ccccc1",
    "c2h4o2": "CC(=O)O", "acetic acid": "CC(=O)O",
    "c1h2o2": "O=CO", "formic acid": "O=CO",
    "c2hf3o2": "O=C(O)C(F)(F)F", "tfa": "O=C(O)C(F)(F)F",
    "c8h6o4": "O=C(O)c1ccc(C(=O)O)cc1", "bdc": "O=C(O)c1ccc(C(=O)O)cc1",
    "c9h6o6": "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1", "btc": "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1",
    "2-mim": "Cc1c[nH]cn1", "h2bdc": "O=C(O)c1ccc(C(=O)O)cc1",
    "h3btc": "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1", "bpdc": "O=C(O)c1ccc(cc1)-c1ccc(C(=O)O)cc1"
}

ADDITIVES_DATABASE = {
    'Nessuno': {'type': 'None', 'MW': 0.0, 'pKa': 0.0, 'density': 0.0},
    'Acido Acetico (AcOH)': {'type': 'Acid', 'MW': 60.05, 'pKa': 4.76, 'density': 1.05},
    'Acido Formico (HCOOH)': {'type': 'Acid', 'MW': 46.03, 'pKa': 3.75, 'density': 1.22},
    'Acido Benzoico': {'type': 'Acid', 'MW': 122.12, 'pKa': 4.20, 'density': 1.27},
    'Acido Trifluoroacetico (TFA)': {'type': 'Acid', 'MW': 114.02, 'pKa': 0.23, 'density': 1.49},
    'Acido Cloridrico (HCl)': {'type': 'Acid', 'MW': 36.46, 'pKa': -6.30, 'density': 1.19},
    'Trietilammina (TEA)': {'type': 'Base', 'MW': 101.19, 'pKa': 10.75, 'density': 0.728},
    'Piridina': {'type': 'Base', 'MW': 79.10, 'pKa': 5.25, 'density': 0.982},
    'NaOH': {'type': 'Base', 'MW': 40.00, 'pKa': 13.8, 'density': 2.13},
    'HF': {'type': 'Acid', 'MW': 20.01, 'pKa': 3.17, 'density': 1.15}
}

metal_props = {
    'Cu': {'Z': 29, 'Electronegativity': 1.90, 'Radius_pm': 132, 'Group': 11, 'Period': 4, 'MW': 63.55, 'HSAB': 'Intermediate', 'Valence_Common': 2},
    'Zn': {'Z': 30, 'Electronegativity': 1.65, 'Radius_pm': 122, 'Group': 12, 'Period': 4, 'MW': 65.38, 'HSAB': 'Intermediate', 'Valence_Common': 2},
    'Zr': {'Z': 40, 'Electronegativity': 1.33, 'Radius_pm': 160, 'Group': 4,  'Period': 5, 'MW': 91.22, 'HSAB': 'Hard',         'Valence_Common': 4},
    'Fe': {'Z': 26, 'Electronegativity': 1.83, 'Radius_pm': 126, 'Group': 8,  'Period': 4, 'MW': 55.85, 'HSAB': 'Hard',         'Valence_Common': 3},
    'Co': {'Z': 27, 'Electronegativity': 1.88, 'Radius_pm': 126, 'Group': 9,  'Period': 4, 'MW': 58.93, 'HSAB': 'Intermediate', 'Valence_Common': 2},
    'Ni': {'Z': 28, 'Electronegativity': 1.91, 'Radius_pm': 124, 'Group': 10, 'Period': 4, 'MW': 58.69, 'HSAB': 'Intermediate', 'Valence_Common': 2},
    'Mn': {'Z': 25, 'Electronegativity': 1.55, 'Radius_pm': 139, 'Group': 7,  'Period': 4, 'MW': 54.94, 'HSAB': 'Intermediate', 'Valence_Common': 2},
    'Al': {'Z': 13, 'Electronegativity': 1.61, 'Radius_pm': 121, 'Group': 13, 'Period': 3, 'MW': 26.98, 'HSAB': 'Hard',         'Valence_Common': 3},
    'Cr': {'Z': 24, 'Electronegativity': 1.66, 'Radius_pm': 128, 'Group': 6,  'Period': 4, 'MW': 51.996,'HSAB': 'Hard',         'Valence_Common': 3},
    'Mg': {'Z': 12, 'Electronegativity': 1.31, 'Radius_pm': 160, 'Group': 2,  'Period': 3, 'MW': 24.305,'HSAB': 'Hard',         'Valence_Common': 2},
    'Cd': {'Z': 48, 'Electronegativity': 1.69, 'Radius_pm': 154, 'Group': 12, 'Period': 5, 'MW': 112.41,'HSAB': 'Soft',         'Valence_Common': 2},
    'Ln': {'Z': 57, 'Electronegativity': 1.10, 'Radius_pm': 187, 'Group': 3,  'Period': 6, 'MW': 138.90,'HSAB': 'Hard',         'Valence_Common': 3}
}

SMARTS_PATTERNS = {
    'COOH': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),
    'COO_minus': Chem.MolFromSmarts('[CX3](=O)[O-]'),
    'Aromatic_N': Chem.MolFromSmarts('[n]'),
    'Pyridine_N': Chem.MolFromSmarts('c1cnccc1'),
    'Imidazole_N': Chem.MolFromSmarts('c1cncn1'),
    'Phenol_OH': Chem.MolFromSmarts('c[OH]'),
    'Aliphatic_OH': Chem.MolFromSmarts('[CX4][OH]'),
    'Amine_Primary': Chem.MolFromSmarts('[NX3H2]'),
    'Amine_Secondary': Chem.MolFromSmarts('[NX3H1]'),
    'Amine_Tertiary': Chem.MolFromSmarts('[NX3H0]'),
    'Nitro': Chem.MolFromSmarts('[N+](=O)[O-]'),
    'Halogen': Chem.MolFromSmarts('[F,Cl,Br,I]'),
    'Sulfonate': Chem.MolFromSmarts('S(=O)(=O)[O-]'),
    'Phosphonate': Chem.MolFromSmarts('P(=O)([O-])[O-]')
}

def clean_float_val(val, default_val=0.0):
    """ Converte in float gestendo stringhe speciali come T.A. (Temperatura Ambiente = 25) """
    if pd.isna(val):
        return float(default_val)
    s_val = str(val).strip().upper()
    if s_val in ['T.A.', 'TA', 'RT', 'ROOM TEMP', 'ROOM TEMPERATURA', 'AMBIENTE']:
        return 25.0
    # Rimozione caratteri non numerici salvo punto e segno
    s_clean = re.sub(r'[^0-9\.-]', '', str(val))
    try:
        return float(s_clean)
    except Exception:
        return float(default_val)

def extract_extended_rdkit_descriptors(mol):
    if not mol:
        return {
            'MW_Legante': 0.0, 'LogP_Legante': 0.0, 'HBD_Legante': 0, 'HBA_Legante': 0,
            'TPSA_Legante': 0.0, 'RotatableBonds_Legante': 0, 'AromaticRings_Legante': 0,
            'FractionCSP3_Legante': 0.0, 'HeavyAtomCount_Legante': 0, 'NHOHCount_Legante': 0,
            'NOCount_Legante': 0, 'RingCount_Legante': 0, 'LabuteASA_Legante': 0.0,
            'HallKierAlpha_Legante': 0.0, 'BertzCT_Legante': 0.0, 'Chi0v_Legante': 0.0,
            'Chi1v_Legante': 0.0, 'Kappa1_Legante': 0.0, 'Kappa2_Legante': 0.0
        }
    return {
        'MW_Legante': float(Descriptors.MolWt(mol)),
        'LogP_Legante': float(Descriptors.MolLogP(mol)),
        'HBD_Legante': int(Descriptors.NumHDonors(mol)),
        'HBA_Legante': int(Descriptors.NumHAcceptors(mol)),
        'TPSA_Legante': float(Descriptors.TPSA(mol)),
        'RotatableBonds_Legante': int(Descriptors.NumRotatableBonds(mol)),
        'AromaticRings_Legante': int(Descriptors.NumAromaticRings(mol)),
        'FractionCSP3_Legante': float(Descriptors.FractionCSP3(mol)),
        'HeavyAtomCount_Legante': int(Descriptors.HeavyAtomCount(mol)),
        'NHOHCount_Legante': int(Descriptors.NHOHCount(mol)),
        'NOCount_Legante': int(Descriptors.NOCount(mol)),
        'RingCount_Legante': int(Descriptors.RingCount(mol)),
        'LabuteASA_Legante': float(Descriptors.LabuteASA(mol)),
        'HallKierAlpha_Legante': float(Descriptors.HallKierAlpha(mol)),
        'BertzCT_Legante': float(Descriptors.BertzCT(mol)),
        'Chi0v_Legante': float(rdMolDescriptors.CalcChi0v(mol)),
        'Chi1v_Legante': float(rdMolDescriptors.CalcChi1v(mol)),
        'Kappa1_Legante': float(rdMolDescriptors.CalcKappa1(mol)),
        'Kappa2_Legante': float(rdMolDescriptors.CalcKappa2(mol))
    }

def extract_smarts_features(mol):
    if not mol:
        res = {f"SMARTS_n_{key}": 0 for key in SMARTS_PATTERNS.keys()}
        res['SMARTS_fraction_sp2'] = 0.0
        return res
    
    res = {}
    for name, pattern in SMARTS_PATTERNS.items():
        if pattern:
            res[f"SMARTS_n_{name}"] = len(mol.GetSubstructMatches(pattern))
        else:
            res[f"SMARTS_n_{name}"] = 0
            
    c_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
    fraction_sp2 = (sum(1 for atom in c_atoms if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2) / len(c_atoms)) if c_atoms else 0.0
    res['SMARTS_fraction_sp2'] = float(fraction_sp2)
    return res

def calculate_hsab_match(metal_hsab, n_cooh, n_aro_n):
    if metal_hsab == 'Hard':
        return 1.0 if n_cooh > 0 else 0.2
    elif metal_hsab == 'Intermediate':
        return 1.0 if (n_aro_n > 0 or n_cooh > 0) else 0.5
    elif metal_hsab == 'Soft':
        return 1.0 if n_aro_n > 0 else 0.3
    return 0.5

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
            s = res.text.strip()
            if Chem.MolFromSmiles(s):
                return s
    except Exception:
        pass
    try:
        url_pub = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(query)}/property/IsomericSMILES/JSON"
        res = requests.get(url_pub, headers=headers, timeout=3)
        if res.status_code == 200:
            return res.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
    except Exception:
        pass
    if TAVILY_API_KEY:
        return search_tavily_for_ligand_smiles(query)
    return None

def calculate_solvent_mix_properties(solv_p, ml_p, cosolv, ml_cosolv):
    prop_p = SOLVENT_PROPERTIES.get(solv_p, SOLVENT_PROPERTIES['DMF'])
    prop_co = SOLVENT_PROPERTIES.get(cosolv, SOLVENT_PROPERTIES['Nessuno'])
    tot_vol = ml_p + ml_cosolv
    if tot_vol <= 0:
        return {
            'mix_alpha': prop_p['alpha'], 'mix_beta': prop_p['beta'],
            'mix_pi_star': prop_p['pi_star'], 'mix_dielectric': prop_p['dielectric'],
            'mix_boiling_pt': prop_p['boiling_pt'], 'mix_viscosity': prop_p['viscosity'],
            'mix_dipole_moment': prop_p['dipole_moment'], 'mix_molar_vol': prop_p['molar_vol']
        }
    f_p, f_co = ml_p / tot_vol, ml_cosolv / tot_vol
    return {
        'mix_alpha': (prop_p['alpha'] * f_p) + (prop_co['alpha'] * f_co),
        'mix_beta': (prop_p['beta'] * f_p) + (prop_co['beta'] * f_co),
        'mix_pi_star': (prop_p['pi_star'] * f_p) + (prop_co['pi_star'] * f_co),
        'mix_dielectric': (prop_p['dielectric'] * f_p) + (prop_co['dielectric'] * f_co),
        'mix_boiling_pt': (prop_p['boiling_pt'] * f_p) + (prop_co['boiling_pt'] * f_co),
        'mix_viscosity': (prop_p['viscosity'] * f_p) + (prop_co['viscosity'] * f_co),
        'mix_dipole_moment': (prop_p['dipole_moment'] * f_p) + (prop_co['dipole_moment'] * f_co),
        'mix_molar_vol': (prop_p['molar_vol'] * f_p) + (prop_co['molar_vol'] * f_co)
    }

def process_unified_dataset(df):
    target_col = None
    for col in ['Esito_ML', 'Target_Esito_Classe', 'Esito', 'Target', 'Classe']:
        if col in df.columns:
            target_col = col
            break

    processed = []
    for idx, row in df.iterrows():
        smiles = str(row.get('SMILES_Legante', row.get('Legante standard', '')))
        mol = Chem.MolFromSmiles(smiles) if smiles and smiles != 'nan' else None
        
        rdkit_f = extract_extended_rdkit_descriptors(mol)
        smarts_f = extract_smarts_features(mol)
        
        met = str(row.get('Metallo', 'Cu')).strip()
        m_info = metal_props.get(met, metal_props['Cu'])
        anione_sel = str(row.get('Sale metallico', row.get('Anione', 'Nitrato'))).strip()
        
        hsab_match = calculate_hsab_match(m_info['HSAB'], smarts_f.get('SMARTS_n_COOH', 0), smarts_f.get('SMARTS_n_Aromatic_N', 0))
        
        m_leg = clean_float_val(row.get('mmol legante', 0.1), default_val=0.1)
        m_sale = clean_float_val(row.get('mmol sale', row.get('mmol metallo', 0.1)), default_val=0.1)
        
        raw_ratio = row.get('Rapporto L/M', np.nan)
        if pd.notnull(raw_ratio):
            ratio = clean_float_val(raw_ratio, default_val=(m_leg / m_sale if m_sale > 0 else 1.0))
        else:
            ratio = m_leg / m_sale if m_sale > 0 else 1.0
        
        solv_p = str(row.get('Solvente', 'DMF')).split('/')[0].strip()
        cosolv = 'Nessuno'
        if '/' in str(row.get('Solvente', '')):
            parts = str(row.get('Solvente', '')).split('/')
            if len(parts) > 1:
                cosolv = parts[1].strip()

        vol_tot = clean_float_val(row.get('Volume solvente', 10.0), default_val=10.0)
        ml_solv_p = vol_tot * 0.8
        ml_cosolv = vol_tot * 0.2 if cosolv != 'Nessuno' else 0.0
        cosolv_pct = (ml_cosolv / vol_tot * 100) if vol_tot > 0 else 0.0
        
        mix_props = calculate_solvent_mix_properties(solv_p, ml_solv_p, cosolv, ml_cosolv)
        
        add_type = str(row.get('Co-linker/Additivo', 'Nessuno'))
        add_eq = clean_float_val(row.get('Quantita additivo', 0.0), default_val=0.0)
        
        # --- PULIZIA TEMPERATURA E TEMPO (Gestisce 'T.A.' -> 25.0) ---
        temp = clean_float_val(row.get('Temperatura', 120.0), default_val=120.0)
        tempo = clean_float_val(row.get('Tempo ore', 48.0), default_val=48.0)
        
        # --- PULIZIA REGEX TARGET ---
        raw_target = row.get(target_col, np.nan) if target_col else np.nan
        target = np.nan
        if pd.notnull(raw_target):
            clean_str = re.sub(r'[\[\]\s\'\"]', '', str(raw_target))
            try:
                target = float(clean_str)
            except Exception:
                target = np.nan
            
        row_data = {
            'SMILES_Group': smiles if smiles and smiles != 'nan' else 'sconosciuto',
            'HSAB_Match_Index': float(hsab_match),
            'Temperatura_num': float(temp),
            'Tempo_ore_num': float(tempo),
            'mmol legante': float(m_leg),
            'mmol sale': float(m_sale),
            'Rapporto L/M': float(ratio),
            'Metallo_Z': m_info['Z'],
            'Metallo_Electronegativity': m_info['Electronegativity'],
            'Metallo_Radius_pm': m_info['Radius_pm'],
            'Metallo_Group': m_info['Group'],
            'Metallo_Period': m_info['Period'],
            'Metallo_Valence': m_info['Valence_Common'],
            'Anion_Acetato': 1 if 'acetat' in anione_sel.lower() else 0,
            'Anion_Cloruro': 1 if 'clor' in anione_sel.lower() or 'cl' in anione_sel.lower() else 0,
            'Anion_Nitrato': 1 if 'nitr' in anione_sel.lower() else 0,
            'Anion_Solfato': 1 if 'solf' in anione_sel.lower() or 'sulf' in anione_sel.lower() else 0,
            'Anion_Altro': 1 if not any(k in anione_sel.lower() for k in ['acetat', 'clor', 'cl', 'nitr', 'solf', 'sulf']) else 0,
            'mL_Solvente_P': float(ml_solv_p),
            'mL_CoSolvente': float(ml_cosolv),
            'Total_Volume_mL': float(vol_tot),
            'CoSolvent_Pct': float(cosolv_pct),
            'Solvent_Mix_Alpha': mix_props['mix_alpha'],
            'Solvent_Mix_Beta': mix_props['mix_beta'],
            'Solvent_Mix_PiStar': mix_props['mix_pi_star'],
            'Solvent_Mix_Dielectric': mix_props['mix_dielectric'],
            'Solvent_Mix_BoilingPt': mix_props['mix_boiling_pt'],
            'Solvent_Mix_Viscosity': mix_props['mix_viscosity'],
            'Solvent_Mix_Dipole': mix_props['mix_dipole_moment'],
            'Solvent_Mix_MolarVol': mix_props['mix_molar_vol'],
            'Additive_Eq': float(add_eq),
            'Additive_Is_Acid': 1 if 'acid' in add_type.lower() else 0,
            'Additive_Is_Base': 1 if 'base' in add_type.lower() or 'amine' in add_type.lower() or 'et3n' in add_type.lower() else 0,
            'Additive_Is_Neutral': 1 if add_type == 'Nessuno' or add_type == 'None' else 0,
            'Target_Esito_Classe': target
        }
        
        row_data.update(rdkit_f)
        row_data.update(smarts_f)
        processed.append(row_data)
        
    return pd.DataFrame(processed)

def create_stacking_ensemble():
    estimators = [
        ('lgb', LGBMClassifier(n_estimators=180, learning_rate=0.03, max_depth=6, num_leaves=31, class_weight='balanced', random_state=42, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1))
    ]
    if HAS_CATBOOST:
        estimators.append(('cat', CatBoostClassifier(iterations=200, learning_rate=0.04, depth=6, verbose=0, random_seed=42, auto_class_weights='Balanced')))
    
    meta_model = LogisticRegression(class_weight='balanced', max_iter=500)
    return StackingClassifier(estimators=estimators, final_estimator=meta_model, stack_method='predict_proba', n_jobs=-1)

@st.cache_resource
def load_or_train_model():
    pkl_file = "modello_sintesi_mof_ottimizzato.pkl"
    
    dataset_base = "Dataset_Sintesi_Unificato_aggiornato"
    if os.path.exists(f"{dataset_base}.csv"):
        csv_file = f"{dataset_base}.csv"
    elif os.path.exists(dataset_base):
        csv_file = dataset_base
    else:
        csv_file = None

    if os.path.exists(pkl_file):
        try:
            saved_data = joblib.load(pkl_file)
            if isinstance(saved_data, dict) and 'model' in saved_data:
                classes = getattr(saved_data['model'], 'classes_', [])
                if len(classes) >= 2:
                    return saved_data['model'], saved_data['features'], saved_data.get('metrics', {}), saved_data.get('importances', [])
        except Exception:
            pass

    if not csv_file:
        st.error("❌ Nessun dataset trovato! Assicurati che 'Dataset_Sintesi_Unificato_aggiornato.csv' sia presente nella cartella principale.")
        st.stop()
        
    raw_df = pd.read_csv(csv_file)
    df = process_unified_dataset(raw_df)
    
    valid_mask = df['Target_Esito_Classe'].notna()
    X = df.drop(columns=['Target_Esito_Classe', 'SMILES_Group'])[valid_mask].copy().reset_index(drop=True)
    y = df['Target_Esito_Classe'][valid_mask].astype(int).copy().reset_index(drop=True)
    
    # Conversione forzata a numerico di sicurezza
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    if y.nunique() < 2:
        st.error(f"⚠️ **Errore Dataset:** Trovati i seguenti valori per il Target: {y.unique().tolist()}. Il modello richiede almeno 2 classi distinte.")
        st.stop()

    feature_names = X.columns.tolist()
    final_model = create_stacking_ensemble()
    final_model.fit(X, y)

    try:
        importances = final_model.named_estimators_['lgb'].feature_importances_
    except Exception:
        importances = np.zeros(len(feature_names))

    metrics = {
        'train_accuracy': accuracy_score(y, final_model.predict(X)),
        'n_samples': len(X),
        'n_features': len(feature_names)
    }
    
    save_dict = {'model': final_model, 'features': feature_names, 'importances': importances, 'metrics': metrics}
    joblib.dump(save_dict, pkl_file)
    return final_model, feature_names, metrics, importances

try:
    model, feature_names, metrics, importances = load_or_train_model()
    st.sidebar.success("Modello Ensemble Stacking Attivo!")
except Exception as e:
    st.sidebar.error(f"Errore caricamento modello: {e}")
    st.stop()

# --- SIDEBAR INFO ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Stato & Performance Modello")
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    acc_val = metrics.get('train_accuracy', 0.85) * 100
    st.metric("Accuratezza", f"{acc_val:.1f}%")
with col_sb2:
    st.metric("Sintesi DB", metrics.get('n_samples', 'N/A'))

# --- TAB INTERFACCIA ---
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predizione Singola", "📂 Predizione Batch", "⚡ Ottimizzatore Automatico", "🌐 Ricerca Web (Tavily AI)"])

def build_feature_row(mol, temp, tempo, mmol_legante, mmol_sale, metallo_sel, anione_sel, solvente_p, ml_solv_p, cosolvente, ml_cosolv, additivo_sel, add_eq):
    add_info = ADDITIVES_DATABASE.get(additivo_sel, ADDITIVES_DATABASE['Nessuno'])
    add_type = add_info['type']
    total_vol = float(ml_solv_p) + float(ml_cosolv)
    cosolv_pct = (float(ml_cosolv) / total_vol * 100.0) if total_vol > 0 else 0.0
    mix_props = calculate_solvent_mix_properties(solvente_p, float(ml_solv_p), cosolvente, float(ml_cosolv))
    rdkit_f = extract_extended_rdkit_descriptors(mol)
    smarts_f = extract_smarts_features(mol)
    metal_m = metal_props[metallo_sel]
    hsab_match = calculate_hsab_match(metal_m['HSAB'], smarts_f.get('SMARTS_n_COOH', 0), smarts_f.get('SMARTS_n_Aromatic_N', 0))
    
    input_dict = {
        'HSAB_Match_Index': float(hsab_match),
        'Temperatura_num': float(temp),
        'Tempo_ore_num': float(tempo),
        'mmol legante': float(mmol_legante),
        'mmol sale': float(mmol_sale),
        'Rapporto L/M': float(mmol_legante) / float(mmol_sale) if float(mmol_sale) > 0 else 1.0,
        'Metallo_Z': metal_m['Z'],
        'Metallo_Electronegativity': metal_m['Electronegativity'],
        'Metallo_Radius_pm': metal_m['Radius_pm'],
        'Metallo_Group': metal_m['Group'],
        'Metallo_Period': metal_m['Period'],
        'Metallo_Valence': metal_m['Valence_Common'],
        'Anion_Acetato': 1 if 'acetat' in anione_sel.lower() else 0,
        'Anion_Cloruro': 1 if 'clor' in anione_sel.lower() else 0,
        'Anion_Nitrato': 1 if 'nitr' in anione_sel.lower() else 0,
        'Anion_Solfato': 1 if 'solf' in anione_sel.lower() or 'sulf' in anione_sel.lower() else 0,
        'Anion_Altro': 1 if not any(k in anione_sel.lower() for k in ['acetat', 'clor', 'cl', 'nitr', 'solf', 'sulf']) else 0,
        'mL_Solvente_P': float(ml_solv_p),
        'mL_CoSolvente': float(ml_cosolv),
        'Total_Volume_mL': float(total_vol),
        'CoSolvent_Pct': float(cosolv_pct),
        'Solvent_Mix_Alpha': mix_props['mix_alpha'],
        'Solvent_Mix_Beta': mix_props['mix_beta'],
        'Solvent_Mix_PiStar': mix_props['mix_pi_star'],
        'Solvent_Mix_Dielectric': mix_props['mix_dielectric'],
        'Solvent_Mix_BoilingPt': mix_props['mix_boiling_pt'],
        'Solvent_Mix_Viscosity': mix_props['mix_viscosity'],
        'Solvent_Mix_Dipole': mix_props['mix_dipole_moment'],
        'Solvent_Mix_MolarVol': mix_props['mix_molar_vol'],
        'Additive_Eq': float(add_eq),
        'Additive_Is_Acid': 1 if add_type == 'Acid' else 0,
        'Additive_Is_Base': 1 if add_type == 'Base' else 0,
        'Additive_Is_Neutral': 1 if add_type == 'None' else 0
    }
    
    input_dict.update(rdkit_f)
    input_dict.update(smarts_f)
    
    df_f = pd.DataFrame([input_dict])
    for col in feature_names:
        if col not in df_f.columns:
            df_f[col] = 0.0
    return df_f[feature_names]

# --- TAB 1: PREDIZIONE SINGOLA ---
with tab1:
    st.subheader("Inserisci i parametri della reazione")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1. Legante Chimico")
        mode_legante = st.radio("Modalità Input Legante:", ["SMILES", "Nome / Formula / CAS"], horizontal=True)
        mol = None
        if mode_legante == "SMILES":
            smiles_input = st.text_input("SMILES del Legante:", value="c1cc(C(=O)O)cc(C(=O)O)c1")
            if smiles_input:
                mol = Chem.MolFromSmiles(smiles_input)
        else:
            query_input = st.text_input("Nome, Formula o CAS:", value="Benzoic acid")
            if query_input:
                found_smiles = resolve_molecule_to_smiles(query_input)
                if found_smiles:
                    mol = Chem.MolFromSmiles(found_smiles)

        if mol:
            mw = Descriptors.MolWt(mol)
            st.success(f"Molecola Valida! MW: {mw:.2f} g/mol")
        
        mmol_legante = st.number_input("mmol Legante:", min_value=0.001, max_value=20.0, value=0.10, step=0.01)

    with col2:
        st.markdown("### 2. Sale Metallico")
        metal_list = sorted(list(metal_props.keys()))
        metallo_sel = st.selectbox("Metallo:", metal_list, index=metal_list.index('Cu') if 'Cu' in metal_list else 0)
        anione_sel = st.selectbox("Anione / Precursore:", ['Nitrato', 'Acetato', 'Cloruro', 'Solfato', 'Altro'])
        mmol_sale = st.number_input("mmol Sale Metallico:", min_value=0.001, max_value=20.0, value=0.10, step=0.01)

    with col3:
        st.markdown("### 3. Miscela Solvente & Condizioni")
        solvente_p = st.selectbox("Solvente Principale:", ['DMF', 'DEF', 'DMSO', 'MeCN', 'H2O', 'MeOH', 'EtOH', 'THF', 'Acetone'])
        ml_solv_p = st.number_input(f"mL di {solvente_p}:", min_value=0.1, max_value=200.0, value=10.0, step=0.5)
        co_solvente = st.selectbox("Co-Solvente:", ['Nessuno', 'H2O', 'MeOH', 'EtOH', 'CH2Cl2', 'DEF', 'THF'])
        ml_cosolv = st.number_input(f"mL di Co-solvente:", min_value=0.0, max_value=200.0, value=0.0, step=0.5) if co_solvente != 'Nessuno' else 0.0
        
        temp = st.number_input("Temperatura (°C):", min_value=20.0, max_value=250.0, value=120.0, step=5.0)
        tempo = st.number_input("Tempo (Ore):", min_value=1.0, max_value=168.0, value=48.0, step=6.0)

        additivo_sel = st.selectbox("Additivo / Modulatore:", list(ADDITIVES_DATABASE.keys()))
        add_eq = st.number_input("Equivalenti Additivo:", min_value=0.0, max_value=50.0, value=0.0, step=0.5) if additivo_sel != 'Nessuno' else 0.0

    if st.button("🚀 Calcola Probabilità di Successo", type="primary"):
        if not mol:
            st.error("Inserisci una molecola valida prima di continuare.")
        else:
            df_features = build_feature_row(
                mol, temp, tempo, mmol_legante, mmol_sale, metallo_sel, anione_sel, 
                solvente_p, ml_solv_p, co_solvente, ml_cosolv, additivo_sel, add_eq
            )
            probs = model.predict_proba(df_features)[0]
            pred_class = model.predict(df_features)[0]

            st.markdown("---")
            st.subheader("📊 Risultato della Predizione")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            classes_map = {int(cls): idx for idx, cls in enumerate(model.classes_)}
            p0 = probs[classes_map[0]] * 100 if 0 in classes_map else 0.0
            p1 = probs[classes_map[1]] * 100 if 1 in classes_map else 0.0
            p2 = probs[classes_map[2]] * 100 if 2 in classes_map else 0.0

            res_col1.metric("🔴 Insuccesso (0)", f"{p0:.1f}%")
            res_col2.metric("🟡 Intermedio (1)", f"{p1:.1f}%")
            res_col3.metric("🟢 Successo (2)", f"{p2:.1f}%")

            if pred_class == 2:
                st.balloons()
                st.success("✨ **Sintesi Promettente!** Alta probabilità di formazione di cristalli.")
            elif pred_class == 1:
                st.warning("⚠️ **Risultato Intermedio Atteso.** Formazione di fasi amorfe o microcristalline.")
            else:
                st.error("❌ **Insuccesso Probabile.** Nessun solido o precipitazione amorfa. Consigliabile modificare le condizioni.")

            # --- SEZIONE SHAP EXPLAINABILITY ---
            if HAS_SHAP:
                st.markdown("---")
                st.subheader("💡 Spiegabilità Chimica (SHAP Analysis)")
                try:
                    lgb_model = model.named_estimators_['lgb']
                    explainer = shap.TreeExplainer(lgb_model)
                    shap_values = explainer.shap_values(df_features)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[classes_map[pred_class]], df_features, plot_type="bar", show=False)
                    else:
                        shap.summary_plot(shap_values, df_features, plot_type="bar", show=False)
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.info(f"Visualizzazione SHAP non disponibile per questa configurazione ({e}).")

# --- TAB 2: BATCH ---
with tab2:
    st.subheader("Carica un file Excel o CSV per la predizione di più reazioni")
    uploaded_file = st.file_uploader("Carica File (.xlsx o .csv)", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        try:
            input_batch = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.write("📋 **Anteprima Dataset Caricato:**", input_batch.head())
            if st.button("⚡ Elabora Batch"):
                processed_batch = process_unified_dataset(input_batch)
                X_batch = processed_batch.drop(columns=['Target_Esito_Classe', 'SMILES_Group'])
                for col in feature_names:
                    if col not in X_batch.columns:
                        X_batch[col] = 0.0
                X_batch = X_batch[feature_names]
                preds = model.predict(X_batch)
                
                results_df = input_batch.copy()
                results_df['Predizione_Classe_ML'] = preds
                st.success("✅ Predizioni su scala completate!")
                st.dataframe(results_df)
        except Exception as e:
            st.error(f"Errore durante l'elaborazione del file: {e}")

# --- TAB 3: OTTIMIZZATORE ---
with tab3:
    st.subheader("⚡ Ottimizzatore Automatico di Condizioni Solvotermiche")
    st.markdown("Questa funzione simula automaticamente combinazioni di parametri (temperatura, tempo, solventi) per massimizzare la probabilità di ottenere cristalli (Classe 2).")
    
    opt_smiles = st.text_input("SMILES del Legante da Ottimizzare:", value="c1cc(C(=O)O)cc(C(=O)O)c1")
    opt_metal = st.selectbox("Metallo Target:", sorted(list(metal_props.keys())), index=0)
    
    if st.button("🔍 Avvia Griglia di Ottimizzazione"):
        opt_mol = Chem.MolFromSmiles(opt_smiles)
        if not opt_mol:
            st.error("SMILES non valido.")
        else:
            st.info("Ricerca delle condizioni ottimali in corso...")
            candidates = []
            
            for t_val in [100.0, 120.0, 150.0]:
                for solv in ['DMF', 'DEF', 'DMSO']:
                    for ratio_val in [0.5, 1.0, 2.0]:
                        df_cand = build_feature_row(
                            opt_mol, t_val, 48.0, 0.1 * ratio_val, 0.1, opt_metal, 'Nitrato',
                            solv, 10.0, 'Nessuno', 0.0, 'Nessuno', 0.0
                        )
                        p_success = model.predict_proba(df_cand)[0][-1]
                        candidates.append({
                            'Temperatura (°C)': t_val,
                            'Solvente': solv,
                            'Rapporto L/M': ratio_val,
                            'Probabilità Successo (%)': np.round(p_success * 100, 2)
                        })
            
            res_opt_df = pd.DataFrame(candidates).sort_values(by='Probabilità Successo (%)', ascending=False)
            st.success("🎯 Mappatura completata! Ecco le migliori 5 condizioni individuate:")
            st.table(res_opt_df.head(5))

# --- TAB 4: RICERCA TAVILY ---
with tab4:
    st.subheader("🌐 Agente Web con Ricerca Scientifico-Sintetica (Tavily AI)")
    query_tavily = st.text_input("Inserisci la query di ricerca chimica:", value="UiO-66 synthesis conditions solvothermal")
    if st.button("🔎 Cerca con Tavily"):
        if TAVILY_API_KEY:
            with st.spinner("Ricerca nei paper e nel web in corso..."):
                res = search_tavily_web(query_tavily)
                if res and res.get("answer"):
                    st.success("Risposta Generata dall'Agente:")
                    st.info(res["answer"])
                elif res and "results" in res:
                    st.write("### Risultati Principali:")
                    for r in res["results"]:
                        st.markdown(f"- **[{r.get('title')}]({r.get('url')})**: {r.get('content')}")
                else:
                    st.warning("Nessun risultato rilevante trovato.")
        else:
            st.error("🔑 Inserisci la tua Tavily API Key nella barra laterale a sinistra per abilitare la ricerca web.")
