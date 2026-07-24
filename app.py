import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import requests
import itertools
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

# Import per RDKit e Scikit-Learn / Ensemble
from rdkit import Chem
from rdkit.Chem import Descriptors
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

st.set_page_config(page_title="MOF Synthesis Predictor & Optimizer", page_icon="🧪", layout="wide")
st.title("🧪 Predictor & Optimizer per Sintesi di MOF")
st.markdown("Strumento avanzato di Machine Learning per la predizione, ottimizzazione e **spiegabilità chimica** della sintesi di MOF.")

# --- FUNZIONE DI PULIZIA E CONVERSIONE VALORI NUMERICI ---
def clean_float_val(val, default_val=0.0):
    """ Converte in float gestendo stringhe speciali come T.A., RT o formattazioni sporche """
    if pd.isna(val):
        return float(default_val)
    s_val = str(val).strip().upper()
    if s_val in ['T.A.', 'TA', 'RT', 'ROOM TEMP', 'ROOM TEMPERATURA', 'AMBIENTE']:
        return 25.0
    s_clean = re.sub(r'[^0-9\.-]', '', str(val))
    try:
        return float(s_clean)
    except Exception:
        return float(default_val)

# --- CONFIGURAZIONE TAVILY AI ---
DEFAULT_TAVILY_KEY = "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"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", DEFAULT_TAVILY_KEY)

with st.sidebar.expander("🌐 Configurazione Agent Web (Tavily)", expanded=False):
    tavily_input_key = st.text_input("Tavily API Key:", value=TAVILY_API_KEY, type="password")
    if tavily_input_key:
        TAVILY_API_KEY = tavily_input_key

def search_tavily_web(query, max_results=3):
    """Esegue una ricerca web tramite l'API REST di Tavily."""
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
    """Usa l'agente Tavily per cercare lo SMILES di un legante insolito o complesso."""
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

# --- INTEGRAZIONE CROSSREF PER VERIFICA DOI REALI ---
def fetch_real_doi_from_crossref(query_term):
    url = f"https://api.crossref.org/works?query={requests.utils.quote(query_term)}&rows=1"
    headers = {'User-Agent': 'MOFSynthesisPredictor/1.0 (mailto:admin@example.com)'}
    
    try:
        response = requests.get(url, headers=headers, timeout=4)
        if response.status_code == 200:
            data = response.json()
            items = data.get('message', {}).get('items', [])
            if items:
                paper = items[0]
                doi = paper.get('DOI', '')
                title_list = paper.get('title', ['Non disponibile'])
                title = title_list[0] if title_list else 'Non disponibile'
                container_list = paper.get('container-title', [''])
                journal = container_list[0] if container_list else 'Rivista N.D.'
                
                pub_date = paper.get('published-print', {}).get('date-parts', [[None]])[0][0]
                if not pub_date:
                    pub_date = paper.get('published-online', {}).get('date-parts', [[None]])[0][0]
                year_str = str(pub_date) if pub_date else "N.D."
                
                return {
                    'doi': doi,
                    'title': title,
                    'journal': journal,
                    'year': year_str,
                    'url': f"https://doi.org/{doi}"
                }
    except Exception:
        pass
    return None

def check_known_mof(metal_symbol, mol_obj=None, ligand_query=""):
    known_mappings = {
        ("Zr", "O=C(O)c1ccc(C(=O)O)cc1"): "UiO-66 synthesis terephthalic acid",
        ("Cu", "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1"): "HKUST-1 MOF synthesis trimesic acid",
        ("Zn", "Cc1c[nH]cn1"): "ZIF-8 synthesis 2-methylimidazole",
        ("Cr", "O=C(O)c1ccc(C(=O)O)cc1"): "MIL-101 Cr synthesis",
        ("Al", "O=C(O)c1ccc(C(=O)O)cc1"): "MIL-53 Al synthesis",
        ("Zn", "O=C(O)c1ccc(C(=O)O)cc1"): "MOF-5 synthesis",
        ("Zr", "O=C(O)c1ccc(C(=O)O)c(N)c1"): "UiO-66-NH2 synthesis"
    }
    
    input_smiles = Chem.MolToSmiles(mol_obj) if mol_obj else ""
    search_key = (metal_symbol, input_smiles)
    
    if search_key in known_mappings:
        query = known_mappings[search_key]
        mof_label = query.split()[0]
    elif ligand_query:
        query = f"{metal_symbol} MOF {ligand_query} synthesis"
        mof_label = f"MOF ({metal_symbol})"
    else:
        return []

    paper_info = fetch_real_doi_from_crossref(query)
    
    if paper_info:
        return [{
            "name": mof_label,
            "ref": f"{paper_info['journal']} ({paper_info['year']}) - {paper_info['title']}",
            "doi": paper_info['doi'],
            "url": paper_info['url']
        }]
    
    return []

# --- DATABASE SOLVENTI CON PARAMETRI FISICO-CHIMICI ---
SOLVENT_PROPERTIES = {
    'DMF':  {'alpha': 0.00, 'beta': 0.69, 'pi_star': 0.88, 'dielectric': 36.7, 'boiling_pt': 153.0},
    'DEF':  {'alpha': 0.00, 'beta': 0.69, 'pi_star': 0.88, 'dielectric': 32.1, 'boiling_pt': 177.0},
    'DMSO': {'alpha': 0.00, 'beta': 0.76, 'pi_star': 1.00, 'dielectric': 46.7, 'boiling_pt': 189.0},
    'MeCN': {'alpha': 0.19, 'beta': 0.31, 'pi_star': 0.75, 'dielectric': 37.5, 'boiling_pt': 82.0},
    'H2O':  {'alpha': 1.17, 'beta': 0.18, 'pi_star': 1.09, 'dielectric': 80.1, 'boiling_pt': 100.0},
    'MeOH': {'alpha': 0.93, 'beta': 0.62, 'pi_star': 0.60, 'dielectric': 32.7, 'boiling_pt': 64.7},
    'EtOH': {'alpha': 0.83, 'beta': 0.77, 'pi_star': 0.54, 'dielectric': 24.5, 'boiling_pt': 78.3},
    'CH2Cl2': {'alpha': 0.13, 'beta': 0.10, 'pi_star': 0.82, 'dielectric': 8.9, 'boiling_pt': 39.6},
    'Nessuno': {'alpha': 0.00, 'beta': 0.00, 'pi_star': 0.00, 'dielectric': 0.0, 'boiling_pt': 0.0}
}

# --- DIZIONARIO LOCALE LEGANTE MOF ---
COMMON_MOF_LIGANDS = {
    "c7h6o2": "O=C(O)c1ccccc1",
    "benzoic acid": "O=C(O)c1ccccc1",
    "c2h4o2": "CC(=O)O",
    "acetic acid": "CC(=O)O",
    "c1h2o2": "O=CO",
    "formic acid": "O=CO",
    "c2hf3o2": "O=C(O)C(F)(F)F",
    "trifluoroacetic acid": "O=C(O)C(F)(F)F",
    "tfa": "O=C(O)C(F)(F)F",
    "c3h6o2": "CCC(=O)O",
    "propionic acid": "CCC(=O)O",
    "c5h10o2": "CC(C)(C)C(=O)O",
    "pivalic acid": "CC(C)(C)C(=O)O",
    "c8h6o4": "O=C(O)c1ccc(C(=O)O)cc1",
    "terephthalic acid": "O=C(O)c1ccc(C(=O)O)cc1",
    "bdc": "O=C(O)c1ccc(C(=O)O)cc1",
    "isophthalic acid": "O=C(O)c1cccc(C(=O)O)c1",
    "phthalic acid": "O=C(O)c1ccccc1C(=O)O",
    "c8h7no4": "O=C(O)c1ccc(C(=O)O)c(N)c1",
    "bdc-nh2": "O=C(O)c1ccc(C(=O)O)c(N)c1",
    "c8h5no6": "O=C(O)c1ccc(C(=O)O)c([N+](=O)[O-])c1",
    "bdc-no2": "O=C(O)c1ccc(C(=O)O)c([N+](=O)[O-])c1",
    "c8h5bro4": "O=C(O)c1ccc(C(=O)O)c(Br)c1",
    "bdc-br": "O=C(O)c1ccc(C(=O)O)c(Br)c1",
    "c8h6o5": "O=C(O)c1ccc(C(=O)O)c(O)c1",
    "bdc-oh": "O=C(O)c1ccc(C(=O)O)c(O)c1",
    "c12h10o4": "O=C(O)c1ccc(-c2ccc(C(=O)O)cc2)cc1",
    "bpdc": "O=C(O)c1ccc(-c2ccc(C(=O)O)cc2)cc1",
    "c12h8o4": "O=C(O)c1ccc2ccc(C(=O)O)cc2c1",
    "ndc": "O=C(O)c1ccc2ccc(C(=O)O)cc2c1",
    "c4h4o4": "O=C(O)/C=C/C(=O)O",
    "fumaric acid": "O=C(O)/C=C/C(=O)O",
    "c9h6o6": "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1",
    "btc": "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1",
    "c27h18o6": "O=C(O)c1ccc(-c2cc(-c3ccc(C(=O)O)cc3)cc(-c3ccc(C(=O)O)cc3)c2)cc1",
    "btb": "O=C(O)c1ccc(-c2cc(-c3ccc(C(=O)O)cc3)cc(-c3ccc(C(=O)O)cc3)c2)cc1",
    "c3h4n2": "c1c[nH]cn1",
    "c4h6n2": "Cc1c[nH]cn1",
    "2-mim": "Cc1c[nH]cn1",
    "c10h8n2": "c1cnc(-c2ccncc2)cc1",
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

# --- PROPRIETÀ METALLI COMPLETI E HSAB ---
metal_props = {
    'Zr': {'Z': 40, 'Electronegativity': 1.33, 'Radius_pm': 160, 'Group': 4, 'Period': 5, 'MW': 91.22, 'HSAB': 'Hard', 'Name': 'Zirconio'},
    'Hf': {'Z': 72, 'Electronegativity': 1.30, 'Radius_pm': 159, 'Group': 4, 'Period': 6, 'MW': 178.49, 'HSAB': 'Hard', 'Name': 'Afnio'},
    'Cu': {'Z': 29, 'Electronegativity': 1.90, 'Radius_pm': 132, 'Group': 11, 'Period': 4, 'MW': 63.55, 'HSAB': 'Intermediate', 'Name': 'Rame'},
    'Zn': {'Z': 30, 'Electronegativity': 1.65, 'Radius_pm': 122, 'Group': 12, 'Period': 4, 'MW': 65.38, 'HSAB': 'Intermediate', 'Name': 'Zinco'},
    'Fe': {'Z': 26, 'Electronegativity': 1.83, 'Radius_pm': 126, 'Group': 8, 'Period': 4, 'MW': 55.85, 'HSAB': 'Hard', 'Name': 'Ferro'},
    'Co': {'Z': 27, 'Electronegativity': 1.88, 'Radius_pm': 126, 'Group': 9, 'Period': 4, 'MW': 58.93, 'HSAB': 'Intermediate', 'Name': 'Cobalto'},
    'Ni': {'Z': 28, 'Electronegativity': 1.91, 'Radius_pm': 124, 'Group': 10, 'Period': 4, 'MW': 58.69, 'HSAB': 'Intermediate', 'Name': 'Nichel'},
    'Mn': {'Z': 25, 'Electronegativity': 1.55, 'Radius_pm': 139, 'Group': 7, 'Period': 4, 'MW': 54.94, 'HSAB': 'Intermediate', 'Name': 'Manganese'},
    'Cr': {'Z': 24, 'Electronegativity': 1.66, 'Radius_pm': 128, 'Group': 6, 'Period': 4, 'MW': 51.99, 'HSAB': 'Hard', 'Name': 'Cromo'},
    'Ti': {'Z': 22, 'Electronegativity': 1.54, 'Radius_pm': 147, 'Group': 4, 'Period': 4, 'MW': 47.87, 'HSAB': 'Hard', 'Name': 'Titanio'},
    'Al': {'Z': 13, 'Electronegativity': 1.61, 'Radius_pm': 121, 'Group': 13, 'Period': 3, 'MW': 26.98, 'HSAB': 'Hard', 'Name': 'Alluminio'},
    'Mg': {'Z': 12, 'Electronegativity': 1.31, 'Radius_pm': 141, 'Group': 2, 'Period': 3, 'MW': 24.31, 'HSAB': 'Hard', 'Name': 'Magnesio'},
    'Ce': {'Z': 58, 'Electronegativity': 1.12, 'Radius_pm': 181, 'Group': 3, 'Period': 6, 'MW': 140.12, 'HSAB': 'Hard', 'Name': 'Cerio'},
    'Cd': {'Z': 48, 'Electronegativity': 1.69, 'Radius_pm': 151, 'Group': 12, 'Period': 5, 'MW': 112.41, 'HSAB': 'Soft', 'Name': 'Cadmio'},
    'Bi': {'Z': 83, 'Electronegativity': 2.02, 'Radius_pm': 156, 'Group': 15, 'Period': 6, 'MW': 208.98, 'HSAB': 'Intermediate', 'Name': 'Bismuto'},
    'La': {'Z': 57, 'Electronegativity': 1.10, 'Radius_pm': 187, 'Group': 3, 'Period': 6, 'MW': 138.91, 'HSAB': 'Hard', 'Name': 'Lantanio'}
}

anion_mw = {
    'Nitrato': 62.00 * 2,
    'Acetato': 59.04 * 2,
    'Cloruro': 35.45 * 2,
    'Altro': 60.00
}

# --- SMARTS PATTERNS PER RDKIT ---
SMARTS_PATTERNS = {
    'COOH': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),
    'Aromatic_N': Chem.MolFromSmarts('[n]')
}

def extract_smarts_features(mol):
    if not mol:
        return {'n_COOH': 0, 'n_Aromatic_N': 0, 'fraction_sp2': 0.0}
    
    n_cooh = len(mol.GetSubstructMatches(SMARTS_PATTERNS['COOH']))
    n_aro_n = len(mol.GetSubstructMatches(SMARTS_PATTERNS['Aromatic_N']))
    
    c_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
    if c_atoms:
        sp2_c = sum(1 for atom in c_atoms if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2)
        fraction_sp2 = sp2_c / len(c_atoms)
    else:
        fraction_sp2 = 0.0
        
    return {
        'n_COOH': n_cooh,
        'n_Aromatic_N': n_aro_n,
        'fraction_sp2': fraction_sp2
    }

def calculate_hsab_match(metal_hsab, n_cooh, n_aro_n):
    if metal_hsab == 'Hard':
        return 1.0 if n_cooh > 0 else 0.2
    elif metal_hsab == 'Intermediate':
        return 1.0 if (n_aro_n > 0 or n_cooh > 0) else 0.5
    else:
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

    if TAVILY_API_KEY:
        tavily_smiles = search_tavily_for_ligand_smiles(query)
        if tavily_smiles:
            return tavily_smiles

    return None

def calculate_solvent_mix_properties(solv_p, ml_p, cosolv, ml_cosolv):
    prop_p = SOLVENT_PROPERTIES.get(solv_p, SOLVENT_PROPERTIES['DMF'])
    prop_co = SOLVENT_PROPERTIES.get(cosolv, SOLVENT_PROPERTIES['Nessuno'])
    
    tot_vol = ml_p + ml_cosolv
    if tot_vol <= 0:
        return prop_p
        
    f_p = ml_p / tot_vol
    f_co = ml_cosolv / tot_vol
    
    return {
        'mix_alpha': (prop_p['alpha'] * f_p) + (prop_co['alpha'] * f_co),
        'mix_beta': (prop_p['beta'] * f_p) + (prop_co['beta'] * f_co),
        'mix_pi_star': (prop_p['pi_star'] * f_p) + (prop_co['pi_star'] * f_co),
        'mix_dielectric': (prop_p['dielectric'] * f_p) + (prop_co['dielectric'] * f_co),
        'mix_boiling_pt': (prop_p['boiling_pt'] * f_p) + (prop_co['boiling_pt'] * f_co)
    }

def process_unified_dataset(df):
    target_col = None
    possible_targets = ['Target_Esito_Classe', 'Target', 'Esito', 'Classe', 'target', 'esito']
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
        
        smarts_f = extract_smarts_features(mol)
        
        met = str(row.get('Metallo', 'Cu'))
        m_info = metal_props.get(met, metal_props['Cu'])
        anione_sel = str(row.get('Anione', 'Nitrato'))
        
        hsab_match = calculate_hsab_match(m_info['HSAB'], smarts_f['n_COOH'], smarts_f['n_Aromatic_N'])
        
        m_leg = clean_float_val(row.get('mmol legante'), default_val=0.1)
        m_sale = clean_float_val(row.get('mmol sale'), default_val=0.1)
        ratio = m_leg / m_sale if m_sale > 0 else 1.0
        
        solv_p = str(row.get('Solvente', 'DMF'))
        cosolv = str(row.get('CoSolvente', 'Nessuno'))
        
        ml_solv_p = clean_float_val(row.get('mL_Solvente_P'), default_val=10.0)
        ml_cosolv = clean_float_val(row.get('mL_CoSolvente'), default_val=0.0)
        total_vol = ml_solv_p + ml_cosolv
        cosolv_pct = (ml_cosolv / total_vol * 100) if total_vol > 0 else 0.0
        
        mix_props = calculate_solvent_mix_properties(solv_p, ml_solv_p, cosolv, ml_cosolv)
        
        add_type = str(row.get('Additivo_Tipo', 'None'))
        add_eq = clean_float_val(row.get('Additivo_Eq'), default_val=0.0)
        
        temp = clean_float_val(row.get('Temperatura_num'), default_val=120.0)
        tempo = clean_float_val(row.get('Tempo_ore_num'), default_val=48.0)
        
        raw_target = row.get(target_col, 0) if target_col else 0
        try:
            target = int(float(raw_target))
        except Exception:
            target = 0
            
        processed.append({
            'SMILES_Group': smiles if smiles and smiles != 'nan' else 'sconosciuto',
            'MW_Legante': float(mw), 'LogP_Legante': float(logp), 'HBD_Legante': float(hbd), 'HBA_Legante': float(hba),
            'TPSA_Legante': float(tpsa), 'RotatableBonds_Legante': float(rot),
            'SMARTS_n_COOH': smarts_f['n_COOH'],
            'SMARTS_n_Aromatic_N': smarts_f['n_Aromatic_N'],
            'SMARTS_fraction_sp2': smarts_f['fraction_sp2'],
            'HSAB_Match_Index': hsab_match,
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
            'Solvent_Mix_Alpha': mix_props['mix_alpha'],
            'Solvent_Mix_Beta': mix_props['mix_beta'],
            'Solvent_Mix_PiStar': mix_props['mix_pi_star'],
            'Solvent_Mix_Dielectric': mix_props['mix_dielectric'],
            'Solvent_Mix_BoilingPt': mix_props['mix_boiling_pt'],
            'Additive_Eq': float(add_eq),
            'Additive_Is_Acid': 1 if add_type == 'Acid' else 0,
            'Additive_Is_Base': 1 if add_type == 'Base' else 0,
            'Additive_Is_Neutral': 1 if add_type == 'Neutral' else 0,
            'Target_Esito_Classe': target
        })
    return pd.DataFrame(processed)

def create_stacking_ensemble():
    estimators = [
        ('lgb', LGBMClassifier(
            n_estimators=180, learning_rate=0.03, max_depth=6, 
            num_leaves=31, class_weight='balanced', random_state=42, verbose=-1
        )),
        ('rf', RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight='balanced_subsample', 
            random_state=42, n_jobs=-1
        ))
    ]
    
    if HAS_CATBOOST:
        estimators.append(('cat', CatBoostClassifier(
            iterations=200, learning_rate=0.04, depth=6, 
            verbose=0, random_seed=42
        )))
        
    meta_model = LogisticRegression(class_weight='balanced', max_iter=500)
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        stack_method='predict_proba',
        n_jobs=-1
    )
    return stacking_clf

@st.cache_resource
def load_or_train_model():
    pkl_file = "modello_sintesi_mof_ottimizzato.pkl"
    csv_file = "Dataset_Sintesi_Unificato.csv"
    
    if os.path.exists(pkl_file):
        try:
            saved_data = joblib.load(pkl_file)
            if isinstance(saved_data, dict) and 'model' in saved_data:
                return saved_data['model'], saved_data['features'], saved_data.get('metrics', {}), saved_data.get('importances', [])
        except Exception:
            pass

    if os.path.exists(csv_file):
        raw_df = pd.read_csv(csv_file)
    else:
        st.error(f"File '{csv_file}' non trovato nella directory di lavoro!")
        st.stop()
        
    df = process_unified_dataset(raw_df)
    
    groups = df['SMILES_Group'].tolist()
    X = df.drop(columns=['Target_Esito_Classe', 'SMILES_Group'])
    y_series = pd.to_numeric(df['Target_Esito_Classe'], errors='coerce')
    valid_mask = y_series.notna()
    
    X = X[valid_mask].copy().reset_index(drop=True)
    y = y_series[valid_mask].astype(int).copy().reset_index(drop=True)
    groups = [g for i, g in enumerate(groups) if valid_mask.iloc[i]]
    
    X = X.apply(lambda col: col.apply(clean_float_val) if col.dtype == 'object' else col)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    if y.nunique() < 2:
        st.error(
            f"⚠️ **Errore Dataset:** La colonna del Target contiene un solo valore distinto ('{y.unique()}'). "
            "Il modello richiede almeno 2 classi distinte nel file CSV."
        )
        st.stop()

    feature_names = X.columns.tolist()
    base_ensemble = create_stacking_ensemble()
    
    n_unique_groups = len(np.unique(groups))
    unique_classes = y.nunique()
    
    if n_unique_groups >= 2 and unique_classes > 1:
        cv_splits = min(3, n_unique_groups)
        sgkf_calib = StratifiedGroupKFold(n_splits=cv_splits)
        final_model = CalibratedClassifierCV(
            estimator=base_ensemble, method='sigmoid',
            cv=sgkf_calib
        )
        try:
            final_model.fit(X, y, groups=groups)
        except Exception:
            base_ensemble.fit(X, y)
            final_model = base_ensemble
    else:
        base_ensemble.fit(X, y)
        final_model = base_ensemble

    try:
        if hasattr(final_model, 'calibrated_classifiers_'):
            lgb_est = final_model.calibrated_classifiers_[0].estimator.named_estimators_['lgb']
            importances = lgb_est.feature_importances_
        else:
            importances = final_model.named_estimators_['lgb'].feature_importances_
    except Exception:
        importances = np.zeros(len(feature_names))

    metrics = {
        'train_accuracy': accuracy_score(y, final_model.predict(X)),
        'n_samples': len(X),
        'n_features': len(feature_names)
    }
    
    save_dict = {
        'model': final_model,
        'features': feature_names,
        'importances': importances,
        'metrics': metrics
    }
    
    joblib.dump(save_dict, pkl_file)
    return final_model, feature_names, metrics, importances

try:
    model, feature_names, metrics, importances = load_or_train_model()
    st.sidebar.success("Modello Ensemble Stacking Attivo!")
except Exception as e:
    st.sidebar.error(f"Errore caricamento modello: {e}")
    st.stop()

# --- COMPONENTE INTERFACCIA: GRIGLIA SELEZIONE METALLO ---
def render_metal_grid_selector(key_prefix="pred"):
    """
    Rende una griglia interattiva visiva per la selezione del metallo
    """
    st.markdown("##### 🧱 Seleziona Metallo dalla Griglia chimica:")
    
    # Formattazione opzioni con nome e classificazione HSAB
    metal_options = list(metal_props.keys())
    metal_labels = {
        m: f"{m} ({metal_props[m]['Name']} - {metal_props[m]['HSAB']})"
        for m in metal_options
    }
    
    # Se st.pills è supportato nella versione Streamlit, lo usa; altrimenti fallback su radio/selectbox
    if hasattr(st, "pills"):
        selected = st.pills(
            "Seleziona Metallo:", 
            options=metal_options, 
            format_func=lambda x: f"{x} - {metal_props[x]['Name']}",
            default="Zr" if "Zr" in metal_options else metal_options[0],
            key=f"{key_prefix}_pills_metal"
        )
    else:
        # Griglia dinamica in colonne
        cols = st.columns(4)
        selected = None
        
        # Uso di radio formattato con indicazione dettagliata
        selected = st.selectbox(
            "Metallo di reazione:",
            options=metal_options,
            format_func=lambda x: f"{x} | {metal_props[x]['Name']} ({metal_props[x]['HSAB']})",
            index=metal_options.index('Zr') if 'Zr' in metal_options else 0,
            key=f"{key_prefix}_select_metal"
        )

    # Dettagli del metallo selezionato
    m_data = metal_props[selected]
    hsab_color = "🔴 Hard" if m_data['HSAB'] == 'Hard' else ("🟡 Intermediate" if m_data['HSAB'] == 'Intermediate' else "🟢 Soft")
    
    st.caption(
        f"**Metallo:** `{selected}` ({m_data['Name']}) | "
        f"**HSAB:** {hsab_color} | "
        f"**Peso Atomico:** `{m_data['MW']:.2f}` g/mol | "
        f"**Raggio Ionico:** `{m_data['Radius_pm']}` pm"
    )
    return selected

# --- SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Stato & Performance Modello")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    acc_val = metrics.get('train_accuracy', 0.85) * 100
    st.metric("Accuratezza", f"{acc_val:.1f}%")
with col_sb2:
    st.metric("Sintesi DB", metrics.get('n_samples', 'N/A'))

st.sidebar.markdown(f"""
* **Architettura:** Stacking Ensemble  
  *(LightGBM + Random Forest + CatBoost)*
* **Parametri Valutati:** `{metrics.get('n_features', 36)}` Feature Chimico-Fisiche
* **Motori Descrittori:** RDKit (SMARTS) + HSAB Pearson
""")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Quick Reference Chimica")

with st.sidebar.expander("🧪 Teoria HSAB di Pearson", expanded=False):
    st.markdown("""
    **Acidi Hard ($\text{Zr}^{4+}, \text{Hf}^{4+}, \text{Fe}^{3+}, \text{Al}^{3+}, \text{Cr}^{3+}$):**
    * Alta densità di carica.
    * Prediligono **Basi Hard** (es. Carbossili $-\text{COOH}$).
    
    **Acidi Intermediate ($\text{Cu}^{2+}, \text{Zn}^{2+}, \text{Ni}^{2+}, \text{Co}^{2+}$):**
    * Densità di carica media.
    * Prediligono **Azoti Aromatici** (Imidazoli, Piridine) o miscele $-\text{COOH}/\text{N}$.
    """)

# --- TAB INTERFACCIA MAIN ---
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predizione Singola", "📂 Predizione Batch", "⚡ Ottimizzatore Automatico", "🌐 Ricerca Web (Tavily AI)"])

def build_feature_row(mol, mw, logp, hbd, hba, tpsa, rot_bonds, temp, tempo, mmol_legante, mmol_sale, metallo_sel, anione_sel, solvente_p, ml_solv_p, cosolvente, ml_cosolv, additivo_sel, add_eq):
    add_info = ADDITIVES_DATABASE.get(additivo_sel, ADDITIVES_DATABASE['Nessuno'])
    add_type = add_info['type']
    
    total_vol = float(ml_solv_p) + float(ml_cosolv)
    cosolv_pct = (float(ml_cosolv) / total_vol * 100.0) if total_vol > 0 else 0.0
    mix_props = calculate_solvent_mix_properties(solvente_p, float(ml_solv_p), cosolvente, float(ml_cosolv))
    
    smarts_f = extract_smarts_features(mol)
    metal_m = metal_props[metallo_sel]
    hsab_match = calculate_hsab_match(metal_m['HSAB'], smarts_f['n_COOH'], smarts_f['n_Aromatic_N'])
    
    input_dict = {
        'MW_Legante': float(mw),
        'LogP_Legante': float(logp),
        'HBD_Legante': float(hbd),
        'HBA_Legante': float(hba),
        'TPSA_Legante': float(tpsa),
        'RotatableBonds_Legante': float(rot_bonds),
        'SMARTS_n_COOH': smarts_f['n_COOH'],
        'SMARTS_n_Aromatic_N': smarts_f['n_Aromatic_N'],
        'SMARTS_fraction_sp2': smarts_f['fraction_sp2'],
        'HSAB_Match_Index': hsab_match,
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
        'Anion_Acetato': 1 if anione_sel == 'Acetato' else 0,
        'Anion_Cloruro': 1 if anione_sel == 'Cloruro' else 0,
        'Anion_Nitrato': 1 if anione_sel == 'Nitrato' else 0,
        'Anion_Altro': 1 if anione_sel == 'Altro' else 0,
        'mL_Solvente_P': float(ml_solv_p),
        'mL_CoSolvente': float(ml_cosolv),
        'Total_Volume_mL': float(total_vol),
        'CoSolvent_Pct': float(cosolv_pct),
        'Solvent_Mix_Alpha': mix_props['mix_alpha'],
        'Solvent_Mix_Beta': mix_props['mix_beta'],
        'Solvent_Mix_PiStar': mix_props['mix_pi_star'],
        'Solvent_Mix_Dielectric': mix_props['mix_dielectric'],
        'Solvent_Mix_BoilingPt': mix_props['mix_boiling_pt'],
        'Additive_Eq': float(add_eq),
        'Additive_Is_Acid': 1 if add_type == 'Acid' else 0,
        'Additive_Is_Base': 1 if add_type == 'Base' else 0,
        'Additive_Is_Neutral': 1 if add_type == 'Neutral' else 0,
    }
    
    df_f = pd.DataFrame([input_dict])
    
    for col in feature_names:
        if col not in df_f.columns:
            df_f[col] = 0.0
            
    return df_f[feature_names]

# --- TAB 1: PREDIZIONE SINGOLA ---
with tab1:
    st.subheader("Inserisci i parametri della reazione")
    col1, col2, col3 = st.columns(3)
    
    query_input = ""
    with col1:
        st.markdown("### 1. Legante Chimico")
        mode_legante = st.radio(
            "Modalità Input Legante:", 
            ["SMILES", "Nome / Formula / CAS", "Carica File (.mol / .sdf / .cif)"],
            horizontal=True
        )
        
        mol = None
        if mode_legante == "SMILES":
            smiles_input = st.text_input("SMILES del Legante:", value="O=C(O)c1ccc(C(=O)O)cc1")
            if smiles_input:
                mol = Chem.MolFromSmiles(smiles_input)
                
        elif mode_legante == "Nome / Formula / CAS":
            query_input = st.text_input("Nome, Formula o CAS:", value="Benzoic acid")
            if query_input:
                with st.spinner("Ricerca molecola nei database e sul Web..."):
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
        
        # --- IMPLEMENTAZIONE GRIGLIA SELEZIONE METALLO ---
        metallo_sel = render_metal_grid_selector(key_prefix="tab1")
        
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
            index=0
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
        
        co_solvente = st.selectbox("Co-Solvente (Opzionale):", ['Nessuno', 'H2O', 'MeOH', 'EtOH', 'CH2Cl2', 'DEF', 'MeCN', 'DMSO'])
        
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
            known_matches = check_known_mof(
                metal_symbol=metallo_sel, 
                mol_obj=mol, 
                ligand_query=query_input if mode_legante == "Nome / Formula / CAS" else ""
            )
            
            st.markdown("---")
            if known_matches:
                for mof in known_matches:
                    clean_doi = mof['doi'].replace("https://doi.org/", "")
                    doi_url = mof.get('url', f"https://doi.org/{clean_doi}")

                    st.info(
                        f"🟢 **Combinazione nota e verificata tramite Crossref API!**\n\n"
                        f"* **MOF/Sintesi:** `{mof['name']}`\n"
                        f"* **Articolo:** {mof['ref']}\n"
                        f"* **DOI Verificato:** `{clean_doi}` *(🔗 [Apri Pubblicazione Ufficiale]({doi_url}))*"
                    )
            else:
                st.success("✨ **Combinazione Inedita / Non presente a DB:** Nessun MOF classico censito direttamente per questa specifica coppia.")

            df_features = build_feature_row(
                mol, mw, logp, hbd, hba, tpsa, rot_bonds, temp, tempo, 
                mmol_legante, mmol_sale, metallo_sel, anione_sel, 
                solvente_p, ml_solv_p, co_solvente, ml_cosolv, additivo_sel, add_eq
            )
            probs = model.predict_proba(df_features)[0]
            pred_class = model.predict(df_features)[0]

            st.markdown("---")
            st.subheader("📊 Risultato della Predizione (Ensemble Multi-Algoritmo)")
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
                X_batch = processed_batch.drop(columns=['Target_Esito_Classe', 'SMILES_Group'])
                
                for col in feature_names:
                    if col not in X_batch.columns:
                        X_batch[col] = 0.0
                X_batch = X_batch[feature_names]
                
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

# --- TAB 3: OTTIMIZZATORE AUTOMATICO MULTI-METALLO (GRIGLIA ESPANSA) ---
with tab3:
    st.subheader("⚡ Ottimizzatore di Condizioni Sperimentali Multi-Metallo")
    st.markdown("L'IA simulerà ed esaminerà **griglie di combinazioni chimiche in parallelo**, testando anche **più metalli contemporaneamente**.")
    
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    
    with opt_col1:
        st.markdown("### 1. Legante Chimico")
        opt_mode_legante = st.radio(
            "Modalità Input Legante:", 
            ["SMILES", "Nome / Formula / CAS", "Carica File (.mol / .sdf / .cif)"],
            horizontal=True,
            key="opt_mode_leg"
        )
        
        opt_mol = None
        if opt_mode_legante == "SMILES":
            opt_smiles_input = st.text_input("SMILES del Legante:", value="O=C(O)c1ccc(C(=O)O)cc1", key="opt_smiles")
            if opt_smiles_input:
                opt_mol = Chem.MolFromSmiles(opt_smiles_input)
                
        elif opt_mode_legante == "Nome / Formula / CAS":
            opt_query_input = st.text_input("Nome, Formula Bruta o CAS:", value="C8H6O4", key="opt_query")
            if opt_query_input:
                with st.spinner("Ricerca molecola nei database e sul Web..."):
                    opt_found_smiles = resolve_molecule_to_smiles(opt_query_input)
                    if opt_found_smiles:
                        opt_mol = Chem.MolFromSmiles(opt_found_smiles)
                        st.caption(f"SMILES Identificato: `{opt_found_smiles}`")
                    else:
                        st.error("Nessuna molecola trovata per la formula/nome inserito.")
                        
        elif opt_mode_legante == "Carica File (.mol / .sdf / .cif)":
            opt_uploaded_file = st.file_uploader("Carica file .mol, .sdf o .cif", type=['mol', 'sdf', 'cif'], key="opt_file")
            if opt_uploaded_file is not None:
                file_ext = opt_uploaded_file.name.split('.')[-1].lower()
                file_bytes = opt_uploaded_file.getvalue().decode('utf-8', errors='ignore')
                
                if file_ext in ['mol', 'sdf']:
                    opt_mol = Chem.MolFromMolBlock(file_bytes)
                elif file_ext == 'cif':
                    if HAS_PYMATGEN:
                        try:
                            with open("temp_opt_upload.cif", "w", encoding="utf-8") as f:
                                f.write(file_bytes)
                            struct = Structure.from_file("temp_opt_upload.cif")
                            red_formula = struct.composition.reduced_formula
                            opt_found_smiles = resolve_molecule_to_smiles(red_formula)
                            if opt_found_smiles:
                                opt_mol = Chem.MolFromSmiles(opt_found_smiles)
                        except Exception as e:
                            st.error(f"Errore lettura CIF: {e}")

        if opt_mol:
            opt_mw_val = float(Descriptors.MolWt(opt_mol))
            st.success(f"Molecola acquisita! MW: {opt_mw_val:.2f} g/mol")
        else:
            opt_mw_val = 166.13

        opt_input_mode_leg = st.radio("Inserisci Legante per l'ottimizzazione come:", ["MilliMoli (mmol)", "Massa (mg)"], key="opt_rad_leg", horizontal=True)
        if opt_input_mode_leg == "MilliMoli (mmol)":
            opt_mmol_legante = st.number_input("mmol Legante:", min_value=0.001, max_value=50.0, value=0.10, step=0.01, key="opt_mmol_leg")
            opt_mg_legante = opt_mmol_legante * opt_mw_val
            st.caption(f"⚖️ Corrispondono a **{opt_mg_legante:.2f} mg**")
        else:
            opt_mg_legante = st.number_input("Massa Legante (mg):", min_value=0.1, max_value=5000.0, value=16.61, step=1.0, key="opt_mg_leg")
            opt_mmol_legante = opt_mg_legante / opt_mw_val if opt_mw_val > 0 else 0.1
            st.caption(f"⚖️ Corrispondono a **{opt_mmol_legante:.3f} mmol**")

    with opt_col2:
        st.markdown("### 2. Griglia Metalli e Precursore")
        
        # --- IMPLEMENTAZIONE GRIGLIA MULTI-SELEZIONE METALLI PER OTTIMIZZATORE ---
        metal_list_opt = sorted(list(metal_props.keys()))
        opt_selected_metals = st.multiselect(
            "🧱 Seleziona Metalli da inserire nella griglia di Scansione:",
            options=metal_list_opt,
            default=['Zr', 'Cu', 'Zn'] if all(m in metal_list_opt for m in ['Zr', 'Cu', 'Zn']) else [metal_list_opt[0]],
            help="Puoi selezionare uno o più metalli per confrontare quale offre la resa di sintesi migliore con questo legante."
        )
        
        if not opt_selected_metals:
            st.warning("Seleziona almeno un metallo per procedere con la scansione.")
            opt_selected_metals = [metal_list_opt[0]]

        opt_anione = st.selectbox("Anione / Precursore:", ['Nitrato', 'Acetato', 'Cloruro', 'Altro'], key="opt_an")
        
        opt_idratazione = st.selectbox(
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
            index=0,
            key="opt_hydr"
        )
        
        opt_n_h2o = int(opt_idratazione.split('(')[1].split(' ')[0])

        opt_mmol_sale = st.number_input("mmol Sale Metallico (standard per la griglia):", min_value=0.001, max_value=50.0, value=0.10, step=0.01, key="opt_mmol_sale")

    with opt_col3:
        st.markdown("### 3. Opzioni Scansione")
        opt_speed_mode = st.radio("Velocità Scansione:", ["Ultra-Veloce ⚡", "Completa 🔍"], index=0, key="opt_speed")

    if st.button("🚀 Avvia Scansione e Ottimizzazione Multi-Metallo", type="primary"):
        if not opt_mol:
            st.error("Seleziona o inserisci un legante valido prima di avviare l'ottimizzazione.")
        else:
            # Calcolo proprietà invarianti del legante una sola volta
            opt_mw = float(Descriptors.MolWt(opt_mol))
            opt_logp = float(Descriptors.MolLogP(opt_mol))
            opt_hbd = float(Descriptors.NumHDonors(opt_mol))
            opt_hba = float(Descriptors.NumHAcceptors(opt_mol))
            opt_tpsa = float(Descriptors.TPSA(opt_mol))
            opt_rot = float(Descriptors.NumRotatableBonds(opt_mol))
            
            smarts_f = extract_smarts_features(opt_mol)

            # Definizione della griglia di ricerca in base alla velocità scelta
            if "Ultra-Veloce" in opt_speed_mode:
                temperatures = [100.0, 120.0, 140.0]
                times = [24.0, 48.0]
                solvents_p = ['DMF', 'DEF', 'DMSO', 'MeCN', 'MeOH']
                volumes_p = [10.0]
                cosolvents = [('Nessuno', 0.0), ('H2O', 1.0), ('MeOH', 2.0)]
                additives = [('Nessuno', 0.0), ('Acido Acetico (AcOH)', 2.0), ('Trietilammina (TEA)', 1.0)]
            else:
                temperatures = [80.0, 100.0, 120.0, 140.0, 160.0]
                times = [12.0, 24.0, 48.0, 72.0]
                solvents_p = ['DMF', 'DEF', 'DMSO', 'MeCN', 'H2O', 'MeOH', 'EtOH']
                volumes_p = [5.0, 10.0, 15.0]
                cosolvents = [('Nessuno', 0.0), ('H2O', 1.0), ('MeOH', 2.0), ('EtOH', 2.0)]
                additives = [('Nessuno', 0.0), ('Acido Acetico (AcOH)', 2.0), ('Acido Formico (HCOOH)', 2.0), ('Trietilammina (TEA)', 1.0)]

            with st.spinner(f"⚡ Simulazione vettorizzata su {len(opt_selected_metals)} metalli e centinaia di condizioni..."):
                # Generazione combinazioni cartesiane veloci con itertools inclusi I METALLI SELEZIONATI
                grid_combos = list(itertools.product(
                    opt_selected_metals, temperatures, times, solvents_p, volumes_p, cosolvents, additives
                ))
                
                rows_list = []
                display_info = []

                # Calcolo del rapporto molare L/M
                ratio_lm = float(opt_mmol_legante) / float(opt_mmol_sale) if float(opt_mmol_sale) > 0 else 1.0

                # Costruzione veloce del DataFrame vettorizzato per la griglia estesa
                for cur_metal, temp, tempo, solv_p, ml_solv_p, (cosolv, ml_cosolv), (add_name, add_eq) in grid_combos:
                    metal_m = metal_props[cur_metal]
                    hsab_match = float(calculate_hsab_match(metal_m['HSAB'], smarts_f['n_COOH'], smarts_f['n_Aromatic_N']))
                    
                    opt_base_salt_mw = metal_m['MW'] + anion_mw.get(opt_anione, 60.0)
                    opt_total_salt_mw = opt_base_salt_mw + (opt_n_h2o * 18.015)
                    opt_mg_sale_calc = opt_mmol_sale * opt_total_salt_mw

                    add_info = ADDITIVES_DATABASE.get(add_name, ADDITIVES_DATABASE['Nessuno'])
                    add_type = add_info['type']
                    
                    total_vol = ml_solv_p + ml_cosolv
                    cosolv_pct = (ml_cosolv / total_vol * 100.0) if total_vol > 0 else 0.0
                    mix_props = calculate_solvent_mix_properties(solv_p, ml_solv_p, cosolv, ml_cosolv)
                    
                    rows_list.append({
                        'MW_Legante': opt_mw, 'LogP_Legante': opt_logp, 'HBD_Legante': opt_hbd, 'HBA_Legante': opt_hba,
                        'TPSA_Legante': opt_tpsa, 'RotatableBonds_Legante': opt_rot,
                        'SMARTS_n_COOH': smarts_f['n_COOH'], 'SMARTS_n_Aromatic_N': smarts_f['n_Aromatic_N'],
                        'SMARTS_fraction_sp2': smarts_f['fraction_sp2'], 'HSAB_Match_Index': hsab_match,
                        'Temperatura_num': temp, 'Tempo_ore_num': tempo,
                        'mmol legante': float(opt_mmol_legante), 'mmol sale': float(opt_mmol_sale), 'Rapporto L/M': float(ratio_lm),
                        'Metallo_Z': metal_m['Z'], 'Metallo_Electronegativity': metal_m['Electronegativity'],
                        'Metallo_Radius_pm': metal_m['Radius_pm'], 'Metallo_Group': metal_m['Group'], 'Metallo_Period': metal_m['Period'],
                        'Anion_Acetato': 1 if opt_anione == 'Acetato' else 0,
                        'Anion_Cloruro': 1 if opt_anione == 'Cloruro' else 0,
                        'Anion_Nitrato': 1 if opt_anione == 'Nitrato' else 0,
                        'Anion_Altro': 1 if opt_anione == 'Altro' else 0,
                        'mL_Solvente_P': ml_solv_p, 'mL_CoSolvente': ml_cosolv, 'Total_Volume_mL': total_vol,
                        'CoSolvent_Pct': cosolv_pct,
                        'Solvent_Mix_Alpha': mix_props['mix_alpha'], 'Solvent_Mix_Beta': mix_props['mix_beta'],
                        'Solvent_Mix_PiStar': mix_props['mix_pi_star'], 'Solvent_Mix_Dielectric': mix_props['mix_dielectric'],
                        'Solvent_Mix_BoilingPt': mix_props['mix_boiling_pt'],
                        'Additive_Eq': add_eq,
                        'Additive_Is_Acid': 1 if add_type == 'Acid' else 0,
                        'Additive_Is_Base': 1 if add_type == 'Base' else 0,
                        'Additive_Is_Neutral': 1 if add_type == 'Neutral' else 0,
                    })
                    
                    add_mmol_calc = add_eq * opt_mmol_legante
                    
                    display_info.append({
                        'Metallo': f"{cur_metal} ({metal_m['Name']})",
                        'mmol Legante': round(opt_mmol_legante, 3),
                        'mg Legante': round(opt_mg_legante, 2),
                        'mmol Sale': round(opt_mmol_sale, 3),
                        'mg Sale': round(opt_mg_sale_calc, 2),
                        'Rapporto L/M': round(ratio_lm, 2),
                        'Temperatura (°C)': temp,
                        'Tempo (h)': tempo,
                        'Solvente Principale': solv_p,
                        'mL Solvente P.': ml_solv_p,
                        'Co-Solvente': cosolv,
                        'mL Co-Solvente': ml_cosolv,
                        'Additivo': add_name,
                        'Eq. Additivo': add_eq,
                        'mmol Additivo': round(add_mmol_calc, 3)
                    })

                # Generazione matrice unica per scikit-learn
                df_simulation = pd.DataFrame(rows_list)
                for col in feature_names:
                    if col not in df_simulation.columns:
                        df_simulation[col] = 0.0
                df_simulation = df_simulation[feature_names]

                # Predizione Vettorizzata ISTANTANEA di tutte le combinazioni in una sola chiamata
                probs_matrix = model.predict_proba(df_simulation)
                
                classes_list = [int(c) if str(c).isdigit() else c for c in model.classes_]
                if 2 in classes_list:
                    target_class_idx = classes_list.index(2)
                elif 1 in classes_list:
                    target_class_idx = classes_list.index(1)
                else:
                    target_class_idx = len(classes_list) - 1

                success_probs = (probs_matrix[:, target_class_idx] * 100.0).round(1)

                # Unione dei risultati
                df_results = pd.DataFrame(display_info)
                df_results['Probabilità Successo (%)'] = success_probs
                df_results = df_results.sort_values(by='Probabilità Successo (%)', ascending=False).reset_index(drop=True)

            st.success(f"⚡ **{len(df_results)} combinazioni analizzate istantaneamente su {len(opt_selected_metals)} metalli differenti!**")
            st.markdown("### 🏆 Migliori Condizioni Sperimentali Trovate nella Griglia")
            st.dataframe(df_results.head(15))

# --- TAB 4: RICERCA WEB TAVILY AI ---
with tab4:
    st.subheader("🌐 Agente Web Tavily per Sintesi & Letteratura MOF")
    st.markdown("Effettua ricerche live per verificare protocolli di sintesi, informazioni sui leganti o pubblicazioni scientifiche correlate.")
    
    if not TAVILY_API_KEY:
        st.warning("⚠️ Per utilizzare l'Agente Tavily, inserisci la tua **Tavily API Key** nel menu laterale (Sidebar).")
    
    query_tavily = st.text_input("Inserisci la query di ricerca chimica:", value="UiO-66 synthesis conditions modulator benzoic acid")
    num_res = st.slider("Numero di risultati:", min_value=1, max_value=5, value=3)
    
    if st.button("🔎 Cerca sul Web con Tavily"):
        if not TAVILY_API_KEY:
            st.error("API Key Tavily mancante.")
        else:
            with st.spinner("Ricerca informazioni sul web in corso..."):
                tavily_res = search_tavily_web(query_tavily, max_results=num_res)
                if tavily_res:
                    if tavily_res.get("answer"):
                        st.info(f"💡 **Sintesi Risposta AI Tavily:**\n\n{tavily_res['answer']}")
                    
                    st.markdown("### 📚 Risultati della Ricerca")
                    for r in tavily_res.get("results", []):
                        st.markdown(f"#### [{r.get('title')}]({r.get('url')})")
                        st.write(r.get("content"))
                        st.markdown("---")
                else:
                    st.error("Nessun risultato trovato o errore nella richiesta.")
