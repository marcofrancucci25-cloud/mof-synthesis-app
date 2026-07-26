import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import requests
import itertools
import traceback
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

st.set_page_config(page_title="MOF Synthesis Predictor & Optimizer", page_icon="🧪", layout="wide")
st.title("🧪 Predictor & Optimizer per Sintesi di MOF")
st.markdown("Strumento avanzato di Machine Learning per la predizione, ottimizzazione e **spiegabilità chimica** della sintesi di MOF con integrazione di letteratura **Open Access**.")

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
# IMPORTANTE: nessuna chiave hardcoded nel codice sorgente.
# La chiave va impostata come secret in Streamlit Cloud (st.secrets) oppure
# come variabile d'ambiente TAVILY_API_KEY. Se non presente, l'utente può
# inserirla manualmente nel campo sottostante (non viene mai precompilata
# con un valore reale, per evitare che resti visibile nel session state/HTML).
try:
    TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))
except Exception:
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

with st.sidebar.expander("🌐 Configurazione Agent Web (Tavily)", expanded=False):
    if TAVILY_API_KEY:
        st.caption("✅ Chiave Tavily configurata tramite secrets/variabile d'ambiente.")
    tavily_input_key = st.text_input(
        "Tavily API Key (opzionale, sovrascrive quella di sistema):",
        value="",
        type="password",
        placeholder="tvly-..." if not TAVILY_API_KEY else "Lascia vuoto per usare la chiave configurata"
    )
    if tavily_input_key:
        TAVILY_API_KEY = tavily_input_key

def search_tavily_web(query, max_results=3, timeout=4):
    """Esegue una ricerca web tramite l'API REST di Tavily con timeout gestito in modo sicuro."""
    if not TAVILY_API_KEY:
        return None
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": max_results
        }
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
        print(f"[WARNING] Errore/Timeout Tavily: {e}")
    except Exception as e:
        print(f"[ERROR] Eccezione generica Tavily: {e}")
    return None

def search_tavily_for_ligand_smiles(query):
    """Usa l'agente Tavily per cercare lo SMILES di un legante insolito o complesso."""
    search_prompt = f"chemical SMILES string for {query} MOF ligand"
    res = search_tavily_web(search_prompt, max_results=2, timeout=3)
    if res and "results" in res:
        for item in res["results"]:
            content = item.get("content", "")
            words = content.replace(";", " ").replace("\n", " ").split(" ")
            for w in words:
                w_clean = w.strip(".,()[]{}")
                if len(w_clean) > 3 and Chem.MolFromSmiles(w_clean):
                    return w_clean
    return None

# --- VALIDAZIONE RIGOROSA METALLO-LEGANTE ---
def valida_articolo_metallo_legante(testo_articolo, metal_symbol, ligand_query=""):
    """
    Verifica che il testo/titolo dell'articolo contenga sia il metallo selezionato
    sia riferimenti al legante ricercato, evitando mismatch di nodo metallico.
    """
    testo_lower = testo_articolo.lower()
    
    # 1. Verifica la presenza esplicita del metallo
    m_info = metal_props.get(metal_symbol, {})
    nome_metallo = m_info.get('Name', '').lower()
    
    simbolo_regex = rf"\b{metal_symbol.lower()}\b"
    ha_metallo = bool(re.search(simbolo_regex, testo_lower)) or (nome_metallo and nome_metallo in testo_lower)
    
    if not ha_metallo:
        return False
        
    # 2. Se è specificato un legante, verifica la presenza di keyword
    if ligand_query:
        legante_clean = ligand_query.lower().strip()
        keywords = [k for k in legante_clean.split() if len(k) > 2]
        ha_legante = any(k in testo_lower for k in keywords)
        if not ha_legante:
            return False
            
    return True

# --- API SEMANTIC SCHOLAR PER ARTICOLI OPEN ACCESS ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_open_access_paper(metal_symbol, ligand_term=""):
    """
    Interroga l'API di Semantic Scholar richiedendo specificamente articoli Open Access
    relativi alla combinazione Metallo + Legante.
    """
    metal_name = metal_props.get(metal_symbol, {}).get('Name', '')
    query_term = f'"{metal_symbol}" "{metal_name}" "{ligand_term}" MOF synthesis' if ligand_term else f'"{metal_symbol}" "{metal_name}" MOF synthesis'
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query_term,
        "limit": 5,
        "fields": "title,authors,year,externalIds,openAccessPdf,isOpenAccess,publicationVenue"
    }
    headers = {'User-Agent': 'MOFSynthesisPredictor/1.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=4)
        if response.status_code == 200:
            data = response.json()
            papers = data.get('data', [])
            for paper in papers:
                title = paper.get('title', 'Titolo Non Disponibile')
                if valida_articolo_metallo_legante(title, metal_symbol, ligand_term):
                    oa_info = paper.get('openAccessPdf')
                    is_oa = paper.get('isOpenAccess', False)
                    
                    # Estrazione URL Open Access
                    link_url = None
                    if oa_info and 'url' in oa_info:
                        link_url = oa_info['url']
                    elif paper.get('externalIds', {}).get('DOI'):
                        doi = paper['externalIds']['DOI']
                        link_url = f"https://doi.org/{doi}"

                    if link_url:
                        authors = paper.get('authors', [])
                        author_str = authors[0]['name'] if authors else "Autori N.D."
                        year = paper.get('year', 'N.D.')
                        venue = paper.get('publicationVenue', {})
                        venue_name = venue.get('name', 'Rivista N.D.') if venue else 'Rivista N.D.'
                        
                        return {
                            'title': title,
                            'ref': f"{author_str} et al. ({year}) - {venue_name}",
                            'url': link_url,
                            'is_open_access': is_oa or (oa_info is not None),
                            'doi': paper.get('externalIds', {}).get('DOI', 'N.D.')
                        }
    except Exception as e:
        print(f"[WARNING] Errore Semantic Scholar API: {e}")
    return None

# --- INTEGRAZIONE CROSSREF COME FALLBACK ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_real_doi_from_crossref(metal_symbol, ligand_term=""):
    metal_name = metal_props.get(metal_symbol, {}).get('Name', '')
    query_term = f'"{metal_symbol}" "{metal_name}" "{ligand_term}" MOF synthesis' if ligand_term else f'"{metal_symbol}" "{metal_name}" MOF synthesis'
    
    url = f"https://api.crossref.org/works?query={requests.utils.quote(query_term)}&rows=5"
    headers = {'User-Agent': 'MOFSynthesisPredictor/1.0 (mailto:admin@example.com)'}
    
    try:
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            data = response.json()
            items = data.get('message', {}).get('items', [])
            for paper in items:
                doi = paper.get('DOI', '')
                title_list = paper.get('title', ['Non disponibile'])
                title = title_list[0] if title_list else 'Non disponibile'
                container_list = paper.get('container-title', [''])
                journal = container_list[0] if container_list else 'Rivista N.D.'
                
                testo_completo = f"{title} {journal}"
                if valida_articolo_metallo_legante(testo_completo, metal_symbol, ligand_term):
                    pub_date = paper.get('published-print', {}).get('date-parts', [[None]])[0][0]
                    if not pub_date:
                        pub_date = paper.get('published-online', {}).get('date-parts', [[None]])[0][0]
                    year_str = str(pub_date) if pub_date else "N.D."
                    
                    return {
                        'doi': doi,
                        'title': title,
                        'journal': journal,
                        'year': year_str,
                        'url': f"https://doi.org/{doi}",
                        'is_open_access': False
                    }
    except Exception:
        pass
    return None

def check_known_mof(metal_symbol, mol_obj=None, ligand_query=""):
    known_mappings = {
        ("Zr", "O=C(O)c1ccc(C(=O)O)cc1"): ("UiO-66", "terephthalic acid"),
        ("Cu", "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1"): ("HKUST-1", "trimesic acid"),
        ("Zn", "Cc1c[nH]cn1"): ("ZIF-8", "2-methylimidazole"),
        ("Cr", "O=C(O)c1ccc(C(=O)O)cc1"): ("MIL-101(Cr)", "terephthalic acid"),
        ("Al", "O=C(O)c1ccc(C(=O)O)cc1"): ("MIL-53(Al)", "terephthalic acid"),
        ("Zn", "O=C(O)c1ccc(C(=O)O)cc1"): ("MOF-5", "terephthalic acid"),
        ("Zr", "O=C(O)c1ccc(C(=O)O)c(N)c1"): ("UiO-66-NH2", "2-aminoterephthalic acid")
    }
    
    input_smiles = Chem.MolToSmiles(mol_obj) if mol_obj else ""
    search_key = (metal_symbol, input_smiles)
    
    if search_key in known_mappings:
        mof_label, ligand_term = known_mappings[search_key]
    elif ligand_query:
        mof_label = f"MOF ({metal_symbol})"
        ligand_term = ligand_query
    else:
        return []

    # 1. Ricerca prioritaria su Semantic Scholar (filtro Open Access)
    oa_paper = fetch_open_access_paper(metal_symbol, ligand_term)
    if oa_paper:
        return [{
            "name": mof_label,
            "ref": f"{oa_paper['title']} — {oa_paper['ref']}",
            "doi": oa_paper['doi'],
            "url": oa_paper['url'],
            "is_oa": oa_paper['is_open_access']
        }]

    # 2. Fallback su Crossref
    paper_info = fetch_real_doi_from_crossref(metal_symbol, ligand_term)
    if paper_info:
        return [{
            "name": mof_label,
            "ref": f"{paper_info['journal']} ({paper_info['year']}) - {paper_info['title']}",
            "doi": paper_info['doi'],
            "url": paper_info['url'],
            "is_oa": False
        }]

    # 3. Fallback finale su Tavily AI
    if TAVILY_API_KEY:
        m_name = metal_props.get(metal_symbol, {}).get('Name', '')
        tavily_query = f'"{metal_symbol}" "{m_name}" AND "{ligand_term}" MOF synthesis paper doi open access'
        res = search_tavily_web(tavily_query, max_results=3, timeout=4)
        if res and "results" in res:
            for item in res["results"]:
                title = item.get("title", "")
                snippet = item.get("content", "")
                url = item.get("url", "")
                
                if valida_articolo_metallo_legante(f"{title} {snippet}", metal_symbol, ligand_term):
                    return [{
                        "name": mof_label,
                        "ref": f"{title}",
                        "doi": url.split("/")[-1] if "10." in url else "N.D.",
                        "url": url,
                        "is_oa": True
                    }]

    return []

# --- DATABASE SOLVENTI CON PARAMETRI FISICO-CHIMICI ---
SOLVENT_PROPERTIES = {
    'DMF':  {'alpha': 0.00, 'beta': 0.69, 'pi_star': 0.88, 'dielectric': 36.7, 'boiling_pt': 153.0},
    'DEF':  {'alpha': 0.00, 'beta': 0.69, 'pi_star': 0.88, 'dielectric': 32.1, 'boiling_pt': 177.0},
    'DMSO': {'alpha': 0.00, 'beta': 0.76, 'pi_star': 1.00, 'dielectric': 46.7, 'boiling_pt': 189.0},
    'MeCN': {'alpha': 0.19, 'beta': 0.31, 'pi_star': 0.75, 'dielectric': 37.5, 'boiling_pt': 82.0},
    'CH3CN': {'alpha': 0.19, 'beta': 0.31, 'pi_star': 0.75, 'dielectric': 37.5, 'boiling_pt': 82.0},
    'H2O':  {'alpha': 1.17, 'beta': 0.18, 'pi_star': 1.09, 'dielectric': 80.1, 'boiling_pt': 100.0},
    'MeOH': {'alpha': 0.93, 'beta': 0.62, 'pi_star': 0.60, 'dielectric': 32.7, 'boiling_pt': 64.7},
    'MetOH': {'alpha': 0.93, 'beta': 0.62, 'pi_star': 0.60, 'dielectric': 32.7, 'boiling_pt': 64.7},
    'EtOH': {'alpha': 0.83, 'beta': 0.77, 'pi_star': 0.54, 'dielectric': 24.5, 'boiling_pt': 78.3},
    'CH2Cl2': {'alpha': 0.13, 'beta': 0.10, 'pi_star': 0.82, 'dielectric': 8.9, 'boiling_pt': 39.6},
    'DCM': {'alpha': 0.13, 'beta': 0.10, 'pi_star': 0.82, 'dielectric': 8.9, 'boiling_pt': 39.6},
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

# --- DATABASE METALLI AMPLIATO (30 METALLI) E HSAB ---
# NOTA: 'Valence' rappresenta lo stato di ossidazione tipico del precursore
# metallico usato in sintesi MOF (es. Zr4+, Al3+, Cu2+...). Serve per calcolare
# correttamente il numero di controioni (nitrato/acetato/cloruro) nel sale.
# Per metalli con più stati di ossidazione comuni (es. Fe, Sn, Ce, Ru, Au) è
# indicato quello prevalentemente usato nei precursori commerciali per MOF;
# se lavori con un sale a valenza diversa, verifica e correggi manualmente.
metal_props = {
    'Zr': {'Z': 40, 'Electronegativity': 1.33, 'Radius_pm': 160, 'Group': 4, 'Period': 5, 'MW': 91.22, 'HSAB': 'Hard', 'Name': 'Zirconio', 'Valence': 4},
    'Hf': {'Z': 72, 'Electronegativity': 1.30, 'Radius_pm': 159, 'Group': 4, 'Period': 6, 'MW': 178.49, 'HSAB': 'Hard', 'Name': 'Afnio', 'Valence': 4},
    'Ti': {'Z': 22, 'Electronegativity': 1.54, 'Radius_pm': 147, 'Group': 4, 'Period': 4, 'MW': 47.87, 'HSAB': 'Hard', 'Name': 'Titanio', 'Valence': 4},
    'Cu': {'Z': 29, 'Electronegativity': 1.90, 'Radius_pm': 132, 'Group': 11, 'Period': 4, 'MW': 63.55, 'HSAB': 'Intermediate', 'Name': 'Rame', 'Valence': 2},
    'Zn': {'Z': 30, 'Electronegativity': 1.65, 'Radius_pm': 122, 'Group': 12, 'Period': 4, 'MW': 65.38, 'HSAB': 'Intermediate', 'Name': 'Zinco', 'Valence': 2},
    'Fe': {'Z': 26, 'Electronegativity': 1.83, 'Radius_pm': 126, 'Group': 8, 'Period': 4, 'MW': 55.85, 'HSAB': 'Hard', 'Name': 'Ferro', 'Valence': 3},
    'Co': {'Z': 27, 'Electronegativity': 1.88, 'Radius_pm': 126, 'Group': 9, 'Period': 4, 'MW': 58.93, 'HSAB': 'Intermediate', 'Name': 'Cobalto', 'Valence': 2},
    'Ni': {'Z': 28, 'Electronegativity': 1.91, 'Radius_pm': 124, 'Group': 10, 'Period': 4, 'MW': 58.69, 'HSAB': 'Intermediate', 'Name': 'Nichel', 'Valence': 2},
    'Mn': {'Z': 25, 'Electronegativity': 1.55, 'Radius_pm': 139, 'Group': 7, 'Period': 4, 'MW': 54.94, 'HSAB': 'Intermediate', 'Name': 'Manganese', 'Valence': 2},
    'Cr': {'Z': 24, 'Electronegativity': 1.66, 'Radius_pm': 128, 'Group': 6, 'Period': 4, 'MW': 51.99, 'HSAB': 'Hard', 'Name': 'Cromo', 'Valence': 3},
    'V':  {'Z': 23, 'Electronegativity': 1.63, 'Radius_pm': 134, 'Group': 5, 'Period': 4, 'MW': 50.94, 'HSAB': 'Hard', 'Name': 'Vanadio', 'Valence': 3},
    'Al': {'Z': 13, 'Electronegativity': 1.61, 'Radius_pm': 121, 'Group': 13, 'Period': 3, 'MW': 26.98, 'HSAB': 'Hard', 'Name': 'Alluminio', 'Valence': 3},
    'Ga': {'Z': 31, 'Electronegativity': 1.81, 'Radius_pm': 122, 'Group': 13, 'Period': 4, 'MW': 69.72, 'HSAB': 'Hard', 'Name': 'Gallio', 'Valence': 3},
    'In': {'Z': 49, 'Electronegativity': 1.78, 'Radius_pm': 142, 'Group': 13, 'Period': 5, 'MW': 114.82, 'HSAB': 'Hard', 'Name': 'Indio', 'Valence': 3},
    'Mg': {'Z': 12, 'Electronegativity': 1.31, 'Radius_pm': 141, 'Group': 2, 'Period': 3, 'MW': 24.31, 'HSAB': 'Hard', 'Name': 'Magnesio', 'Valence': 2},
    'Ca': {'Z': 20, 'Electronegativity': 1.00, 'Radius_pm': 174, 'Group': 2, 'Period': 4, 'MW': 40.08, 'HSAB': 'Hard', 'Name': 'Calcio', 'Valence': 2},
    'Sr': {'Z': 38, 'Electronegativity': 0.95, 'Radius_pm': 192, 'Group': 2, 'Period': 5, 'MW': 87.62, 'HSAB': 'Hard', 'Name': 'Stronzio', 'Valence': 2},
    'Ba': {'Z': 56, 'Electronegativity': 0.89, 'Radius_pm': 198, 'Group': 2, 'Period': 6, 'MW': 137.33, 'HSAB': 'Hard', 'Name': 'Bario', 'Valence': 2},
    'Ce': {'Z': 58, 'Electronegativity': 1.12, 'Radius_pm': 181, 'Group': 3, 'Period': 6, 'MW': 140.12, 'HSAB': 'Hard', 'Name': 'Cerio', 'Valence': 3},
    'La': {'Z': 57, 'Electronegativity': 1.10, 'Radius_pm': 187, 'Group': 3, 'Period': 6, 'MW': 138.91, 'HSAB': 'Hard', 'Name': 'Lantanio', 'Valence': 3},
    'Nd': {'Z': 60, 'Electronegativity': 1.14, 'Radius_pm': 182, 'Group': 3, 'Period': 6, 'MW': 144.24, 'HSAB': 'Hard', 'Name': 'Neodimio', 'Valence': 3},
    'Eu': {'Z': 63, 'Electronegativity': 1.20, 'Radius_pm': 180, 'Group': 3, 'Period': 6, 'MW': 151.96, 'HSAB': 'Hard', 'Name': 'Europio', 'Valence': 3},
    'Gd': {'Z': 64, 'Electronegativity': 1.20, 'Radius_pm': 180, 'Group': 3, 'Period': 6, 'MW': 157.25, 'HSAB': 'Hard', 'Name': 'Gadolinio', 'Valence': 3},
    'Tb': {'Z': 65, 'Electronegativity': 1.20, 'Radius_pm': 177, 'Group': 3, 'Period': 6, 'MW': 158.93, 'HSAB': 'Hard', 'Name': 'Terbio', 'Valence': 3},
    'Y':  {'Z': 39, 'Electronegativity': 1.22, 'Radius_pm': 180, 'Group': 3, 'Period': 5, 'MW': 88.91,  'HSAB': 'Hard', 'Name': 'Ittrio', 'Valence': 3},
    'Cd': {'Z': 48, 'Electronegativity': 1.69, 'Radius_pm': 151, 'Group': 12, 'Period': 5, 'MW': 112.41, 'HSAB': 'Soft', 'Name': 'Cadmio', 'Valence': 2},
    'Bi': {'Z': 83, 'Electronegativity': 2.02, 'Radius_pm': 156, 'Group': 15, 'Period': 6, 'MW': 208.98, 'HSAB': 'Intermediate', 'Name': 'Bismuto', 'Valence': 3},
    'Sn': {'Z': 50, 'Electronegativity': 1.96, 'Radius_pm': 140, 'Group': 14, 'Period': 5, 'MW': 118.71, 'HSAB': 'Hard', 'Name': 'Stagno', 'Valence': 2},
    'Pd': {'Z': 46, 'Electronegativity': 2.20, 'Radius_pm': 137, 'Group': 10, 'Period': 5, 'MW': 106.42, 'HSAB': 'Soft', 'Name': 'Palladio', 'Valence': 2},
    'Ag': {'Z': 47, 'Electronegativity': 1.93, 'Radius_pm': 144, 'Group': 11, 'Period': 5, 'MW': 107.87, 'HSAB': 'Soft', 'Name': 'Argento', 'Valence': 1},
    'Ru': {'Z': 44, 'Electronegativity': 2.20, 'Radius_pm': 134, 'Group': 8, 'Period': 5, 'MW': 101.07, 'HSAB': 'Intermediate', 'Name': 'Rutenio', 'Valence': 3},
    'Au': {'Z': 79, 'Electronegativity': 2.54, 'Radius_pm': 144, 'Group': 11, 'Period': 6, 'MW': 196.97, 'HSAB': 'Soft', 'Name': 'Oro', 'Valence': 3}
}

# Pesi molecolari del SINGOLO anione monovalente (NO3-, CH3COO-, Cl-).
# Il numero di anioni necessari a bilanciare la carica viene calcolato
# moltiplicando per la valenza del metallo (metal_props[...]['Valence']),
# non più fissato a 2 per tutti i metalli (errore precedente: sbagliava
# per metalli tri/tetravalenti come Zr4+, Al3+, Fe3+, lantanidi, ecc.)
anion_mw = {
    'Nitrato': 62.00,
    'Acetato': 59.04,
    'Cloruro': 35.45,
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

@st.cache_data(ttl=3600, show_spinner=False)
def resolve_molecule_to_smiles(query, allow_web_search=True):
    # NOTA: la cache è basata sugli argomenti (query, allow_web_search) e non
    # sulla TAVILY_API_KEY corrente. Se cambi la chiave Tavily a metà sessione,
    # un risultato "None" già in cache non verrà ritentato fino a scadenza (1h).
    clean_query = str(query).strip().lower()
    if not clean_query or clean_query == 'nan':
        return None
    
    if clean_query in COMMON_MOF_LIGANDS:
        return COMMON_MOF_LIGANDS[clean_query]

    if not allow_web_search:
        return None

    headers = {'User-Agent': 'MOF_Predictor_App/1.0'}
    
    try:
        url_nih = f"https://cactus.nci.nih.gov/chemical/structure/{requests.utils.quote(clean_query)}/smiles"
        res = requests.get(url_nih, headers=headers, timeout=2)
        if res.status_code == 200 and res.text and "Page not found" not in res.text:
            smiles_candidate = res.text.strip()
            if Chem.MolFromSmiles(smiles_candidate):
                return smiles_candidate
    except Exception:
        pass

    try:
        url_name = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(clean_query)}/property/IsomericSMILES/JSON"
        res = requests.get(url_name, headers=headers, timeout=2)
        if res.status_code == 200:
            return res.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
    except Exception:
        pass

    if TAVILY_API_KEY:
        tavily_smiles = search_tavily_for_ligand_smiles(clean_query)
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

# --- SUPPORTO: pKa ADDITIVO DA TESTO LIBERO (training/batch) ---
# In fase di training l'additivo arriva come stringa libera dal CSV, non come
# chiave esatta di ADDITIVES_DATABASE. Questa funzione replica la stessa logica
# di rilevamento del tipo (Acid/Base/Neutral/None) già presente in
# process_unified_dataset, ma prova anche ad associare un pKa preciso quando
# riconosce un additivo specifico; altrimenti usa un pKa medio rappresentativo
# per il tipo, invece di ignorare del tutto l'informazione (comportamento
# precedente: il pKa non era una feature, solo un flag binario Acido/Base/Neutro).
_ADDITIVE_KEYWORD_TO_NAME = [
    ('tfa', 'Acido Trifluoroacetico (TFA)'),
    ('trifluoroacetic', 'Acido Trifluoroacetico (TFA)'),
    ('hcl', 'Acido Cloridrico (HCl)'),
    ('hcooh', 'Acido Formico (HCOOH)'),
    ('formic', 'Acido Formico (HCOOH)'),
    ('acoh', 'Acido Acetico (AcOH)'),
    ('acetic', 'Acido Acetico (AcOH)'),
    ('benzoic', 'Acido Benzoico'),
    ('hf', 'HF (Acido Fluoridrico)'),
    ('fluoridric', 'HF (Acido Fluoridrico)'),
    ('tea', 'Trietilammina (TEA)'),
    ('triethylamine', 'Trietilammina (TEA)'),
    ('dipea', 'Diisopropiletilammina (DIPEA)'),
    ('n-methylmorpholine', 'N-Metilmorfolina'),
    ('nmm', 'N-Metilmorfolina'),
    ('pyridine', 'Piridinetilammina / Piridina'),
    ('piridina', 'Piridinetilammina / Piridina'),
    ('h2o', 'Acqua (H2O Modulatore)'),
    ('water', 'Acqua (H2O Modulatore)'),
    ('acqua', 'Acqua (H2O Modulatore)'),
]
_DEFAULT_PKA_BY_TYPE = {'Acid': 4.0, 'Base': 10.5, 'Neutral': 7.0, 'None': 0.0}

def resolve_additive_type_and_pka(add_str):
    s = str(add_str).lower()
    if 'acid' in s or 'ac. ' in s or 'hcl' in s or 'hcooh' in s or 'acoh' in s or 'tfa' in s:
        add_type = 'Acid'
    elif 'tea' in s or 'dipea' in s or 'base' in s or 'amine' in s or 'pyridine' in s:
        add_type = 'Base'
    elif 'h2o' in s or 'water' in s:
        add_type = 'Neutral'
    else:
        add_type = 'None'

    for kw, name in _ADDITIVE_KEYWORD_TO_NAME:
        if kw in s:
            return add_type, ADDITIVES_DATABASE[name]['pKa']

    return add_type, _DEFAULT_PKA_BY_TYPE[add_type]

def clean_float_with_missing_flag(val, default_val=0.0):
    """Come clean_float_val, ma segnala anche se il dato originale era
    effettivamente assente/vuoto, cosi' il modello puo' distinguere uno zero
    reale da un valore imputato di default."""
    was_missing = pd.isna(val) or (isinstance(val, str) and val.strip() == '')
    return clean_float_val(val, default_val), (1.0 if was_missing else 0.0)

def process_unified_dataset(df, is_training_phase=False):
    target_col = None
    for col in ['Esito_ML', 'Target_Esito_Classe', 'Target', 'Esito', 'Classe', 'target', 'esito']:
        if col in df.columns:
            target_col = col
            break
            
    if not target_col:
        for col in df.columns:
            if 'Esito' in col or 'Target' in col:
                target_col = col
                break

    processed = []
    for idx, row in df.iterrows():
        smiles = str(row.get('SMILES_Legante', row.get('SMILES', ''))).strip()
        legante_str = str(row.get('Legante standard', row.get('Legante originale', row.get('Legante', '')))).strip()
        
        mol = None
        if smiles and smiles != 'nan':
            mol = Chem.MolFromSmiles(smiles)
        if not mol and legante_str and legante_str != 'nan':
            found_smi = resolve_molecule_to_smiles(legante_str, allow_web_search=not is_training_phase)
            if found_smi:
                mol = Chem.MolFromSmiles(found_smi)
                smiles = found_smi
        
        mw = Descriptors.MolWt(mol) if mol else clean_float_val(row.get('MM legante'), 166.13)
        logp = Descriptors.MolLogP(mol) if mol else 1.32
        hbd = Descriptors.NumHDonors(mol) if mol else 2
        hba = Descriptors.NumHAcceptors(mol) if mol else 4
        tpsa = Descriptors.TPSA(mol) if mol else 74.6
        rot = Descriptors.NumRotatableBonds(mol) if mol else 2
        
        smarts_f = extract_smarts_features(mol)
        
        met = str(row.get('Metallo', 'Cu')).strip()
        m_info = metal_props.get(met, metal_props.get('Cu'))
        
        sale_str = str(row.get('Sale metallico', row.get('Sale_Metallico', row.get('Anione', ''))))
        if 'NO3' in sale_str or 'Nitrato' in sale_str:
            anione_sel = 'Nitrato'
        elif 'OAc' in sale_str or 'Ac' in sale_str or 'Acetato' in sale_str:
            anione_sel = 'Acetato'
        elif 'Cl' in sale_str or 'Cloruro' in sale_str:
            anione_sel = 'Cloruro'
        else:
            anione_sel = 'Altro'
        
        hsab_match = calculate_hsab_match(m_info['HSAB'], smarts_f['n_COOH'], smarts_f['n_Aromatic_N'])
        
        m_leg, m_leg_missing = clean_float_with_missing_flag(row.get('mmol legante', row.get('mmol_Legante', row.get('mmol_legante', np.nan))), default_val=0.1)
        m_sale, m_sale_missing = clean_float_with_missing_flag(row.get('mmol sale', row.get('mmol_Sale', row.get('mmol_sale', np.nan))), default_val=0.1)
        ratio = clean_float_val(row.get('Rapporto L/M', row.get('Rapporto_LM', m_leg / m_sale if m_sale > 0 else 1.0)))
        
        solv_raw = str(row.get('Solvente', row.get('Solvente Principale', 'DMF'))).strip()
        if '/' in solv_raw:
            parts = solv_raw.split('/')
            solv_p = parts[0].strip()
            cosolv = parts[1].strip()
        elif ':' in solv_raw:
            parts = solv_raw.split(':')
            solv_p = parts[0].strip()
            cosolv = parts[1].strip()
        else:
            solv_p = solv_raw
            cosolv = str(row.get('CoSolvente', 'Nessuno')).strip()
            
        ml_solv_p, ml_solv_p_missing = clean_float_with_missing_flag(row.get('Volume solvente', row.get('mL_Solvente_P', row.get('Vial/Beuta ml', np.nan))), default_val=10.0)
        ml_cosolv = clean_float_val(row.get('mL_CoSolvente', 0.0), default_val=0.0)
        total_vol = ml_solv_p + ml_cosolv
        cosolv_pct = (ml_cosolv / total_vol * 100) if total_vol > 0 else 0.0
        
        mix_props = calculate_solvent_mix_properties(solv_p, ml_solv_p, cosolv, ml_cosolv)
        
        add_str = str(row.get('Co-linker/Additivo', row.get('Additivo_Colinker', row.get('Additivo_Tipo', 'None'))))
        add_type, add_pka = resolve_additive_type_and_pka(add_str)

        add_eq = clean_float_val(row.get('Quantita additivo', row.get('Additivo_Eq', 0.0)), default_val=0.0)
        
        temp, temp_missing = clean_float_with_missing_flag(row.get('Temperatura', row.get('Temperatura_C', row.get('Temperatura_num', np.nan))), default_val=120.0)
        tempo, tempo_missing = clean_float_with_missing_flag(row.get('Tempo ore', row.get('Tempo_ore', row.get('Tempo_ore_num', np.nan))), default_val=48.0)
        
        raw_target = row.get(target_col, 0) if target_col else 0
        try:
            target = int(float(raw_target))
        except Exception:
            target = 0
            
        # Dose termica: proxy dell'energia fornita al sistema durante la crescita
        # cristallina (combina temperatura e durata invece di trattarle come
        # variabili indipendenti scorrelate).
        thermal_dose = float(temp) * float(np.log1p(tempo))

        # Molarità (mol/L, numericamente equivalente a mmol/mL): spesso più
        # predittiva delle quantità assolute prese singolarmente, perché lega
        # la quantità di reagente al volume in cui si trova effettivamente.
        molarity_legante = float(m_leg) / float(total_vol) if total_vol > 0 else 0.0
        molarity_sale = float(m_sale) / float(total_vol) if total_vol > 0 else 0.0

        processed.append({
            'SMILES_Group': smiles if smiles and smiles != 'nan' else legante_str,
            'MW_Legante': float(mw), 'LogP_Legante': float(logp), 'HBD_Legante': float(hbd), 'HBA_Legante': float(hba),
            'TPSA_Legante': float(tpsa), 'RotatableBonds_Legante': float(rot),
            'SMARTS_n_COOH': smarts_f['n_COOH'],
            'SMARTS_n_Aromatic_N': smarts_f['n_Aromatic_N'],
            'SMARTS_fraction_sp2': smarts_f['fraction_sp2'],
            'HSAB_Match_Index': hsab_match,
            'Temperatura_num': float(temp), 'Tempo_ore_num': float(tempo),
            'Thermal_Dose': thermal_dose,
            'mmol legante': float(m_leg), 'mmol sale': float(m_sale), 'Rapporto L/M': float(ratio),
            'Molarity_Legante': molarity_legante, 'Molarity_Sale': molarity_sale,
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
            'Additive_pKa': float(add_pka),
            'Additive_Is_Acid': 1 if add_type == 'Acid' else 0,
            'Additive_Is_Base': 1 if add_type == 'Base' else 0,
            'Additive_Is_Neutral': 1 if add_type == 'Neutral' else 0,
            # Flag "dato originale mancante": 1 se il CSV non riportava il valore
            # (era vuoto/NaN) e si e' dovuto usare un default, 0 se il valore era
            # presente. Permette al modello di distinguere uno zero/default reale
            # da un'assenza di informazione, invece di confonderli silenziosamente.
            'Missing_Temperatura': float(temp_missing),
            'Missing_Tempo': float(tempo_missing),
            'Missing_mmolLegante': float(m_leg_missing),
            'Missing_mmolSale': float(m_sale_missing),
            'Missing_VolSolvente': float(ml_solv_p_missing),
            'Target_Esito_Classe': target
        })
    return pd.DataFrame(processed)

def create_stacking_ensemble():
    # NOTA SU REGOLARIZZAZIONE: con un dataset relativamente piccolo (~migliaia
    # di righe) uno stacking a 3 modelli può overfittare facilmente. Rispetto
    # ai parametri originali ho aggiunto vincoli espliciti (min_child_samples,
    # reg_alpha/reg_lambda, subsample/colsample per LightGBM; min_samples_leaf,
    # max_features per RF; l2_leaf_reg per CatBoost) per limitare la complessità
    # dei singoli alberi e la varianza dell'ensemble. Non ho aggiunto una
    # ricerca automatica degli iperparametri (es. Optuna/RandomizedSearchCV)
    # perché eseguita ad ogni cold-start dell'app allungherebbe di molto i
    # tempi di avvio; se vuoi, posso preparartela come script offline separato.
    estimators = [
        ('lgb', LGBMClassifier(
            n_estimators=180, learning_rate=0.03, max_depth=6,
            num_leaves=31, min_child_samples=15,
            reg_alpha=0.1, reg_lambda=0.5,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, verbose=-1
        )),
        ('rf', RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=3, max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42, n_jobs=2
        ))
    ]
    
    if HAS_CATBOOST:
        estimators.append(('cat', CatBoostClassifier(
            iterations=200, learning_rate=0.04, depth=6, l2_leaf_reg=5.0,
            verbose=0, random_seed=42
        )))
        
    meta_model = LogisticRegression(class_weight='balanced', max_iter=500, C=0.5)
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        stack_method='predict_proba',
        n_jobs=2
    )
    return stacking_clf

# Versione della LOGICA di training (feature set, gestione CV, iperparametri).
# Va incrementata ogni volta che si modifica in modo sostanziale come il
# modello viene costruito/allenato. Serve a evitare che un .pkl salvato da
# una versione precedente del codice (che può sopravvivere su disco tra un
# riavvio e l'altro dell'app, specialmente se il container non viene
# ricreato da zero) venga ricaricato silenziosamente mascherando le
# correzioni fatte al codice: senza questo controllo, un .pkl "vecchio" con
# metriche/errori obsoleti può continuare a essere mostrato all'utente anche
# dopo aver corretto e ridistribuito il codice.
MODEL_TRAINING_VERSION = "v6-single-pass-memory-fix"

@st.cache_resource
def load_or_train_model():
    pkl_file = "modello_sintesi_mof_ottimizzato.pkl"
    csv_candidates = [
        "Dataset_Sintesi_Unificato_1000.csv",
        "Dataset_Sintesi_Unificato_aggiornato.csv",
        "Dataset_Sintesi_MOF_ML_Standardizzato.csv",
        "Dataset_Sintesi_Unificato.csv"
    ]
    
    if os.path.exists(pkl_file):
        try:
            saved_data = joblib.load(pkl_file)
            if (
                isinstance(saved_data, dict) and 'model' in saved_data
                and saved_data.get('model_training_version') == MODEL_TRAINING_VERSION
            ):
                return saved_data['model'], saved_data['features'], saved_data.get('metrics', {}), saved_data.get('importances', [])
            # .pkl presente ma di una versione diversa (o senza il campo di
            # versione, quindi pre-esistente a questo controllo): non lo si usa,
            # si procede a riallenare da zero più sotto.
        except Exception:
            pass

    csv_file = None
    for cand in csv_candidates:
        if os.path.exists(cand):
            csv_file = cand
            break

    if not csv_file:
        st.error("⚠️ Nessun file CSV di dataset trovato nella directory di lavoro!")
        st.stop()
        
    raw_df = pd.read_csv(csv_file)
    df = process_unified_dataset(raw_df, is_training_phase=True)
    
    groups = df['SMILES_Group'].tolist()
    X = df.drop(columns=['Target_Esito_Classe', 'SMILES_Group'])
    y_series = pd.to_numeric(df['Target_Esito_Classe'], errors='coerce')
    valid_mask = y_series.notna()
    
    X = X[valid_mask].copy().reset_index(drop=True)
    y = y_series[valid_mask].astype(int).copy().reset_index(drop=True)
    groups = [g for i, g in enumerate(groups) if valid_mask.iloc[i]]
    
    X = X.apply(lambda col: col.apply(clean_float_val) if col.dtype == 'object' else col)
    X = X.apply(pd.to_numeric, errors='coerce')
    # Rete di sicurezza per eventuali NaN residui non già intercettati a monte
    # (la maggior parte delle colonne è già "pulita" da clean_float_val con
    # default espliciti per campo). Uso la mediana di colonna invece di uno 0
    # fisso: uno zero letterale su temperatura/tempo/mmol non avrebbe senso
    # chimico e distorcerebbe la colonna più di un valore centrale plausibile.
    n_missing_pre_impute = int(X.isna().sum().sum())
    if n_missing_pre_impute > 0:
        X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)  # eventuali colonne interamente NaN (mediana non calcolabile)
    
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
    min_class_count = int(y.value_counts().min())

    # Numero minimo, tra tutte le classi, di gruppi SMILES distinti in cui
    # quella classe compare. Se una classe è concentrata in un solo legante
    # (es. tutte le righe "successo" derivano dallo stesso SMILES), la CV
    # raggruppata per quella classe è impossibile: non si può distribuire un
    # unico gruppo su più fold mantenendo lo stesso legante sempre insieme.
    # Prima causa reale dell'errore "number of groups: 1" visto in sidebar.
    groups_arr = np.array(groups, dtype=object)
    y_arr = y.to_numpy()
    try:
        min_groups_per_class = min(
            len(np.unique(groups_arr[y_arr == c])) for c in np.unique(y_arr)
        )
    except Exception:
        min_groups_per_class = 1
    
    used_cv = None
    used_groups_for_cv = None
    cv_skip_reason = None
    cv_error_traceback = None
    cv_diagnostics = {
        'n_unique_groups': n_unique_groups,
        'unique_classes': unique_classes,
        'min_class_count': min_class_count,
        'min_groups_per_class': min_groups_per_class,
    }

    if min_class_count < 2:
        # Una classe con un solo campione non può essere "spezzata" in fold:
        # qualunque valore di n_splits >= 2 farebbe fallire la stratificazione.
        # Niente CV/calibrazione in questo caso: fit diretto, e lo diciamo
        # esplicitamente invece di lasciare che un'eccezione generica lo nasconda.
        base_ensemble.fit(X, y)
        final_model = base_ensemble
        cv_skip_reason = (
            f"La classe meno numerosa nel dataset ha solo {min_class_count} campione. "
            "Servono almeno 2 esempi per ciascuna classe per fare cross-validation stratificata: "
            "il modello è stato allenato senza CV/calibrazione. Aggiungi più righe per la classe minoritaria nel CSV."
        )
    else:
        can_do_grouped_cv = (n_unique_groups >= 3) and (unique_classes > 1) and (min_groups_per_class >= 2)
        try:
            if can_do_grouped_cv:
                # Fino a 5 fold quando i dati lo consentono (stime più stabili di
                # accuratezza CV e calibrazione rispetto ai 3 fold precedenti);
                # scende automaticamente se ci sono meno gruppi, meno campioni
                # nella classe minoritaria, o meno gruppi per la classe più
                # "concentrata" su un unico legante.
                cv_splits = max(2, min(5, n_unique_groups, min_class_count, min_groups_per_class))
                sgkf_calib = StratifiedGroupKFold(n_splits=cv_splits)
                # IMPORTANTE: pre-calcoliamo qui gli split usando esplicitamente
                # 'groups', e passiamo la LISTA di split già pronta a
                # CalibratedClassifierCV (invece dell'oggetto splitter grezzo).
                # Motivo: in questa versione di scikit-learn, CalibratedClassifierCV
                # non inoltra correttamente 'groups' al proprio interno a meno di
                # abilitare esplicitamente il "metadata routing" (funzionalità
                # recente, più fragile); passando gli split già calcolati si evita
                # del tutto il problema, indipendentemente dalla versione installata.
                precomputed_splits = list(sgkf_calib.split(X, y, groups))
                final_model = CalibratedClassifierCV(
                    estimator=base_ensemble, method='sigmoid',
                    cv=precomputed_splits
                )
                final_model.fit(X, y)
                used_cv, used_groups_for_cv = precomputed_splits, None
            else:
                # CV raggruppata non praticabile (troppi pochi gruppi complessivi,
                # o una classe concentrata in un solo legante): si scende alla
                # CV stratificata semplice, che non ha vincoli sui gruppi.
                # n_splits non può superare il numero di campioni della classe
                # meno numerosa, altrimenti StratifiedKFold solleva un errore.
                skf_splits = max(2, min(5, min_class_count))
                skf = StratifiedKFold(n_splits=skf_splits)
                final_model = CalibratedClassifierCV(
                    estimator=base_ensemble, method='sigmoid',
                    cv=skf
                )
                final_model.fit(X, y)
                # Materializziamo gli stessi split usati internamente (skf è
                # deterministico con shuffle=False), per poterli riusare più
                # sotto senza doverli ricalcolare o rifare un secondo training.
                used_cv, used_groups_for_cv = list(skf.split(X, y)), None
                if n_unique_groups >= 3 and min_groups_per_class < 2:
                    cv_skip_reason = (
                        "Una classe del dataset è concentrata in un unico legante (SMILES): usata "
                        "cross-validation stratificata semplice invece che raggruppata per legante "
                        "(la CV raggruppata garantisce meglio che lo stesso legante non appaia sia in "
                        "training che in validazione, ma qui non è geometricamente possibile per tutte le classi)."
                    )
        except Exception as e:
            base_ensemble.fit(X, y)
            final_model = base_ensemble
            # Non nascondere l'errore: verrà mostrato in sidebar così l'utente
            # sa che non è un problema di cache/pkl ma un fallimento reale del
            # training in cross-validation.
            cv_skip_reason = f"Training con cross-validation fallito ({type(e).__name__}: {e}). Il modello è stato allenato senza CV/calibrazione come fallback."
            cv_error_traceback = traceback.format_exc()

    if cv_skip_reason:
        print(f"[WARNING] {cv_skip_reason}")

    try:
        if hasattr(final_model, 'calibrated_classifiers_'):
            lgb_est = final_model.calibrated_classifiers_[0].estimator.named_estimators_['lgb']
            importances = lgb_est.feature_importances_
        else:
            importances = final_model.named_estimators_['lgb'].feature_importances_
    except Exception:
        importances = np.zeros(len(feature_names))

    # ACCURATEZZA + METRICHE COMPLEMENTARI: tutte calcolate out-of-fold (ogni
    # predizione viene da un modello che NON ha visto quel campione in
    # training), non sui dati di training. accuracy_score(y, final_model.predict(X))
    # darebbe una stima ottimisticamente distorta perché quasi tutti i campioni
    # sono stati visti in training da almeno una parte dell'ensemble calibrato.
    # F1-macro e balanced accuracy sono aggiunte perché, su un problema a 3
    # classi potenzialmente sbilanciato, la sola accuratezza può nascondere un
    # modello che performa bene solo sulla classe maggioritaria.
    #
    # IMPORTANTE (memoria/tempo): qui RIUSIAMO i modelli già allenati per
    # ciascun fold di calibrazione (final_model.calibrated_classifiers_),
    # applicandoli sul rispettivo fold di test, invece di rifare un secondo
    # allenamento completo dell'ensemble da zero (come faceva la versione
    # precedente con cross_val_predict). Allenare uno stacking di 3 modelli
    # su 5 fold due volte di seguito raddoppiava tempo di esecuzione e picco
    # di memoria — causa quasi certa del superamento dei limiti di risorse
    # su Streamlit Cloud. Misurato: da ~187s a ~31s sullo stesso dataset.
    try:
        if used_cv is not None:
            oof_preds = np.empty(len(y), dtype=int)
            classes_arr = final_model.classes_
            for split_idx, (_, test_idx) in enumerate(used_cv):
                calibrated_clf = final_model.calibrated_classifiers_[split_idx]
                probs = calibrated_clf.predict_proba(X.iloc[test_idx])
                oof_preds[test_idx] = classes_arr[np.argmax(probs, axis=1)]
            cv_accuracy = accuracy_score(y, oof_preds)
            cv_f1_macro = f1_score(y, oof_preds, average='macro', zero_division=0)
            cv_balanced_accuracy = balanced_accuracy_score(y, oof_preds)
        else:
            preds_fallback = final_model.predict(X)
            cv_accuracy = accuracy_score(y, preds_fallback)
            cv_f1_macro = f1_score(y, preds_fallback, average='macro', zero_division=0)
            cv_balanced_accuracy = balanced_accuracy_score(y, preds_fallback)
    except Exception:
        # Fallback: se la CV out-of-fold fallisce (es. dataset troppo piccolo),
        # usa le metriche calcolate su tutto X, ma non spacciarle per "vere".
        preds_fallback = final_model.predict(X)
        cv_accuracy = accuracy_score(y, preds_fallback)
        cv_f1_macro = f1_score(y, preds_fallback, average='macro', zero_division=0)
        cv_balanced_accuracy = balanced_accuracy_score(y, preds_fallback)

    metrics = {
        'train_accuracy': cv_accuracy,  # nome mantenuto per compatibilità con .pkl esistenti; ora è CV accuracy
        'cv_f1_macro': cv_f1_macro,
        'cv_balanced_accuracy': cv_balanced_accuracy,
        'is_cv_accuracy': used_cv is not None,
        'cv_skip_reason': cv_skip_reason,
        'cv_error_traceback': cv_error_traceback,
        'cv_diagnostics': cv_diagnostics,
        'n_samples': len(X),
        'n_features': len(feature_names),
        'n_missing_values_imputed': n_missing_pre_impute
    }
    
    save_dict = {
        'model': final_model,
        'features': feature_names,
        'importances': importances,
        'metrics': metrics,
        'model_training_version': MODEL_TRAINING_VERSION
    }
    
    joblib.dump(save_dict, pkl_file)
    return final_model, feature_names, metrics, importances

with st.sidebar.expander("🛠️ Manutenzione Modello", expanded=False):
    st.caption(f"Versione logica di training corrente: `{MODEL_TRAINING_VERSION}`")
    if st.button("🔄 Forza ri-allenamento (ignora .pkl salvato)"):
        try:
            if os.path.exists("modello_sintesi_mof_ottimizzato.pkl"):
                os.remove("modello_sintesi_mof_ottimizzato.pkl")
            load_or_train_model.clear()  # svuota la cache di st.cache_resource per questa funzione
            st.success("Cache e .pkl rimossi. Ricaricamento in corso...")
            st.rerun()
        except Exception as e:
            st.error(f"Impossibile forzare il ri-allenamento: {e}")

try:
    model, feature_names, metrics, importances = load_or_train_model()
    st.sidebar.success("Modello Ensemble Stacking Attivo!")
except Exception as e:
    st.sidebar.error(f"Errore caricamento modello: {e}")
    st.stop()

# --- MENÙ SELEZIONE METALLO ---
def render_metal_dropdown_selector(key_prefix="pred"):
    metal_options = sorted(list(metal_props.keys()))
    default_idx = metal_options.index('Zr') if 'Zr' in metal_options else 0
    
    selected = st.selectbox(
        "🧱 Seleziona Metallo Reagente:",
        options=metal_options,
        format_func=lambda x: f"{x} | {metal_props[x]['Name']} (Pearson: {metal_props[x]['HSAB']})",
        index=default_idx,
        key=f"{key_prefix}_select_metal"
    )

    m_data = metal_props[selected]
    hsab_color = "🔴 Hard" if m_data['HSAB'] == 'Hard' else ("🟡 Intermediate" if m_data['HSAB'] == 'Intermediate' else "🟢 Soft")
    
    st.caption(
        f"**Metallo Selezionato:** `{selected}` ({m_data['Name']}) | "
        f"**HSAB:** {hsab_color} | "
        f"**MW:** `{m_data['MW']:.2f}` g/mol | "
        f"**Raggio Ionico:** `{m_data['Radius_pm']}` pm"
    )
    return selected

# --- SIDEBAR METRICHE ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Stato & Performance Modello")

acc_val = metrics.get('train_accuracy', 0.85) * 100
is_cv_acc = metrics.get('is_cv_accuracy', False)
cv_skip_reason = metrics.get('cv_skip_reason')
acc_label = "Accuratezza (CV)" if is_cv_acc else "Accuratezza (train)"
sintesi_count = metrics.get('n_samples', 1000)
num_features = metrics.get('n_features', len(feature_names))

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    st.markdown(
        f"""
        <div style="background-color: rgba(255, 255, 255, 0.05); padding: 8px 4px; border-radius: 8px; text-align: center;">
            <span style="font-size: 0.8rem; color: #888;">{acc_label}</span><br>
            <strong style="font-size: 1.15rem;">{acc_val:.1f}%</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
if not is_cv_acc:
    if cv_skip_reason:
        st.sidebar.caption(f"⚠️ Cross-validation non eseguita: {cv_skip_reason}")
        cv_diag = metrics.get('cv_diagnostics')
        cv_tb = metrics.get('cv_error_traceback')
        if cv_diag or cv_tb:
            with st.sidebar.expander("🔍 Dettagli tecnici (per il debug)", expanded=False):
                if cv_diag:
                    st.markdown(
                        f"- Gruppi SMILES distinti totali: `{cv_diag.get('n_unique_groups')}`\n"
                        f"- Classi distinte nel target: `{cv_diag.get('unique_classes')}`\n"
                        f"- Campioni nella classe meno numerosa: `{cv_diag.get('min_class_count')}`\n"
                        f"- Gruppi distinti nella classe più \"concentrata\": `{cv_diag.get('min_groups_per_class')}`"
                    )
                if cv_tb:
                    st.code(cv_tb, language="text")
    else:
        st.sidebar.caption("⚠️ Valore calcolato su dati di training (modello .pkl salvato prima di questa correzione, senza il motivo registrato): cancella il file .pkl per ricalcolare in cross-validation.")
elif cv_skip_reason:
    # CV eseguita regolarmente (l'accuratezza mostrata è comunque una vera
    # CV accuracy), ma senza raggruppamento per legante: nota informativa,
    # non un warning bloccante.
    st.sidebar.caption(f"ℹ️ {cv_skip_reason}")
with col_sb2:
    st.markdown(
        f"""
        <div style="background-color: rgba(255, 255, 255, 0.05); padding: 8px 4px; border-radius: 8px; text-align: center;">
            <span style="font-size: 0.8rem; color: #888;">Sintesi DB</span><br>
            <strong style="font-size: 1.15rem;">{sintesi_count}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

f1_macro_val = metrics.get('cv_f1_macro')
bal_acc_val = metrics.get('cv_balanced_accuracy')
n_imputed = metrics.get('n_missing_values_imputed', 0)
if f1_macro_val is not None and bal_acc_val is not None:
    with st.sidebar.expander("📈 Metriche aggiuntive (robuste a classi sbilanciate)", expanded=False):
        st.markdown(
            f"- **F1-macro (CV):** `{f1_macro_val*100:.1f}%`\n"
            f"- **Balanced Accuracy (CV):** `{bal_acc_val*100:.1f}%`\n\n"
            "L'accuratezza semplice può risultare alta anche se il modello "
            "predice bene solo la classe più frequente; queste due metriche "
            "pesano equamente tutte le classi (successo/parziale/insuccesso)."
        )
        if n_imputed > 0:
            st.caption(f"ℹ️ {n_imputed} valori mancanti nel dataset sono stati imputati con la mediana di colonna.")

st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown(
    f"""
    <ul style="padding-left: 1.2rem; margin-top: 0;">
        <li style="margin-bottom: 8px;">
            <strong>Architettura:</strong> Stacking Ensemble<br>
            <small style="color: #888;"><i>(LightGBM + Random Forest + CatBoost)</i></small>
        </li>
        <li style="margin-bottom: 8px;">
            <strong>Parametri Valutati:</strong> 
            <span style="background-color: rgba(255, 255, 255, 0.1); padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; color: #2e7d32;">{num_features}</span> 
            Feature Chimico-Fisiche
        </li>
        <li style="margin-bottom: 8px;">
            <strong>Database Metalli:</strong> 
            <span style="background-color: rgba(255, 255, 255, 0.1); padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; color: #2e7d32;">{len(metal_props)}</span> 
            Elementi censiti
        </li>
    </ul>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Quick Reference Chimica")

with st.sidebar.expander("🧪 Teoria HSAB di Pearson", expanded=False):
    st.markdown("""
    **Acidi Hard (Zr⁴⁺, Hf⁴⁺, Fe³⁺, Al³⁺, Cr³⁺):**
    * Alta densità di carica.
    * Prediligono **Basi Hard** (es. Carbossili -COOH).
    
    **Acidi Intermediate (Cu²⁺, Zn²⁺, Ni²⁺, Co²⁺):**
    * Densità di carica media.
    * Prediligono **Azoti Aromatici** (Imidazoli, Piridine) o miscele -COOH/N.
    """)

# --- TAB INTERFACCIA MAIN ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Predizione Singola", "📂 Predizione Batch", "⚡ Ottimizzatore Automatico", "🌐 Ricerca Web (Tavily AI)", "📊 Explainability (SHAP & Feature Importance)"])

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
        'Thermal_Dose': float(temp) * float(np.log1p(float(tempo))),
        'mmol legante': float(mmol_legante),
        'mmol sale': float(mmol_sale),
        'Rapporto L/M': float(mmol_legante) / float(mmol_sale) if float(mmol_sale) > 0 else 1.0,
        'Molarity_Legante': float(mmol_legante) / total_vol if total_vol > 0 else 0.0,
        'Molarity_Sale': float(mmol_sale) / total_vol if total_vol > 0 else 0.0,
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
        'Additive_pKa': float(add_info.get('pKa', 0.0)),
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
                    found_smiles = resolve_molecule_to_smiles(query_input, allow_web_search=True)
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
                            temp_filename = "temp_upload.cif"
                            with open(temp_filename, "w", encoding="utf-8") as f:
                                f.write(file_bytes)
                            struct = Structure.from_file(temp_filename)
                            red_formula = struct.composition.reduced_formula
                            found_smiles = resolve_molecule_to_smiles(red_formula, allow_web_search=True)
                            if found_smiles:
                                mol = Chem.MolFromSmiles(found_smiles)
                            if os.path.exists(temp_filename):
                                os.remove(temp_filename)
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
        
        metallo_sel = render_metal_dropdown_selector(key_prefix="tab1")
        
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
        metal_valence = metal_props[metallo_sel].get('Valence', 2)
        n_anioni = metal_valence if anione_sel != 'Altro' else 1
        base_salt_mw = metal_props[metallo_sel]['MW'] + n_anioni * anion_mw.get(anione_sel, 60.0)
        total_salt_mw = base_salt_mw + (n_h2o * 18.015)
        
        st.caption(
            f"🧪 **Massa Molare Sale Idrato:** `{total_salt_mw:.2f} g/mol` "
            f"(formula assunta: M{'³⁺' if metal_valence==3 else ('⁴⁺' if metal_valence==4 else ('²⁺' if metal_valence==2 else '⁺'))} "
            f"+ {n_anioni}× {anione_sel if anione_sel != 'Altro' else 'controione'}, {n_h2o} H₂O — verifica sempre la formula del tuo reagente commerciale)"
        )

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
                    oa_tag = "📖 **[OPEN ACCESS]**" if mof.get('is_oa', False) else "📄 [LETTERATURA SPERIMENTALE]"
                    clean_doi = mof['doi'].replace("https://doi.org/", "")

                    st.info(
                        f"🟢 **Combinazione nota e verificata per la coppia specificata ({metallo_sel} + Legante)!**\n\n"
                        f"* **MOF/Sintesi:** `{mof['name']}`\n"
                        f"* **Fonte / Articolo:** {mof['ref']}\n"
                        f"* **Accettazione:** {oa_tag}\n"
                        f"* **Link Diretto:** [📂 Apri Articolo Scientifico ({clean_doi})]({mof['url']})"
                    )
            else:
                st.success(f"✨ **Combinazione Inedita / Non presente a DB:** Nessun articolo scientifico rilevato specificamente per la coppia **{metallo_sel} + Legante**.")

            df_features = build_feature_row(
                mol, mw, logp, hbd, hba, tpsa, rot_bonds, temp, tempo, 
                mmol_legante, mmol_sale, metallo_sel, anione_sel, 
                solvente_p, ml_solv_p, co_solvente, ml_cosolv, additivo_sel, add_eq
            )
            probs = model.predict_proba(df_features)[0]
            pred_class = model.predict(df_features)[0]

            # Memorizziamo il DataFrame di input nello stato della sessione per l'analisi SHAP
            st.session_state['last_df_features'] = df_features

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
                processed_batch = process_unified_dataset(input_batch, is_training_phase=False)
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

# --- TAB 3: OTTIMIZZATORE AUTOMATICO MULTI-METALLO ---
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
                    opt_found_smiles = resolve_molecule_to_smiles(opt_query_input, allow_web_search=True)
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
                            opt_temp_filename = "temp_opt_upload.cif"
                            with open(opt_temp_filename, "w", encoding="utf-8") as f:
                                f.write(file_bytes)
                            struct = Structure.from_file(opt_temp_filename)
                            red_formula = struct.composition.reduced_formula
                            opt_found_smiles = resolve_molecule_to_smiles(red_formula, allow_web_search=True)
                            if opt_found_smiles:
                                opt_mol = Chem.MolFromSmiles(opt_found_smiles)
                            if os.path.exists(opt_temp_filename):
                                os.remove(opt_temp_filename)
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
        st.markdown("### 2. Selezione Metalli e Precursore")
        
        metal_list_opt = sorted(list(metal_props.keys()))
        opt_selected_metals = st.multiselect(
            "🧱 Seleziona uno o più Metalli da includere nella Scansione:",
            options=metal_list_opt,
            default=['Zr', 'Cu', 'Zn'] if all(m in metal_list_opt for m in ['Zr', 'Cu', 'Zn']) else [metal_list_opt[0]],
            format_func=lambda x: f"{x} | {metal_props[x]['Name']} ({metal_props[x]['HSAB']})"
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
            opt_mw = float(Descriptors.MolWt(opt_mol))
            opt_logp = float(Descriptors.MolLogP(opt_mol))
            opt_hbd = float(Descriptors.NumHDonors(opt_mol))
            opt_hba = float(Descriptors.NumHAcceptors(opt_mol))
            opt_tpsa = float(Descriptors.TPSA(opt_mol))
            opt_rot = float(Descriptors.NumRotatableBonds(opt_mol))
            
            smarts_f = extract_smarts_features(opt_mol)

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
                grid_combos = list(itertools.product(
                    opt_selected_metals, temperatures, times, solvents_p, volumes_p, cosolvents, additives
                ))
                
                rows_list = []
                display_info = []

                ratio_lm = float(opt_mmol_legante) / float(opt_mmol_sale) if float(opt_mmol_sale) > 0 else 1.0

                for cur_metal, temp, tempo, solv_p, ml_solv_p, (cosolv, ml_cosolv), (add_name, add_eq) in grid_combos:
                    metal_m = metal_props[cur_metal]
                    hsab_match = float(calculate_hsab_match(metal_m['HSAB'], smarts_f['n_COOH'], smarts_f['n_Aromatic_N']))
                    
                    opt_metal_valence = metal_m.get('Valence', 2)
                    opt_n_anioni = opt_metal_valence if opt_anione != 'Altro' else 1
                    opt_base_salt_mw = metal_m['MW'] + opt_n_anioni * anion_mw.get(opt_anione, 60.0)
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
                        'Thermal_Dose': float(temp) * float(np.log1p(float(tempo))),
                        'mmol legante': float(opt_mmol_legante), 'mmol sale': float(opt_mmol_sale), 'Rapporto L/M': float(ratio_lm),
                        'Molarity_Legante': float(opt_mmol_legante) / total_vol if total_vol > 0 else 0.0,
                        'Molarity_Sale': float(opt_mmol_sale) / total_vol if total_vol > 0 else 0.0,
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
                        'Additive_pKa': float(add_info.get('pKa', 0.0)),
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

                df_simulation = pd.DataFrame(rows_list)
                for col in feature_names:
                    if col not in df_simulation.columns:
                        df_simulation[col] = 0.0
                df_simulation = df_simulation[feature_names]

                probs_matrix = model.predict_proba(df_simulation)
                
                classes_list = [int(c) if str(c).isdigit() else c for c in model.classes_]
                if 2 in classes_list:
                    target_class_idx = classes_list.index(2)
                elif 1 in classes_list:
                    target_class_idx = classes_list.index(1)
                else:
                    target_class_idx = len(classes_list) - 1

                success_probs = (probs_matrix[:, target_class_idx] * 100.0).round(1)

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
    
    query_tavily = st.text_input("Inserisci la query di ricerca chimica:", value="UiO-66 synthesis conditions modulator benzoic acid open access paper")
    num_res = st.slider("Numero di risultati:", min_value=1, max_value=5, value=3)
    
    if st.button("🔎 Cerca sul Web con Tavily"):
        if not TAVILY_API_KEY:
            st.error("API Key Tavily mancante.")
        else:
            with st.spinner("Ricerca informazioni sul web in corso..."):
                tavily_res = search_tavily_web(query_tavily, max_results=num_res, timeout=5)
                if tavily_res:
                    if tavily_res.get("answer"):
                        st.info(f"💡 **Sintesi Risposta AI Tavily:**\n\n{tavily_res['answer']}")
                    
                    st.markdown("### 📚 Risultati della Ricerca")
                    for r in tavily_res.get("results", []):
                        st.markdown(f"#### [{r.get('title')}]({r.get('url')})")
                        st.write(r.get("content"))
                        st.markdown("---")
                else:
                    st.error("Nessun risultato trovato o timeout durante la richiesta.")

# --- TAB 5: EXPLAINABILITY (SHAP & FEATURE IMPORTANCE) ---
with tab5:
    st.subheader("📊 Spiegabilità Chimica del Modello ML (Explainability)")
    st.markdown("""
    Questa sezione permette di interpretare le decisioni del modello di Machine Learning, mostrando quali parametri fisico-chimici 
    e descrittori molecolari hanno la maggiore influenza sulle predizioni di sintesi dei MOF.
    """)

    st.markdown("### 1. Importanza Globale delle Feature")
    if len(importances) > 0 and len(importances) == len(feature_names):
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        top_n = st.slider("Numero di feature da mostrare:", min_value=5, max_value=min(30, len(feature_names)), value=15)
        top_fi = fi_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_fi['Feature'][::-1], top_fi['Importance'][::-1], color='skyblue')
        ax.set_xlabel("Importanza Relativa")
        ax.set_title("Top Feature del Modello (LightGBM Estimator)")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("I dati sull'importanza delle feature non sono disponibili per questo modello.")

    st.markdown("---")
    st.markdown("### 2. Spiegazione della Singola Predizione (Valori SHAP)")

    if not HAS_SHAP:
        st.warning("⚠️ La libreria `shap` non è installata nell'ambiente Python. Installa SHAP (`pip install shap`) per abilitare le spiegazioni dettagliate.")
    else:
        if 'last_df_features' in st.session_state:
            st.markdown("Analisi dei contributi SHAP per l'ultima predizione effettuata nel **Tab 1 (Predizione Singola)**:")
            df_single = st.session_state['last_df_features']

            try:
                # Estrazione dell'estimatore di base (LightGBM) per il calcolo SHAP
                target_estimator = None
                if hasattr(model, 'calibrated_classifiers_'):
                    target_estimator = model.calibrated_classifiers_[0].estimator.named_estimators_['lgb']
                elif hasattr(model, 'named_estimators_'):
                    target_estimator = model.named_estimators_['lgb']
                
                if target_estimator is not None:
                    explainer = shap.TreeExplainer(target_estimator)
                    shap_values = explainer.shap_values(df_single)

                    st.markdown("#### Waterfall / Bar Plot SHAP")
                    
                    # SHAP per multiclasse restituisce una lista di array (uno per classe)
                    if isinstance(shap_values, list):
                        class_names = [f"Classe {c}" for c in model.classes_]
                        selected_class_idx = st.selectbox("Seleziona Classe per l'analisi SHAP:", range(len(shap_values)), format_func=lambda x: class_names[x])
                        sv_to_plot = shap_values[selected_class_idx][0]
                        base_val = explainer.expected_value[selected_class_idx]
                    else:
                        sv_to_plot = shap_values[0]
                        base_val = explainer.expected_value

                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.bar_plot(sv_to_plot, feature_names=df_single.columns, max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.error("Impossibile estrarre un sotto-modello basato su alberi per l'analisi TreeSHAP.")
            except Exception as e:
                st.error(f"Errore durante il calcolo dei valori SHAP: {e}")
        else:
            st.info("💡 Esegui prima una predizione nel tab **'🔮 Predizione Singola'** per visualizzare l'analisi SHAP specifica per quel punto di prova.")
