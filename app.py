import os
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

# --- IMPORTAZIONI CHIMICHE OPZIONALI (RDKit) ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- IMPORTAZIONI VISUALIZZAZIONE 3D OPZIONALI (py3Dmol / stmol) ---
try:
    import py3Dmol
    from stmol import showmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

# Ignora warning non critici
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURAZIONE PAGINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="MOF Synthesis AI Predictor & Optimizer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS per migliorare l'estetica dell'interfaccia
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧪 MOF Synthesis AI Predictor & Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Piattaforma avanzata di Machine Learning per la predizione, ottimizzazione e spiegabilità chimica della sintesi di Metal-Organic Frameworks.</div>', unsafe_allow_html=True)

# ==========================================
# 2. GENERATORE DATASET DEMO DI BACKUP
# ==========================================
def generate_synthetic_mof_dataset():
    """Genera un dataset bilanciato e realistico di sintesi MOF se il CSV non è presente."""
    np.random.seed(42)
    n_samples = 300
    
    temperatures = np.random.uniform(60, 220, n_samples)
    times = np.random.uniform(2, 96, n_samples)
    metal_concs = np.random.uniform(0.02, 0.5, n_samples)
    ligand_concs = np.random.uniform(0.02, 0.5, n_samples)
    modulator_eqs = np.random.uniform(0, 60, n_samples)
    phs = np.random.uniform(1.0, 11.0, n_samples)
    stirring_speeds = np.random.choice([0, 200, 500, 800], size=n_samples)
    
    metals = np.random.choice(['Zr', 'Cu', 'Zn', 'Fe', 'Al', 'UiO-Metal'], size=n_samples)
    solvents = np.random.choice(['DMF', 'DEF', 'H2O', 'Ethanol', 'DMF/H2O'], size=n_samples)
    
    esiti = []
    for i in range(n_samples):
        score = 0.0
        if 100 <= temperatures[i] <= 160:
            score += 2.5
        if 5 <= modulator_eqs[i] <= 30:
            score += 2.0
        ratio = metal_concs[i] / (ligand_concs[i] + 1e-5)
        if 0.8 <= ratio <= 1.2:
            score += 2.0
        if 2.0 <= phs[i] <= 5.0:
            score += 1.5
            
        noise = np.random.normal(0, 1.0)
        final_score = score + noise
        
        if final_score > 5.5:
            esiti.append(2)
        elif final_score > 3.0:
            esiti.append(1)
        else:
            esiti.append(0)
            
    df_gen = pd.DataFrame({
        'Temperatura_C': np.round(temperatures, 1),
        'Tempo_h': np.round(times, 1),
        'Conc_Metallo_M': np.round(metal_concs, 3),
        'Conc_Legante_M': np.round(ligand_concs, 3),
        'Eq_Modulatore': np.round(modulator_eqs, 1),
        'pH_Apparente': np.round(phs, 1),
        'Velocita_Agitazione_RPM': stirring_speeds,
        'Tipo_Metallo': metals,
        'Solvente_Principale': solvents,
        'Target_Esito_Classe': esiti
    })
    return df_gen

# ==========================================
# 3. PREPROCESSING E FEATURE ENGINEERING
# ==========================================
def process_unified_dataset(raw_df):
    """Esegue pulizia, feature engineering e encoding categoriale del dataset."""
    df = raw_df.copy()
    df.columns = df.columns.str.strip()
    
    target_col = None
    possible_targets = ['Target_Esito_Classe', 'Esito', 'Outcome', 'Synthesis_Success', 'Result', 'Target']
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
            
    if target_col is None:
        target_col = df.columns[-1]
        
    df = df.rename(columns={target_col: 'Target_Esito_Classe'})
    
    # Rimuovi eventuali righe dove il target è nullo/NaN
    df = df.dropna(subset=['Target_Esito_Classe'])
    
    # Feature Engineering Chimico-Fisico
    if 'Conc_Metallo_M' in df.columns and 'Conc_Legante_M' in df.columns:
        df['Rapporto_Metallo_Legante'] = df['Conc_Metallo_M'] / (df['Conc_Legante_M'] + 1e-5)
    
    if 'Temperatura_C' in df.columns and 'Tempo_h' in df.columns:
        df['Energia_Termica_Effettiva'] = df['Temperatura_C'] * np.log1p(df['Tempo_h'])
        
    feature_cols = [c for c in df.columns if c != 'Target_Esito_Classe']
    encoded_df = pd.get_dummies(df[feature_cols], drop_first=True)
    encoded_df['Target_Esito_Classe'] = df['Target_Esito_Classe']
    
    return encoded_df

# ==========================================
# 4. CARICAMENTO O ADDESTRAMENTO MODELLO
# ==========================================
@st.cache_resource
def load_or_train_model():
    pkl_file = "modello_sintesi_mof_ottimizzato.pkl"
    csv_file = "Dataset_Sintesi_Unificato.csv"
    
    if os.path.exists(pkl_file):
        try:
            saved_data = joblib.load(pkl_file)
            if isinstance(saved_data, dict):
                return saved_data['model'], saved_data['features'], saved_data.get('metrics', {}), saved_data.get('importances', [])
            return saved_data, getattr(saved_data, 'feature_names_in_', []), {}, getattr(saved_data, 'feature_importances_', [])
        except Exception:
            pass
            
    if os.path.exists(csv_file):
        raw_df = pd.read_csv(csv_file)
    else:
        raw_df = generate_synthetic_mof_dataset()
        raw_df.to_csv(csv_file, index=False)
        
    df = process_unified_dataset(raw_df)
    
    X = df.drop(columns=['Target_Esito_Classe'])
    X = X.fillna(X.mean()).fillna(0)
    
    # PULIZIA SICURA TARGET (Conversione priva di errori NaN)
    y_raw = pd.to_numeric(df['Target_Esito_Classe'], errors='coerce')
    valid_mask = y_raw.notna()
    X = X[valid_mask]
    y = y_raw[valid_mask].astype(int)
    
    # Gestione di sicurezza per classi con campioni insufficienti (< 3)
    counts = y.value_counts()
    for cls, count in counts.items():
        if count < 3:
            needed = 3 - count
            rows_to_dup = X[y == cls]
            if len(rows_to_dup) > 0:
                for _ in range(needed):
                    X = pd.concat([X, rows_to_dup.iloc[[0]]], ignore_index=True)
                    y = pd.concat([y, pd.Series([cls])], ignore_index=True)

    # Modello di Base LightGBM
    base_model = LGBMClassifier(
        n_estimators=180,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    min_class_samples = y.value_counts().min()
    cv_splits = min(3, max(2, min_class_samples))
    
    # Calibrazione delle probabilità
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method='sigmoid',
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    )
    
    calibrated_model.fit(X, y)
    
    feature_names = X.columns.tolist()
    try:
        importances = np.mean([est.estimator.feature_importances_ for est in calibrated_model.calibrated_classifiers_], axis=0)
    except Exception:
        importances = np.zeros(len(feature_names))
        
    metrics = {
        'train_accuracy': accuracy_score(y, calibrated_model.predict(X)),
        'n_samples': len(X),
        'n_features': len(feature_names)
    }
    
    save_dict = {
        'model': calibrated_model,
        'features': feature_names,
        'importances': importances,
        'metrics': metrics
    }
    
    joblib.dump(save_dict, pkl_file)
    return calibrated_model, feature_names, metrics, importances

# Caricamento Modello
model, features, metrics, feature_importances = load_or_train_model()

# ==========================================
# 5. SIDEBAR: INPUT PARAMETRI REAZIONE
# ==========================================
st.sidebar.header("⚙️ Parametri Reazione MOF")

def user_input_features():
    inputs = {}
    
    st.sidebar.subheader("🌡️ Condizioni Termodinamiche")
    temp = st.sidebar.slider("Temperatura (°C)", 20.0, 250.0, 130.0, 5.0)
    time_h = st.sidebar.slider("Tempo di Reazione (h)", 1.0, 120.0, 24.0, 1.0)
    stirring = st.sidebar.select_slider("Velocità Agitazione (RPM)", options=[0, 200, 500, 800], value=0)
    
    st.sidebar.subheader("⚖️ Stechiometria e Soluzione")
    metal_conc = st.sidebar.number_input("Conc. Metallo (M)", 0.001, 2.0, 0.10, 0.01)
    ligand_conc = st.sidebar.number_input("Conc. Legante (M)", 0.001, 2.0, 0.10, 0.01)
    modulator_eq = st.sidebar.slider("Equivalenti Modulatore", 0.0, 100.0, 15.0, 1.0)
    ph = st.sidebar.slider("pH apparente", 0.0, 14.0, 3.5, 0.1)
    
    st.sidebar.subheader("🧪 Componenti Chimici")
    metal_type = st.sidebar.selectbox("Tipo Metallo", ["Zr", "Cu", "Zn", "Fe", "Al", "UiO-Metal"])
    solvent = st.sidebar.selectbox("Solvente Principale", ["DMF", "DEF", "H2O", "Ethanol", "DMF/H2O"])
    
    st.sidebar.subheader("🧬 Struttura Molecolare Legante")
    smiles_input = st.sidebar.text_input("SMILES Legante", "c1cc(C(=O)O)cc(C(=O)O)c1")

    for f in features:
        f_lower = f.lower()
        if 'temp' in f_lower and 'energia' not in f_lower:
            inputs[f] = temp
        elif 'tempo' in f_lower or 'time' in f_lower:
            inputs[f] = time_h
        elif 'metal' in f_lower and 'conc' in f_lower:
            inputs[f] = metal_conc
        elif 'legan' in f_lower and 'conc' in f_lower:
            inputs[f] = ligand_conc
        elif 'modulat' in f_lower or 'eq' in f_lower:
            inputs[f] = modulator_eq
        elif 'ph' in f_lower:
            inputs[f] = ph
        elif 'rpm' in f_lower or 'agita' in f_lower:
            inputs[f] = stirring
        elif 'rapporto' in f_lower:
            inputs[f] = metal_conc / (ligand_conc + 1e-5)
        elif 'energia' in f_lower:
            inputs[f] = temp * np.log1p(time_h)
        elif 'tipo_metallo_' + metal_type in f:
            inputs[f] = 1.0
        elif 'solvente_principale_' + solvent in f:
            inputs[f] = 1.0
        else:
            inputs[f] = 0.0
            
    return pd.DataFrame([inputs]), smiles_input, temp, modulator_eq, metal_conc, ligand_conc, ph, metal_type, solvent

input_df, smiles_str, current_temp, current_mod, current_m_conc, current_l_conc, current_ph, current_metal, current_solvent = user_input_features()

# ==========================================
# 6. FUNZIONI DI CALCOLO E DESCRITTORI MOLECOLARI
# ==========================================
def calculate_molecular_descriptors(smiles):
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            'Peso Molecolare (g/mol)': np.round(Descriptors.MolWt(mol), 2),
            'LogP (Lipofilia)': np.round(Descriptors.MolLogP(mol), 2),
            'H-Bond Donors': Descriptors.NumHDonors(mol),
            'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
            'TPSA (Å²)': np.round(Descriptors.TPSA(mol), 2),
            'Legami Ruotabili': Descriptors.NumRotatableBonds(mol)
        }
    except Exception:
        return None

# ==========================================
# 7. INTERFACCIA A SCHEDE (TABS)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predizione & Diagnostica", 
    "🧬 Descrittori & Molecola 3D", 
    "🗺️ Mappa di Cristalizzazione 2D", 
    "💡 Ottimizzatore & Ricetta Inversa",
    "📊 Info Dataset & Prestazioni"
])

# ------------------------------------------
# TAB 1: PREDIZIONE & DIAGNOSTICA
# ------------------------------------------
with tab1:
    st.subheader("📊 Esito Predetto della Sintesi MOF")
    
    probs = model.predict_proba(input_df)[0]
    classes = getattr(model, 'classes_', np.array([0, 1, 2]))
    
    class_names = {0: "Amorfo / Precipitato", 1: "Parziale / Miscela", 2: "Cristallino (Successo)"}
    colors = {0: "#EF4444", 1: "#F59E0B", 2: "#10B981"}
    
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    for i, cls in enumerate(classes):
        prob_pct = probs[i] * 100
        name = class_names.get(cls, f"Classe {cls}")
        with cols[i % 3]:
            st.metric(label=name, value=f"{prob_pct:.1f}%")
            st.progress(float(probs[i]))

    st.markdown("---")
    
    col_chart, col_explain = st.columns([1, 1])
    
    with col_chart:
        st.subheader("🥧 Distribuzione Probabilità")
        labels = [class_names.get(c, f"Classe {c}") for c in classes]
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels, 
            values=probs, 
            hole=.4,
            marker_colors=[colors.get(c, '#9CA3AF') for c in classes]
        )])
        fig_donut.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with col_explain:
        st.subheader("💡 Fattori Determinanti (Feature Importance)")
        if len(feature_importances) > 0:
            imp_df = pd.DataFrame({
                'Parametro': features,
                'Importanza': feature_importances
            }).sort_values('Importanza', ascending=True).tail(8)
            
            fig_bar = px.bar(imp_df, x='Importanza', y='Parametro', orientation='h',
                             color='Importanza', color_continuous_scale='Viridis')
            fig_bar.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------
# TAB 2: DESCRITTORI & MOLECOLA 2D/3D
# ------------------------------------------
with tab2:
    st.subheader("🧬 Analisi Molecolare del Legante Organico")
    
    descriptors = calculate_molecular_descriptors(smiles_str)
    
    if descriptors:
        st.write("### 📐 Descrittori Fisico-Chimici Calcolati")
        d_cols = st.columns(len(descriptors))
        for idx, (k, v) in enumerate(descriptors.items()):
            d_cols[idx].metric(k, str(v))
        st.markdown("---")
        
    col_2d, col_3d = st.columns(2)
    
    with col_2d:
        st.write("### 🖼️ Struttura 2D (RDKit)")
        if RDKIT_AVAILABLE and smiles_str:
            try:
                mol = Chem.MolFromSmiles(smiles_str)
                if mol:
                    img = Draw.MolToImage(mol, size=(400, 350))
                    st.image(img, caption=f"SMILES: {smiles_str}")
                else:
                    st.error("SMILES non valido.")
            except Exception as e:
                st.error(f"Errore generazione 2D: {e}")
        else:
            st.info("RDKit non disponibile o SMILES non specificato.")

    with col_3d:
        st.write("### 🧊 Visualizzazione 3D Interattiva")
        if PY3DMOL_AVAILABLE and RDKIT_AVAILABLE and smiles_str:
            try:
                mol = Chem.MolFromSmiles(smiles_str)
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    mblock = Chem.MolToMolBlock(mol)
                    
                    xyzview = py3Dmol.view(width=400, height=350)
                    xyzview.addModel(mblock, 'mol')
                    xyzview.setStyle({'stick': {}})
                    xyzview.zoomTo()
                    showmol(xyzview, height=350, width=400)
            except Exception:
                st.warning("Impossibile generare la conformazione 3D per questa molecola.")
        else:
            st.info("Installa `stmol` e `py3Dmol` per la rendering 3D interattivo.")

# ------------------------------------------
# TAB 3: MAPPA DI CRISTALIZZAZIONE 2D
# ------------------------------------------
with tab3:
    st.subheader("🗺️ Mappa di Cristalizzazione 2D (Isola di Sintesi)")
    st.write("Visualizzazione del profilo di risposta variando **Temperatura** e **Modulatore** a parità di altre condizioni.")
    
    temp_range = np.linspace(40, 220, 30)
    mod_range = np.linspace(0, 60, 30)
    
    grid_z = np.zeros((len(temp_range), len(mod_range)))
    sample_row = input_df.copy()
    
    temp_cols = [c for c in features if 'temp' in c.lower() and 'energia' not in c.lower()]
    mod_cols = [c for c in features if 'modulat' in c.lower() or 'eq' in c.lower()]
    
    if temp_cols and mod_cols:
        t_col = temp_cols[0]
        m_col = mod_cols[0]
        
        for i, t in enumerate(temp_range):
            for j, m in enumerate(mod_range):
                sample_row[t_col] = t
                sample_row[m_col] = m
                if 'Energia_Termica_Effettiva' in sample_row.columns:
                    sample_row['Energia_Termica_Effettiva'] = t * np.log1p(sample_row.get('Tempo_h', 24))
                
                p_succ = model.predict_proba(sample_row)[0]
                idx_succ = np.where(classes == 2)[0]
                grid_z[i, j] = p_succ[idx_succ[0]] if len(idx_succ) > 0 else p_succ[-1]
                
        fig_contour = go.Figure(data=go.Contour(
            z=grid_z,
            x=mod_range,
            y=temp_range,
            colorscale='Viridis',
            colorbar=dict(title='Prob. Cristallinità')
        ))
        
        fig_contour.add_trace(go.Scatter(
            x=[current_mod],
            y=[current_temp],
            mode='markers+text',
            marker=dict(color='red', size=14, symbol='x'),
            text=["Punto Selezionato"],
            textposition="top center",
            name="Condizioni Attuali"
        ))
        
        fig_contour.update_layout(
            xaxis_title="Equivalenti Modulatore",
            yaxis_title="Temperatura (°C)",
            height=500
        )
        st.plotly_chart(fig_contour, use_container_width=True)
    else:
        st.warning("Impossibile generare la mappa: parametri termodinamici non mappati correttamente.")

# ------------------------------------------
# TAB 4: OTTIMIZZATORE & RICETTA INVERSA
# ------------------------------------------
with tab4:
    st.subheader("💡 Ricerca delle Condizioni Ottimali (Inverse Design)")
    st.write("Genera una ricetta consigliata per massimizzare la probabilità di successo cristallino.")
    
    if st.button("🚀 Avvia Ottimizzazione Sintetica"):
        with st.spinner("Ricerca nello spazio delle condizioni di reazione..."):
            best_prob = -1.0
            best_params = None
            
            temps_test = np.linspace(80, 180, 10)
            mods_test = np.linspace(5, 40, 10)
            phs_test = np.linspace(2.0, 6.0, 5)
            
            test_row = input_df.copy()
            
            for t in temps_test:
                for m in mods_test:
                    for p in phs_test:
                        for col in test_row.columns:
                            if 'temp' in col.lower() and 'energia' not in col.lower():
                                test_row[col] = t
                            elif 'modulat' in col.lower() or 'eq' in col.lower():
                                test_row[col] = m
                            elif 'ph' in col.lower():
                                test_row[col] = p
                                
                        p_succ = model.predict_proba(test_row)[0]
                        idx_succ = np.where(classes == 2)[0]
                        prob = p_succ[idx_succ[0]] if len(idx_succ) > 0 else p_succ[-1]
                        
                        if prob > best_prob:
                            best_prob = prob
                            best_params = {'Temperatura': t, 'Modulatore Eq.': m, 'pH': p}
                            
            st.success(f"✅ Condizione Ottimale Trovata con Probabilità di Cristallinità: **{best_prob*100:.1f}%**")
            
            o_col1, o_col2, o_col3 = st.columns(3)
            o_col1.metric("Temperatura Consigliata", f"{best_params['Temperatura']:.1f} °C")
            o_col2.metric("Modulatore Consigliato", f"{best_params['Modulatore Eq.']:.1f} eq")
            o_col3.metric("pH Consigliato", f"{best_params['pH']:.1f}")

# ------------------------------------------
# TAB 5: INFO DATASET & PRESTAZIONI
# ------------------------------------------
with tab5:
    st.subheader("📊 Diagnostica Modello e Stato Dataset")
    
    st.write("### 📈 Metriche dell'Addestramento")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Campioni Totali Dataset", metrics.get('n_samples', 'N/A'))
    m_col2.metric("Numero Feature Processate", metrics.get('n_features', 'N/A'))
    m_col3.metric("Accuratezza Addestramento", f"{metrics.get('train_accuracy', 0)*100:.1f}%")
    
    st.markdown("---")
    st.write("### 📂 Anteprima Dataset Unificato")
    csv_file = "Dataset_Sintesi_Unificato.csv"
    if os.path.exists(csv_file):
        df_preview = pd.read_csv(csv_file)
        st.dataframe(df_preview.head(15), use_container_width=True)
    else:
        st.info("Nessun CSV salvato su disco. Viene utilizzato il dataset generato dinamico.")
