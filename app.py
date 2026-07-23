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
        
        # Gestione dei dati mancanti (NaN)
        X = df.drop(columns=['Target_Esito_Classe'])
        X = X.fillna(X.mean()).fillna(0)  # Riempie eventuali celle vuote con la media
        y = df['Target_Esito_Classe'].fillna(0).astype(int)
        
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)
        model.fit(X, y)
        joblib.dump(model, pkl_file)
        return model
    else:
        st.error(f"File '{csv_file}' non trovato! Assicurati di aver caricato il file su GitHub.")
        st.stop()
