import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# 1. Carica il dataset encodato
df = pd.read_csv("Dataset_Sintesi_ML_Encoded.csv")

# 2. Prepara le variabili
drop_cols = ['ID', 'Sorgente_Database', 'Metallo', 'Sale metallico', 'Legante standard', 
             'SMILES_Legante', 'Solvente', 'Solvente_Clean', 'Stato XRD', 
             'Esito/Osservazioni', 'Anione_Tipo', 'Target_Esito_Classe']

features = [c for c in df.columns if c not in drop_cols]
X = df[features].fillna(df[features].mean())
y = df['Target_Esito_Classe'].astype(int)

# 3. Addestra il modello Gradient Boosting
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)
model.fit(X, y)

# 4. Salva il file .pkl
joblib.dump(model, "modello_sintesi_mof_ottimizzato.pkl")
print("✅ File 'modello_sintesi_mof_ottimizzato.pkl' creato con successo nella cartella!")
