import pandas as pd
import joblib
from codigo import nuevo_dataset
import numpy as np

# Cargar modelo y codificador
xgb_model, label_encoder = joblib.load("modelos/xgboost_entrenado.pkl")

# Cargar tabla maestra de GRD
grd_maestra = pd.read_excel("data/Tablas maestras bases GRD.xlsx", sheet_name="IR - GRD")
grd_maestra = grd_maestra.dropna(subset=[grd_maestra.columns[0]]).drop_duplicates()

# Crear diccionarios: ID → Código → Descripción
codigo_to_id = {}
id_to_codigo = {}
id_to_desc = {}

for idx, row in enumerate(grd_maestra.itertuples(index=False), start=1):
    codigo = str(row[0]).strip()
    descripcion = str(row[1]).strip() if not pd.isna(row[1]) else "(sin descripción)"
    codigo_to_id[codigo] = idx
    id_to_codigo[idx] = codigo
    id_to_desc[idx] = descripcion

# Elegir muestra para predecir
sample = nuevo_dataset.sample(n=10, random_state=42)
X_sample = sample.drop(columns=["ID_GRD"])
y_real = sample["ID_GRD"]

# Predicción
y_real_encoded = label_encoder.transform(y_real)
y_pred_encoded = xgb_model.predict(X_sample)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Crear DataFrame con resultados
comparacion = pd.DataFrame({
    "✔️": ["✅" if real == pred else "❌" for real, pred in zip(y_real, y_pred)],
    "GRD Real": y_real.values,
    "GRD Predicho": y_pred,
    "Descripción Real": [id_to_desc.get(real, "(sin descripción)") for real in y_real],
    "Descripción Predicha": [id_to_desc.get(pred, "(sin descripción)") for pred in y_pred]
})

# Mostrar resultados
print("\n📋 COMPARACIÓN GRD REAL vs PREDICHO:")
print(comparacion.to_string(index=False))

# Guardar en Excel si se desea
comparacion.to_excel("resultados/comparacion_grd.xlsx", index=False)
print("\n📁 Resultados guardados en: resultados/comparacion_grd.xlsx")
