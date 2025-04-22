import pandas as pd
import joblib
import numpy as np
from codigo import nuevo_dataset
from sklearn.model_selection import train_test_split

# Cargar modelo y codificador
xgb_model, label_encoder = joblib.load("modelos/xgboost_entrenado.pkl")

# Preparar datos
X = nuevo_dataset.drop(columns=["ID_GRD"])
y = nuevo_dataset["ID_GRD"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Codificar y predecir
y_test_encoded = label_encoder.transform(y_test)
pred_encoded = xgb_model.predict(X_test)
y_pred = label_encoder.inverse_transform(pred_encoded)

# Mostrar resultados
df_resultados = pd.DataFrame({
    "GRD Real": y_test.values,
    "GRD Predicho": y_pred
}).reset_index(drop=True)

print(df_resultados.head(10))

# Guardar en CSV
df_resultados.to_csv("predicciones_xgboost.csv", index=False)
