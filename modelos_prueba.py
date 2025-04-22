import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from codigo import nuevo_dataset

# === Preparación de los datos ===
# Separamos las variables independientes (X) de la variable objetivo (y)
X = nuevo_dataset.drop(columns=["ID_GRD"])
y = nuevo_dataset["ID_GRD"]

# División de datos en entrenamiento y prueba (80% - 20%) con estratificación por clase
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Crear carpetas donde se guardarán imágenes y modelos
os.makedirs("imagenes", exist_ok=True)
os.makedirs("modelos", exist_ok=True)

# === Modelo Random Forest ===
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)
pred_rf = modelo_rf.predict(X_test)
joblib.dump(modelo_rf, "modelos/random_forest_model.pkl")

# === Árbol de Decisión ===
modelo_dt = DecisionTreeClassifier(max_depth=10, random_state=42)
modelo_dt.fit(X_train, y_train)
pred_dt = modelo_dt.predict(X_test)
joblib.dump(modelo_dt, "modelos/decision_tree_model.pkl")

# === Modelo XGBoost ===
# XGBoost requiere que las clases estén codificadas como enteros consecutivos
le = LabelEncoder()
y_train_xgb = le.fit_transform(y_train)
y_test_xgb = le.transform(y_test)

modelo_xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=len(np.unique(y_train_xgb)),
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42
)
modelo_xgb.fit(X_train, y_train_xgb)
pred_xgb_codificado = modelo_xgb.predict(X_test)
pred_xgb = le.inverse_transform(pred_xgb_codificado)
# Guardamos el modelo junto al codificador de etiquetas
joblib.dump((modelo_xgb, le), "modelos/xgboost_model.pkl")

# === Red Neuronal Multicapa (MLP) ===
modelo_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
modelo_mlp.fit(X_train, y_train)
pred_mlp = modelo_mlp.predict(X_test)
joblib.dump(modelo_mlp, "modelos/mlp_model.pkl")

# === Función para imprimir métricas de evaluación ===
def imprimir_resumen(nombre, y_real, y_predicho):
    acc = accuracy_score(y_real, y_predicho)
    prec = precision_score(y_real, y_predicho, average="macro", zero_division=0)
    rec = recall_score(y_real, y_predicho, average="macro", zero_division=0)
    f1 = f1_score(y_real, y_predicho, average="macro", zero_division=0)
    mae = mean_absolute_error(y_real, y_predicho)
    mse = mean_squared_error(y_real, y_predicho)

    print(f"\n📌 RESUMEN DEL MODELO {nombre.upper()}:")
    print(f"🎯 Exactitud (Accuracy): {round(acc, 4)}")
    print(f"📈 Precisión: {round(prec, 4)}")
    print(f"📈 Recall: {round(rec, 4)}")
    print(f"📈 F1 Score: {round(f1, 4)}")
    print(f"📉 Error absoluto medio (MAE): {round(mae, 4)}")
    print(f"📉 Error cuadrático medio (MSE): {round(mse, 4)}")

# === Evaluación y visualización de resultados ===
resultados_modelos = {
    "Random Forest": pred_rf,
    "Árbol de Decisión": pred_dt,
    "XGBoost": pred_xgb,
    "Red Neuronal": pred_mlp
}

for nombre, pred in resultados_modelos.items():
    imprimir_resumen(nombre, y_test, pred)

    # Generación de la matriz de confusión y guardado como imagen
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, pred), cmap="Blues", cbar=False)
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.tight_layout()
    nombre_archivo = nombre.lower().replace(" ", "_").replace("á", "a")
    plt.savefig(f"imagenes/confusion_{nombre_archivo}.png")
    plt.close()
