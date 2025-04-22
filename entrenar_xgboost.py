import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from codigo import nuevo_dataset

# 1. Separar variables predictoras y objetivo
X = nuevo_dataset.drop(columns=["ID_GRD"])
y = nuevo_dataset["ID_GRD"]

# 2. Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 4. Definir y entrenar el modelo XGBoost
xgb_model = XGBClassifier(
    objective="multi:softmax",
    num_class=len(np.unique(y_train)),
    use_label_encoder=False,
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 5. Predicciones
y_pred = xgb_model.predict(X_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)
y_test_decoded = label_encoder.inverse_transform(y_test)

# 6. Guardar modelo y codificador
os.makedirs("modelos", exist_ok=True)
joblib.dump((xgb_model, label_encoder), "modelos/xgboost_entrenado.pkl")

# 7. Guardar matriz de confusión
os.makedirs("imagenes", exist_ok=True)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test_decoded, y_pred_decoded), cmap="Blues", cbar=False)
plt.title("Matriz de Confusión - XGBoost")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("imagenes/confusion_xgboost_entrenado.png")
plt.close()

# 8. Reporte de clasificación
print("\n📊 Clasification Report (GRD codificado):")
print(classification_report(y_test_decoded, y_pred_decoded))

# 9. Métricas
acc = accuracy_score(y_test_decoded, y_pred_decoded)
prec = precision_score(y_test_decoded, y_pred_decoded, average="macro", zero_division=0)
rec = recall_score(y_test_decoded, y_pred_decoded, average="macro", zero_division=0)
f1 = f1_score(y_test_decoded, y_pred_decoded, average="macro", zero_division=0)
mae = mean_absolute_error(y_test_decoded, y_pred_decoded)
mse = mean_squared_error(y_test_decoded, y_pred_decoded)

print("\n✅ RESUMEN XGBOOST ENTRENADO:")
print(f"🎯 Accuracy: {round(acc, 4)}")
print(f"📈 Precision: {round(prec, 4)}")
print(f"📈 Recall: {round(rec, 4)}")
print(f"📈 F1 Score: {round(f1, 4)}")
print(f"📉 MAE: {round(mae, 4)}")
print(f"📉 MSE: {round(mse, 4)}")
