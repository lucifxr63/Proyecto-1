import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter

# Cargar desde el archivo principal
from codigo import df, y, y_balanced, num_cols

# Asegurar carpeta de salida
os.makedirs("Graficos", exist_ok=True)

# 1. Distribución de GRDs antes del balanceo
plt.figure(figsize=(12, 5))
df['ID_GRD'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Distribución de GRDs antes del balanceo")
plt.xlabel("ID_GRD")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("Graficos/distribucion_grds_original.png")
plt.close()

# 2. Distribución de GRDs después del balanceo
plt.figure(figsize=(12, 5))
plt.bar(Counter(y_balanced).keys(), Counter(y_balanced).values(), color='lightgreen')
plt.title("Distribución de GRDs después del balanceo")
plt.xlabel("ID_GRD")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("Graficos/distribucion_grds_balanceado.png")
plt.close()

# 3. Matriz de correlación (si hay variables numéricas)
if len(num_cols) > 1:
    plt.figure(figsize=(10, 8))
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlación de Variables Numéricas")
    plt.tight_layout()
    plt.savefig("Graficos/matriz_correlacion.png")
    plt.close()

# 4. Distribución por sexo
if 'Sexo (Desc)' in df.columns:
    plt.figure(figsize=(6, 4))
    df['Sexo (Desc)'].value_counts().plot(kind='bar', color=['coral', 'skyblue'])
    plt.title("Distribución por Sexo")
    plt.xlabel("Sexo")
    plt.ylabel("Cantidad")
    plt.xticks(ticks=[0, 1], labels=["Mujer", "Hombre"], rotation=0)
    plt.tight_layout()
    plt.savefig("Graficos/distribucion_sexo.png")
    plt.close()

# 5. Boxplots de variables numéricas (primeras 5)
for col in num_cols[:5]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.savefig(f"Graficos/boxplot_{col}.png")
    plt.close()

# 6. Histogramas de variables numéricas (primeras 5)
for col in num_cols[:5]:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True, color="skyblue", bins=30)
    plt.title(f"Distribución de: {col}")
    plt.tight_layout()
    plt.savefig(f"Graficos/hist_{col}.png")
    plt.close()

# 7. Distribución de los GRDs más frecuentes (≥ 100)
frecuencia = df["ID_GRD"].value_counts().to_dict()
moda = {k: v for k, v in frecuencia.items() if v >= 100}
nuevos_ids = {k: i+1 for i, k in enumerate(moda.keys())}
df_moda = df[df["ID_GRD"].isin(moda.keys())].copy()
df_moda["ID_GRD_mapeado"] = df_moda["ID_GRD"].map(nuevos_ids)

plt.figure(figsize=(10, 6))
sns.histplot(df_moda["ID_GRD_mapeado"], kde=True, bins=40, color="coral", alpha=0.7)
plt.title("Distribución de los GRDs más frecuentes (≥ 100 casos)")
plt.xlabel("ID_GRD (re-mapeado)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.savefig("Graficos/distribucion_grds_frecuentes.png")
plt.close()

print("✅ Todos los gráficos han sido generados correctamente en la carpeta 'Graficos'.")
