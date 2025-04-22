import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler

# Cargar archivos principales
df = pd.read_csv("data/dataset_elpino.csv", delimiter=";", on_bad_lines="skip")
cie9 = pd.read_excel("data/CIE-9.xlsx", converters={"Código": str})
cie10 = pd.read_excel("data/CIE-10.xlsx")
tmb = pd.read_excel("data/Tablas maestras bases GRD.xlsx", sheet_name="IR - GRD")

# Limpieza básica
df = df.loc[:, df.isnull().mean() < 0.3]
df.drop(columns=["ID", "Fecha ingreso", "Fecha egreso"], inplace=True, errors='ignore')
df = df[df["GRD"].notnull()]
df["GRD_codigo"] = df["GRD"].apply(lambda x: str(x).split(" - ")[0].zfill(5).strip())

# Filtrar GRDs con al menos 20 muestras
frecuentes = df["GRD_codigo"].value_counts()
grds_filtrados = frecuentes[frecuentes >= 20].index
df = df[df["GRD_codigo"].isin(grds_filtrados)]

# Codificar sexo
if "Sexo (Desc)" in df.columns:
    df["Sexo (Desc)"] = df["Sexo (Desc)"].map({"Hombre": 1, "Mujer": 0})

# Codificar diagnósticos
diag_dict = {"-": 0}
for idx, code in enumerate(cie10["Código"].dropna().unique(), start=1):
    diag_dict[code.strip()] = idx

for i, col in enumerate([col for col in df.columns if "Diag" in col], start=1):
    idx = df.columns.get_loc(col)
    codigos = df[col].fillna("-").apply(lambda x: diag_dict.get(x.split("-")[0].strip(), 0))
    df.insert(idx, f"ID_diag_{i}", codigos)
    df.drop(col, axis=1, inplace=True)

# Codificar procedimientos
proc_dict = {"-": 0}
for idx, code in enumerate(cie9["Código"].dropna().unique(), start=1):
    proc_dict[code.strip()] = idx

for i, col in enumerate([col for col in df.columns if "Proced" in col], start=1):
    idx = df.columns.get_loc(col)
    codigos = df[col].fillna("-").apply(lambda x: proc_dict.get(x.split("-")[0].strip(), 0))
    df.insert(idx, f"ID_proc_{i}", codigos)
    df.drop(col, axis=1, inplace=True)

# Codificar GRD usando tabla maestra
grd_dict = {"-": 0}
for idx, code in enumerate(tmb.iloc[:, 0].dropna().astype(str).unique(), start=1):
    grd_dict[code.strip()] = idx

df["ID_GRD"] = df["GRD_codigo"].apply(lambda x: grd_dict.get(x.lstrip("0"), 0))
df.drop(columns=["GRD", "GRD_codigo"], inplace=True)

# Preparación de variables
y = df["ID_GRD"]
X = df.drop(columns=["ID_GRD"])
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocesamiento
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

X_processed = preprocessor.fit_transform(X)

# Balanceo de datos
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X_processed, y)

# División de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)

# Dataset equilibrado final exportable
nuevo_dataset = df.copy()

# Exportables
__all__ = ["df", "y", "y_balanced", "num_cols", "nuevo_dataset"]

print("✅ Datos listos para pruebas de modelos con `nuevo_dataset` disponible.")
