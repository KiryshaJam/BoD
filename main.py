import pandas as pd
import numpy as np
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# Настройки

DATA_PATH = "data/Educacao_Basica_2018 - Docentes_Sudeste.csv"
TARGET = "TP_ESCOLARIDADE"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_FEATURES = 10


# Загрузка данных
df = pd.read_csv(DATA_PATH, sep="|")
print("Исходный размер данных:", df.shape)



# Очистка данных
id_cols = [col for col in df.columns if col.startswith("ID_")]
df = df.drop(columns=id_cols, errors="ignore")
print("Удалено ID-столбцов:", len(id_cols))

leakage_cols = [
    "IN_ESPECIALIZACAO", "IN_MESTRADO", "IN_DOUTORADO", "IN_POS_NENHUM",
    "TP_SITUACAO_CURSO_1", "TP_SITUACAO_CURSO_2", "TP_SITUACAO_CURSO_3",
    "CO_CURSO_1", "CO_CURSO_2", "CO_CURSO_3",
    "CO_AREA_CURSO_1", "CO_AREA_CURSO_2", "CO_AREA_CURSO_3",
    "IN_LICENCIATURA_1", "IN_LICENCIATURA_2", "IN_LICENCIATURA_3",
    "IN_COM_PEDAGOGICA_1", "IN_COM_PEDAGOGICA_2", "IN_COM_PEDAGOGICA_3",
    "TP_TIPO_IES_1", "TP_TIPO_IES_2", "TP_TIPO_IES_3",
    "CO_IES_1", "CO_IES_2", "CO_IES_3",
    "NU_ANO_INICIO_1", "NU_ANO_CONCLUSAO_1",
    "NU_ANO_INICIO_2", "NU_ANO_CONCLUSAO_2",
    "NU_ANO_INICIO_3", "NU_ANO_CONCLUSAO_3",
]
existing_leakage_cols = [col for col in leakage_cols if col in df.columns]
df = df.drop(columns=existing_leakage_cols, errors="ignore")
print("Удалено признаков с потенциальной утечкой:", len(existing_leakage_cols))

empty_cols = [col for col in df.columns if df[col].isna().all()]
df = df.drop(columns=empty_cols, errors="ignore")
print("Удалено пустых столбцов:", len(empty_cols))

const_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
df = df.drop(columns=const_cols, errors="ignore")
print("Удалено константных столбцов:", len(const_cols))

before_rows = df.shape[0]
df = df.dropna(subset=[TARGET])
after_rows = df.shape[0]
print("Удалено строк без таргета:", before_rows - after_rows)

print("Размер данных после очистки:", df.shape)



# Подготовка данных
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET]).copy()

for col in X.columns:
    X[col] = X[col].fillna(X[col].median())

print("Количество классов:", y.nunique())
print("Распределение классов:")
print(y.value_counts().sort_index())



# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)


# Обучение CatBoost
start = time.time()

cat_model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    auto_class_weights="Balanced",
    random_seed=RANDOM_STATE,
    verbose=50
)

cat_model.fit(X_train, y_train)
cat_time = time.time() - start



# Обучение LightGBM
start = time.time()

lgb_model = LGBMClassifier(
    objective="multiclass",
    num_class=y.nunique(),
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start



# Предсказания
pred_cat = np.array(cat_model.predict(X_test)).reshape(-1)
pred_lgb = np.array(lgb_model.predict(X_test)).reshape(-1)



# Метрики
acc_cat = accuracy_score(y_test, pred_cat)
f1_cat_weighted = f1_score(y_test, pred_cat, average="weighted")
f1_cat_macro = f1_score(y_test, pred_cat, average="macro")

acc_lgb = accuracy_score(y_test, pred_lgb)
f1_lgb_weighted = f1_score(y_test, pred_lgb, average="weighted")
f1_lgb_macro = f1_score(y_test, pred_lgb, average="macro")

print("\n=== ИТОГОВЫЕ МЕТРИКИ ===")
print(f"CatBoost  | Accuracy: {acc_cat:.4f} | F1_weighted: {f1_cat_weighted:.4f} | F1_macro: {f1_cat_macro:.4f} | Time: {cat_time:.2f} sec")
print(f"LightGBM  | Accuracy: {acc_lgb:.4f} | F1_weighted: {f1_lgb_weighted:.4f} | F1_macro: {f1_lgb_macro:.4f} | Time: {lgb_time:.2f} sec")



# Отчёты по классам
print("\n=== CLASSIFICATION REPORT: CATBOOST ===")
print(classification_report(y_test, pred_cat))

print("\n=== CLASSIFICATION REPORT: LIGHTGBM ===")
print(classification_report(y_test, pred_lgb))



# Матрицы ошибок
print("\n=== CONFUSION MATRIX: CATBOOST ===")
print(confusion_matrix(y_test, pred_cat))

print("\n=== CONFUSION MATRIX: LIGHTGBM ===")
print(confusion_matrix(y_test, pred_lgb))



# Важность признаков
cat_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": cat_model.get_feature_importance()
}).sort_values(by="importance", ascending=False)

lgb_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": lgb_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(f"\n=== TOP-{TOP_FEATURES} FEATURES: CATBOOST ===")
print(cat_importance.head(TOP_FEATURES).to_string(index=False))

print(f"\n=== TOP-{TOP_FEATURES} FEATURES: LIGHTGBM ===")
print(lgb_importance.head(TOP_FEATURES).to_string(index=False))