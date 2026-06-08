import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Carregar os dados com o separador correto (ponto e vírgula)
df = pd.read_csv('default_of_credit_card_clients.csv', sep=';')
print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas")

# 2. Verificar as colunas
print(f"Colunas disponíveis: {list(df.columns)}")

# 3. Separar X e y (a coluna alvo é 'default payment next month')
X = df.drop(['ID', 'default payment next month'], axis=1)
y = df['default payment next month']

print(f"Features (X): {X.shape[1]} colunas")
print(f"Target (y): {y.name}")

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Pipeline de transformação para colunas categóricas
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']

# Verificar quais colunas categóricas existem
existing_categorical = [col for col in categorical_cols if col in X.columns]
print(f"Colunas categóricas encontradas: {existing_categorical}")

# Criar transformador
transformador = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), existing_categorical),
    remainder=StandardScaler()
)

# 6. Criar os modelos
modelo_rf = make_pipeline(transformador, RandomForestClassifier(random_state=42, n_estimators=100))
modelo_lr = make_pipeline(transformador, LogisticRegression(random_state=42, max_iter=1000))

# 7. Treinar e comparar
print("\nTreinando modelos...")
modelo_rf.fit(X_train, y_train)
modelo_lr.fit(X_train, y_train)

# 8. Avaliar
nota_rf = accuracy_score(y_test, modelo_rf.predict(X_test))
nota_lr = accuracy_score(y_test, modelo_lr.predict(X_test))

print(f"\nResultados:")
print(f"Acerto da Random Forest: {nota_rf * 100:.2f}%")
print(f"Acerto da Regressão Logística: {nota_lr * 100:.2f}%")

# 9. Salvar o melhor modelo
if nota_rf > nota_lr:
    print("\nRandom Forest venceu! Salvando modelo final...")
    joblib.dump(modelo_rf, 'modelo_final.pkl')
else:
    print("\nRegressão Logística venceu! Salvando modelo final...")
    joblib.dump(modelo_lr, 'modelo_final.pkl')

print("Modelo salvo como 'modelo_final.pkl'")