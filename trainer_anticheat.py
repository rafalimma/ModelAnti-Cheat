import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ==========================================
# CONFIGURAÇÃO DE SEGURANÇA (ANTI-CHEAT)
# ==========================================
# Limiar de decisão: 0.85 exige 85% de probabilidade da IA para marcar como Hacker.
# Isso reduz drasticamente os Falsos Positivos (jogadores legítimos banidos por engano).
THRESHOLD_HACKER = 0.85

# ==========================================
# 1. CARREGAMENTO E ROTULAGEM DOS DATASETS
# ==========================================
PATH_NORMAL = 'datasets/normal08-18.csv'
PATH_HACKER = 'datasets/dataset_hacker_entropia.csv'

def carregar_dados():
    if not os.path.exists(PATH_NORMAL) or not os.path.exists(PATH_HACKER):
        raise FileNotFoundError("Verifique se os arquivos de dataset estão na pasta 'datasets/'.")

    df_normal = pd.read_csv(PATH_NORMAL)
    df_hacker = pd.read_csv(PATH_HACKER)

    # Atribuição do Target (Rótulo Real): 0 = Jogador Limpo, 1 = Hacker
    df_normal['is_hacker'] = 0
    df_hacker['is_hacker'] = 1

    # Unificação das duas fontes de dados
    df_total = pd.concat([df_normal, df_hacker], ignore_index=True)
    return df_total

df = carregar_dados()

# ==========================================
# 2. DEFINIÇÃO DAS FEATURES E TARGET
# ==========================================
FEATURES = [
    'vel_posicao', 'acel_linear', 'vel_rotacao', 'jitter_mira', 
    'entropia_mov', 'eficiencia_trajeto', 'travado_em_player', 
    'perseguindo_player', 'mirando'
]

TARGET = 'is_hacker'

X = df[FEATURES]
y = df[TARGET]

# DIVISÃO TEMPORAL (Sem embaralhar)
# Impede que segundos vizinhos da mesma partida vazem entre Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"Total de registros: {len(df)}")
print(f"Base de Treino: {len(X_train)} linhas | Base de Teste: {len(X_test)} linhas\n")

# ==========================================
# 3. TREINAMENTO DA FLORESTA (RANDOM FOREST)
# ==========================================
print("Treinando o modelo Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight='balanced', # Penaliza desequilíbrios na distribuição dos dados
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Treinamento concluído!\n")

# ==========================================
# 4. AVALIAÇÃO COM THRESHOLD AJUSTADO
# ==========================================
# Calculamos a probabilidade contínua (0.0 a 1.0) de cada amostra ser de um Hacker
y_probs = model.predict_proba(X_test)[:, 1]

# Aplicamos a regra rígida (exige >= 85% de certeza)
y_pred = (y_probs >= THRESHOLD_HACKER).astype(int)

print(f"--- RELATÓRIO DE DESEMPENHO (Threshold >= {int(THRESHOLD_HACKER*100)}%) ---")
print(classification_report(y_test, y_pred, target_names=['Legítimo (0)', 'Hacker (1)']))
print(f"Acurácia Geral: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

# Gráfico 1: Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Previsto: Normal', 'Previsto: Hacker'], 
            yticklabels=['Real: Normal', 'Real: Hacker'])
plt.title(f'Matriz de Confusão - Anti-Cheat (Certeza >= {int(THRESHOLD_HACKER*100)}%)')
plt.ylabel('Realidade (Rótulo)')
plt.xlabel('Decisão do Modelo')
plt.tight_layout()
plt.show()

# Gráfico 2: Importância de cada Feature (Corrigido aviso do Seaborn)
feature_importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(
    x=feature_importances.values, 
    y=feature_importances.index, 
    hue=feature_importances.index, 
    palette='magma', 
    legend=False
)
plt.title('Importância das Features no Modelo Anti-Cheat')
plt.xlabel('Peso / Relevância da Variável')
plt.tight_layout()
plt.show()

# ==========================================
# 5. SALVAMENTO DO MODELO PARA PRODUÇÃO
# ==========================================
os.makedirs('models', exist_ok=True)
MODEL_PATH = 'models/anticheat_random_forest.pkl'
joblib.dump(model, MODEL_PATH)

print(f"Modelo compilado e salvo com sucesso em: {MODEL_PATH}")