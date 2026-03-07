import json

path = "/home/leonardomaximinobernardo/Downloads/quantum_camp_tech/VQE_vs_VQC_Variational_Classifier_Leonardo_Maximino_Bernardo_EuroSat.ipynb"

with open(path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

new_markdown_cell = """## 2. Carregamento Direto e Pré-processamento de Dados (EuroSat)
Nesta versão independente, os dados em formato nativo (`.jpg`) repousam no diretório do seu projeto principal da mesma forma que vieram no arquivo original baixado. Iremos ler as imagens cruas, redimensionar para (64x64x3), mapear as classes para binário [0, 1] e planificá-las no formato vetorial demandado pelas pipelines clássicas e quânticas da Qiskit.

**Instrução de Pathing**: O carregador utiliza o caminho absoluto contendo as sub-pastas do dataset (`AnnualCrop`, `SeaLake`, etc)."""

new_code_cell = """import os
import sys
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

# === FONTES DO DATASET (Estrutura Física QAC Oficial) ===
BASE_DIR = "/home/leonardomaximinobernardo/My_projects/Quantum_AgriClassifier_QAC/datasets/eurosat"

if not os.path.exists(BASE_DIR):
    sys.exit(f"❌ ERRO CRÍTICO: Raiz do Dataset ausente: {BASE_DIR}\\nPor favor reajuste para o diretório local as imagems brutas foram extraídas.")

images, labels = [], []
class_dirs = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

if not class_dirs:
    sys.exit("❌ ERRO CRÍTICO: Nenhuma classe detectada. Verifique se as distribuições .jpg estão separadas por pastas.")

print(f"Lendo e achatar imagens brutas (64x64) das classes: {class_dirs}...")

# Lendo com limite para acelerar a inicialização baseada no EuroSat NISQ
for class_name in class_dirs:
    class_path = os.path.join(BASE_DIR, class_name)
    files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in files[:250]: # Limite amostral de segurança / rapidez
        try:
            img = Image.open(os.path.join(class_path, img_file)).convert("RGB").resize((64, 64))
            img_array = np.array(img, dtype=np.float32).flatten() / 255.0
            images.append(img_array)
            labels.append(class_name)
        except Exception as e:
            continue

X_bin = np.array(images)
y_str = np.array(labels)

# Transforma as strings das pastas em 0 ou 1
le = LabelEncoder()
y_bin = le.fit_transform(y_str)

print(f"✅ Dados Extraídos em Memória: Mapeamento {dict(zip(le.classes_, range(len(le.classes_))))}")
print(f"Formato da Matriz X Completa (Imagens planificadas): {X_bin.shape}")

SEED = 42
N_QUBITS = 4

# Divisão Treino/Teste
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_bin, y_bin, test_size=0.3, random_state=SEED, stratify=y_bin
)

print(f"Amostras (Treino): {len(X_train_raw)} | (Teste): {len(X_test_raw)}")

# Aplicação do PCA para reduzir toda a grade 64x64x3 para um vetor de tamanho = N_QUBITS
pca = PCA(n_components=N_QUBITS, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_raw)
X_test_pca = pca.transform(X_test_raw)

# Escalonamento Rigoroso pro Algoritmo Quântico [-1, 1] (Evitar rotações infinitas na esfera de bloch)
X_train_scaled = minmax_scale(X_train_pca, feature_range=(-1, 1))
X_test_scaled = minmax_scale(X_test_pca, feature_range=(-1, 1))

print(f"Formato final dos dados quânticos [Inserção Ansatz] (Train): {X_train_scaled.shape}")
print(f"Variância Global explicada pelo PCA com {N_QUBITS} dimensões: {sum(pca.explained_variance_ratio_):.4f}")
"""

# Identificar as células corretas e substituí-las
notebook["cells"][2]["source"] = [l + "\\n" for l in new_markdown_cell.split('\\n')]
notebook["cells"][3]["source"] = [l + "\\n" for l in new_code_cell.split('\\n')]

# Validar se PIL precisa ser importado na primeira celula de dependências
dep_code = "".join(notebook["cells"][1]["source"])
if "PIL" not in dep_code:
    notebook["cells"][1]["source"].append("from PIL import Image\\n")

with open(path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Patch aplicado com sucesso! Carregamento nativo via PIL/os injetado.")
