# MNIST Handwritten Digit Classifier with PyTorch

Este projeto implementa uma rede neural simples para classificação de dígitos manuscritos utilizando o dataset MNIST. Foi desenvolvido com o framework PyTorch e tem como objetivo demonstrar, de forma didática, o funcionamento de uma rede neural artificial treinada do zero.

## 🔍 Sobre o Projeto

O MNIST é um dataset amplamente utilizado na área de aprendizado de máquina, composto por 70.000 imagens em escala de cinza de dígitos manuscritos (60.000 para treino e 10.000 para teste), com tamanho 28x28 pixels.

Neste projeto:
- Utilizamos o PyTorch para definir a arquitetura da rede, otimizar os pesos e realizar as predições.
- O modelo é treinado por padrão com 10 épocas, mas isso pode ser ajustado facilmente.
- Ao final do treinamento, o modelo é salvo no formato `.pth` para reutilização.

## 🧠 Estrutura da Rede Neural

A rede implementada possui a seguinte arquitetura:
- Entrada: 784 neurônios (28x28 pixels achatados)
- Camada oculta: 128 neurônios com ativação ReLU
- Saída: 10 neurônios (um para cada dígito de 0 a 9) com ativação log-softmax

## 📁 Estrutura do Projeto

rede-neural-do-zero/
│
├── rede_neural.py # Código principal do projeto
├── modelo_mnist.pth # Arquivo com os pesos salvos do modelo treinado
├── requirements.txt # Lista de dependências do projeto
└── README.md # Este arquivo

## 🚀 Como Executar o Projeto

### 1. Clone o repositório

git clone https://github.com/Barbaralampert97/mnist-pytorch-classifier.git
cd mnist-pytorch-classifier
2. Crie um ambiente virtual (opcional, mas recomendado)

python -m venv .venv
# Ativando no Windows
.venv\Scripts\activate
# Ativando no Linux/macOS
source .venv/bin/activate
3. Instale as dependências
bash
Copiar
Editar
pip install -r requirements.txt
4. Execute o script
bash
Copiar
Editar
python rede_neural.py
O script irá:

Baixar automaticamente o dataset MNIST

Treinar a rede neural

Exibir algumas imagens de teste com as predições

Salvar o modelo como modelo_mnist.pth

🔄 Ajustes Possíveis
Se quiser treinar por mais ou menos épocas, basta alterar o valor da variável epochs no arquivo rede_neural.py:


epochs = 10  # você pode mudar para qualquer número de épocas
💾 Como carregar o modelo salvo
Você pode carregar o modelo salvo com o seguinte código:

import torch

model = torch.load('modelo_mnist.pth')
model.eval()
🧪 Requisitos
Python 3.8+

PyTorch

Torchvision

Matplotlib

Numpy

📚 Fontes e Referências
MNIST Dataset

PyTorch Documentation

🧑‍💻 Autora
Feito com 💡 por @Barbaralampert97 como parte do aprendizado em inteligência artificial e redes neurais com PyTorch.

