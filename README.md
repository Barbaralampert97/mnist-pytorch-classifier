# MNIST PyTorch Classifier 🧠🔢

Este projeto implementa uma rede neural simples utilizando PyTorch para classificar dígitos manuscritos do famoso dataset MNIST. O projeto cobre desde o pré-processamento dos dados até o treinamento do modelo e visualização de resultados, sendo ideal para iniciantes em redes neurais e aprendizado profundo.

---

## 🧰 Tecnologias utilizadas

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- tqdm

---
## Estrutura do Projeto

```📁 mnist-pytorch-classifier/
├── 📄 rede_neural.py # Script principal com definição da rede e treinamento
├── 📄 modelo_mnist.pth # Arquivo salvo do modelo treinado
├── 📄 requirements.txt # Lista de dependências do projeto
└── 📄 README.md # Documentação do projeto
```




---

## 🚀 Como executar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/Barbaralampert97/mnist-pytorch-classifier.git
cd mnist-pytorch-classifier
```
### 2. Crie um ambiente virtual (opcional, mas recomendado)
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.\.venv\Scripts\activate         # Windows
```
### 3. Instale as dependências
```bash
pip install -r requirements.txt
```
### 4. Execute o código
```bash
python rede_neural.py
```

💡 O que o script faz?
Carrega o dataset MNIST (imagens de dígitos de 0 a 9).

Define uma arquitetura simples de rede neural com camadas lineares e função de ativação ReLU.

Treina a rede neural utilizando o otimizador Adam e perda CrossEntropyLoss.

Salva o modelo treinado no arquivo modelo_mnist.pth.

Exibe algumas imagens de teste com suas previsões.

### 🧪 Resultados
Durante o treinamento, são exibidos gráficos da função de perda para que você possa acompanhar a evolução do modelo.

Além disso, algumas imagens de validação são mostradas com a previsão da rede para análise qualitativa.

### 📦 Modelo Treinado
O modelo treinado é salvo no arquivo modelo_mnist.pth. Você pode reutilizá-lo futuramente para fazer inferência em novas imagens.

### 📚 Fontes de estudo
Documentação do PyTorch

Curso de Machine Learning com PyTorch - DIO

### 🧑‍💻 Autor
## Barbara Lampert 

Engenharia de Produção | Análise e Desenvolvimento de Sistemas | GitHub: @Barbaralampert97

### 📝 Licença
Este projeto está sob a licença MIT. Sinta-se à vontade para usar, modificar e compartilhar.
