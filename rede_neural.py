import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm  # Adicionado para barra de progresso

# 1. Transformação das imagens em tensores
transform = transforms.ToTensor()

# 2. Carregamento dos dados de treino e validação
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# 3. Visualização de uma imagem do conjunto de treino
dataiter = iter(trainloader)
imagens, etiquetas = next(dataiter)

print("Treinamento - Primeira imagem:")
print(imagens[0].shape)  # [1, 28, 28]
print(etiquetas[0])      # Ex: 5
print(type(etiquetas[0]))  # <class 'torch.Tensor'>

plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')
plt.title(f"Etiqueta: {etiquetas[0].item()}")
plt.show()

# 4. Definição da arquitetura do modelo
class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, X):
        x = F.relu(self.linear1(X))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return F.log_softmax(x, dim=1)

# 5. Verificação de GPU ou CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nTreinamento será feito em: {device}")

# 6. Instancia modelo, função de perda e otimizador
modelo = Modelo().to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(modelo.parameters(), lr=0.01)

# 7. Função de treino com barra de progresso e gráfico de perda
def treinar_modelo(modelo, trainloader, criterion, optimizer, epochs=10):
    modelo.train()
    historico_perda = []

    for e in range(epochs):
        perda_total = 0
        loop = tqdm(trainloader, desc=f"Época {e+1}/{epochs}")

        for imagens, etiquetas in loop:
            imagens = imagens.view(imagens.shape[0], -1).to(device)
            etiquetas = etiquetas.to(device)
            optimizer.zero_grad()
            saida = modelo(imagens)
            perda = criterion(saida, etiquetas)
            perda.backward()
            optimizer.step()
            perda_total += perda.item()

        perda_media = perda_total / len(trainloader)
        historico_perda.append(perda_media)
        print(f"Perda média na época {e+1}: {perda_media:.4f}")

    # Gráfico de perda
    plt.plot(range(1, epochs + 1), historico_perda, marker='o')
    plt.xlabel("Época")
    plt.ylabel("Perda média")
    plt.title("Evolução da perda durante o treinamento")
    plt.grid(True)
    plt.show()

    # Salvando o modelo treinado
    torch.save(modelo.state_dict(), "modelo_mnist.pth")
    print("Modelo salvo como 'modelo_mnist.pth'.")

# 8. Treinamento
treinar_modelo(modelo, trainloader, criterion, optimizer, epochs=10)

# 9. Validação
correct = 0
total = 0
mostrou_val = False

modelo.eval()
with torch.no_grad():
    for imagens, etiquetas in valloader:
        imagens = imagens.view(imagens.shape[0], -1).to(device)
        etiquetas = etiquetas.to(device)
        saida = modelo(imagens)
        _, predicted = torch.max(saida.data, 1)
        total += etiquetas.size(0)
        correct += (predicted == etiquetas).sum().item()

        if not mostrou_val:
            print("\nValidação - Primeira imagem:")
            print(imagens[0].shape)
            print(etiquetas[0])
            print(type(etiquetas[0]))

            plt.imshow(imagens[0].cpu().view(28, 28).numpy(), cmap='gray_r')
            plt.title(f"Etiqueta: {etiquetas[0].item()} - Predito: {predicted[0].item()}")
            plt.show()
            mostrou_val = True

# 10. Resultados finais
print(f"\nTotal de imagens testadas: {total}")
print(f"Número de acertos: {correct}")
print(f"Precisão do modelo: {100 * correct / total:.2f}%")
