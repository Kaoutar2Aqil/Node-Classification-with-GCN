import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)  # Première couche GCN
        self.conv2 = GCNConv(16, out_channels)  # Deuxième couche GCN

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # Applique la première convolution + ReLU
        x = self.conv2(x, edge_index)  # Applique la deuxième convolution
        return F.log_softmax(x, dim=1)  # Retourne les probabilités log pour la classification
model = GCN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
def train():
    model.train()  # Met le modèle en mode entraînement
    optimizer.zero_grad()  # Réinitialise les gradients
    out = model(data)  # Effectue le passage avant
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # Calcul de la perte
    loss.backward()  # Rétropropagation des gradients
    optimizer.step()  # Mise à jour des poids
    return loss.item()
def test():
    model.eval()  # Met le modèle en mode évaluation
    out = model(data)  # Effectue le passage avant
    pred = out.argmax(dim=1)  # Prédit les classes avec la probabilité maximale
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()  # Compte les prédictions correctes
    acc = correct / data.test_mask.sum()  # Calcule la précision
    return acc.item()
# Boucle d'entraînement
for epoch in range(200):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Tester la précision
accuracy = test()
print(f'Test Accuracy: {accuracy:.4f}')
