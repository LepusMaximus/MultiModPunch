import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# VAE Definition
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2_mu = nn.Linear(512, latent_dim)  # Mittelwert
        self.fc2_logvar = nn.Linear(512, latent_dim)  # Log-Varianz

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Loss function (Reconstruction + KL Divergence)
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Training loop for VAE
def train_vae(model, train_data, epochs=10, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_data):
            data = Variable(data)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {train_loss / len(train_data.dataset)}')


# Beispielhafte Anwendung auf Bild- oder Zeitreihendaten (dimensionality reduction auf 20)
input_dim = 1310720  # Beispiel für Bilddaten 1280x1024, für Zeitreihen wäre es 5200
latent_dim = 20  # Ziel für die Dimension

# Erstelle VAE Modell
vae_model = VAE(input_dim=input_dim, latent_dim=latent_dim)

# Bereite Daten vor (als PyTorch Tensor, z.B. Zeitreihen oder Bilddaten)
train_data = torch.utils.data.DataLoader(torch.Tensor(image_data), batch_size=32, shuffle=True)

# Trainiere das Modell
train_vae(vae_model, train_data, epochs=10)

# Nach dem Training kannst du die Latent-Space-Features nutzen
encoded_data = vae_model.encode(torch.Tensor(image_data))[0].detach().numpy()
