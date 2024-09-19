class iVAE(nn.Module):
    def __init__(self, input_dim, context_dim, latent_dim):
        super(iVAE, self).__init__()

        # Encoder: input + context
        self.fc1 = nn.Linear(input_dim + context_dim, 512)
        self.fc2_mu = nn.Linear(512, latent_dim)  # Mittelwert
        self.fc2_logvar = nn.Linear(512, latent_dim)  # Log-Varianz

        # Decoder: latent + context
        self.fc3 = nn.Linear(latent_dim + context_dim, 512)
        self.fc4 = nn.Linear(512, input_dim)

    def encode(self, x, c):
        h1 = torch.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h3 = torch.relu(self.fc3(torch.cat([z, c], dim=1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, input_dim), c.view(-1, context_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


# Loss function für iVAE
def ivae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Beispielhafte Anwendung auf Bild- oder Zeitreihendaten
input_dim = 1310720  # Für Bilddaten, für Zeitreihen wäre es 5200
context_dim = 5  # Beispielhafte Dimension der Kontextvariablen
latent_dim = 20  # Ziel für die Dimension

# Erstelle iVAE Modell
ivae_model = iVAE(input_dim=input_dim, context_dim=context_dim, latent_dim=latent_dim)

# Bereite Daten und Kontextvariablen vor (als PyTorch Tensor)
train_data = torch.utils.data.DataLoader(torch.Tensor(image_data), batch_size=32, shuffle=True)
context_data = torch.Tensor(np.random.rand(100, context_dim))  # Zufälliger Kontext

# Beispielhafte Trainierungs-Schleife (wie bei VAE)
for epoch in range(10):
    ivae_model.train()
    train_loss = 0
    for data in train_data:
        data = Variable(data)
        context = Variable(context_data)
        recon_batch, mu, logvar = ivae_model(data, context)
        loss = ivae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {train_loss / len(train_data.dataset)}')

# Extrahiere die Latent-Space-Features
encoded_data = ivae_model.encode(torch.Tensor(image_data), torch.Tensor(context_data))[0].detach().numpy()
