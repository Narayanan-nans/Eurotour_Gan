import torch
import torch.nn as nn
import torch.optim as optim
from eurotourgan_gan import Generator, Discriminator

# Hyperparameters
latent_dim = 20
input_size = 50
epochs = 2000
batch_size = 32
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


generator = Generator(latent_dim, input_size).to(device)
discriminator = Discriminator(input_size).to(device)


criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

real_data = torch.randn(1000, input_size)  # 1000 samples
real_data = (real_data - real_data.min()) / (real_data.max() - real_data.min())  # Normalize to [0, 1]


for epoch in range(epochs):
    for i in range(0, real_data.size(0), batch_size):
        real_batch = real_data[i:i+batch_size].to(device)

        
        valid = torch.ones(real_batch.size(0), 1).to(device)
        fake = torch.zeros(real_batch.size(0), 1).to(device)

       
        optimizer_D.zero_grad()

        real_loss = criterion(discriminator(real_batch), valid)

        z = torch.randn(real_batch.size(0), latent_dim).to(device)
        fake_data = generator(z).detach()
        fake_loss = criterion(discriminator(fake_data), fake)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

     
        optimizer_G.zero_grad()

        z = torch.randn(real_batch.size(0), latent_dim).to(device)
        gen_data = generator(z)
        g_loss = criterion(discriminator(gen_data), valid)

        g_loss.backward()
        optimizer_G.step()

    if epoch % 100 == 0:
        print(f"[Epoch {epoch}/{epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")


torch.save(generator.state_dict(), "generator.pt")
torch.save(discriminator.state_dict(), "discriminator.pt")
print("✅ Training complete. Models saved.")
