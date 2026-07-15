import torch
import torch.nn as nn


class DiffusionScheduler:
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.長期_cumprod(self.alpha, dim=0) if hasattr(torch, '長期_cumprod') else torch.cumprod(
            self.alpha, dim=0)

    def add_noise(self, x_start, t, noise):
        """Ecuación directa para infundir ruido en el paso t"""

        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t])[:, None, None, None]

        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.GELU()
        )

    def forward(self, x): return self.net(x)


class MiniUNetCond(nn.Module):
    def __init__(self, num_classes=4, time_dim=256, class_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.class_emb = nn.Embedding(num_classes, class_dim)

        total_emb_dim = time_dim + class_dim

        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))  # 64x64
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))  # 32x32

        self.emb_proj = nn.Linear(total_emb_dim, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 64x64
        self.conv_up1 = DoubleConv(256, 128)  # 128 (skip) + 128

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 128x128
        self.conv_up2 = DoubleConv(128, 64)  # 64 (skip) + 64

        self.outc = nn.Conv2d(64, 3, 1)

    def forward(self, x, t, y):
        t = t.float().unsqueeze(-1)
        t_emb = self.time_mlp(t)
        y_emb = self.class_emb(y)
        emb = torch.cat([t_emb, y_emb], dim=1)
        emb_spatial = self.emb_proj(emb)[:, :, None, None]

        # 2. Encoder (Down)
        x1 = self.inc(x)  # 128x128, 64 canales
        x2 = self.down1(x1)  # 64x64, 128 canales
        x3 = self.down2(x2)  # 32x32, 256 canales

        x3 = x3 + emb_spatial

        # 3. Decoder (Up con Skip Connections)
        x_up = self.up1(x3)
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.conv_up1(x_up)

        x_up = self.up2(x_up)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.conv_up2(x_up)

        return self.outc(x_up)


import matplotlib.pyplot as plt


def train_diffusion(model, scheduler, dataloader, device, epochs=30, timesteps=300, classes=None):
    if classes is None:
        classes = ['Abstract', 'Classic', 'Fluid', 'Graphic']
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)  # Imágenes reales [-1, 1]
            labels = labels.to(device)  # Etiquetas de estilo (0, 1, 2, 3)
            batch_size = imgs.size(0)

            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            noise = torch.randn_like(imgs)

            x_noisy = scheduler.add_noise(imgs, t, noise)

            predicted_noise = model(x_noisy, t, labels)

            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} -> Loss de Difusión: {running_loss / len(dataloader):.5f}")

        if (epoch + 1) % 5 == 0:
            generate_diffusion_samples(model, scheduler, device, classes)


def generate_diffusion_samples(model, scheduler, device, num_samples=4, timesteps=300, classes=None):
    model.eval()
    with torch.no_grad():
        x = torch.randn((num_samples, 3, 128, 128), device=device)
        labels = torch.tensor([0, 1, 2, 3], device=device)

        for i in reversed(range(timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            predicted_noise = model(x, t, labels)

            alpha_t = scheduler.alpha[i]
            alpha_cumprod_t = scheduler.alpha_cumprod[i]
            beta_t = scheduler.beta[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            sigma_t = torch.sqrt(beta_t)

            x = (1 / torch.sqrt(alpha_t)) * (
                        x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + sigma_t * noise

        x = torch.clamp((x + 1) / 2.0, 0.0, 1.0)
        x = x.cpu().permute(0, 2, 3, 1).numpy()

        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for idx in range(4):
            axes[idx].imshow(x[idx])
            axes[idx].set_title(classes[idx])
            axes[idx].axis('off')
        plt.show()

    model.train()