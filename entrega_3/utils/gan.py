import torch
import torch.nn as nn
import torch.nn.functional as F


class CondGenerator(nn.Module):
    def __init__(self, z_dim=100, num_classes=4, embed_dim=50):
        super().__init__()
        # Incrustamos la etiqueta numérica en un vector
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Proyectamos (Ruido + Etiqueta) a un mapa espacial de 4x4
        self.init_size = 4
        self.l1 = nn.Sequential(nn.Linear(z_dim + embed_dim, 512 * self.init_size ** 2))

        # Upsampling: 4x4 -> 8 -> 16 -> 32 -> 64 -> 128x128
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Unimos el ruido aleatorio con la etiqueta que queremos generar
        label_embed = self.label_embedding(labels)
        gen_input = torch.cat((label_embed, noise), dim=1)

        out = self.l1(gen_input)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class ProjectionCritic(nn.Module):
    def __init__(self, num_classes=4, feature_dim=512):
        super().__init__()
        # WGAN-GP prohíbe BatchNorm en el discriminador, usamos LeakyReLU puro
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc_realism = nn.Linear(512, 1)
        self.label_embedding = nn.Embedding(num_classes, feature_dim)

    def forward(self, img, labels):
        h = self.features(img)
        out = self.fc_realism(h)
        emb = self.label_embedding(labels)
        projection = torch.sum(h * emb, dim=1, keepdim=True)
        return out + projection


def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """Calcula la penalización de gradiente para WGAN-GP"""
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = critic(interpolates, labels)
    fake = torch.ones((real_samples.shape[0], 1), device=device, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def train_wgan_gp(generator, critic, dataloader, device, epochs=100, z_dim=100, classes=None):
    if classes is None:
        classes = ['Abstract', 'Classic', 'Fluid', 'Graphic']
    lr = 0.0001
    b1 = 0.0
    b2 = 0.9
    lambda_gp = 10
    n_critic = 3

    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    opt_C = torch.optim.Adam(critic.parameters(), lr=lr, betas=(b1, b2))

    generator.to(device)
    critic.to(device)

    history = {'G_loss': [], 'C_loss': []}

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, (imgs, labels) in enumerate(progress_bar):
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            batch_size = real_imgs.size(0)


            opt_C.zero_grad()

            z = torch.randn(batch_size, z_dim, device=device)
            fake_imgs = generator(z, labels).detach()

            real_validity = critic(real_imgs, labels)
            fake_validity = critic(fake_imgs, labels)

            gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs, labels, device)

            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            c_loss.backward()
            opt_C.step()


            if i % n_critic == 0:
                opt_G.zero_grad()

                gen_z = torch.randn(batch_size, z_dim, device=device)
                gen_imgs = generator(gen_z, labels)

                fake_validity = critic(gen_imgs, labels)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                opt_G.step()

            progress_bar.set_postfix(C_loss=c_loss.item(), G_loss=g_loss.item())

        history['C_loss'].append(c_loss.item())
        history['G_loss'].append(g_loss.item())

        if (epoch + 1) % 5 == 0:
            show_gan_samples(generator, device, z_dim, classes)

    return history


def show_gan_samples(generator, device, z_dim, classes=None):
    if classes is None:
        classes = ['Abstract', 'Classic', 'Fluid', 'Graphic']

    generator.eval()

    with torch.no_grad():
        num_classes = len(classes)

        z = torch.randn(num_classes, z_dim, device=device)
        labels = torch.arange(num_classes, dtype=torch.long, device=device)

        gen_imgs = generator(z, labels)

        gen_imgs = (gen_imgs + 1) / 2.0
        gen_imgs = gen_imgs.clamp(0, 1)
        gen_imgs = gen_imgs.cpu().permute(0, 2, 3, 1).numpy()

        fig, axes = plt.subplots(1, num_classes, figsize=(3 * num_classes, 3))

        if num_classes == 1:
            axes = [axes]

        for i in range(num_classes):
            axes[i].imshow(gen_imgs[i])
            axes[i].set_title(classes[i])
            axes[i].axis('off')

        plt.show()

    generator.train()