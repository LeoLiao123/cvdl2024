import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Hyperparameters
LATENT_DIM = 100
IMAGE_SIZE = 64
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.0002
BETA1 = 0.5

class MNISTCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # State size: (512, 4, 4)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # State size: (256, 8, 8)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # State size: (128, 16, 16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # State size: (64, 32, 32)
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: (1, 64, 64)
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input size: (1, 64, 64)
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (64, 32, 32)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (128, 16, 16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (256, 8, 8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (512, 4, 4)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_sample_images(netG, epoch, device, fixed_noise, sample_dir='samples'):
    """Save sample images from generator"""
    os.makedirs(sample_dir, exist_ok=True)
    
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        # Rescale images from [-1, 1] to [0, 1]
        fake = (fake + 1) / 2
        
        # Save the images
        image_path = os.path.join(sample_dir, f'fake_samples_epoch_{epoch}.png')
        vutils.save_image(fake, image_path, normalize=True)

def plot_loss_vs_iterations(G_losses, D_losses, save_path='loss_plot.png'):
    """Plot Discriminator and Generator losses vs. iterations"""
    iterations_G, values_G = zip(*G_losses)
    iterations_D, values_D = zip(*D_losses)
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(iterations_G, values_G, label="G", alpha=0.5)
    plt.plot(iterations_D, values_D, label="D", alpha=0.5)
    plt.xlabel("Iterations (K)")
    plt.ylabel("Loss")
    plt.legend()
    
    # Format x-axis to show iterations in thousands (K)
    ax = plt.gca()
    # Calculate appropriate tick positions
    max_iter = max(max(iterations_G), max(iterations_D))
    tick_spacing = max_iter // 10  # 10 ticks across the x-axis
    ticks = np.arange(0, max_iter + tick_spacing, tick_spacing)
    ax.set_xticks(ticks)
    ax.set_xticklabels(['{:.0f}K'.format(x/1000) for x in ticks])
    
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_gan(data_dir, num_epochs=NUM_EPOCHS):
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create the dataset and dataloader
    dataset = MNISTCustomDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Initialize the models
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Setup optimizers and criterion
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    # Create fixed noise for visualization
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    # Lists to store losses
    G_losses = []
    D_losses = []
    
    # Variables for tracking best model
    best_G_loss = float('inf')
    best_epoch = 0
    
    # Training loop
    total_iterations = len(dataloader) * num_epochs
    current_iteration = 0
    
    print("Starting Training Loop...")
    start_time = datetime.now()
    
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            current_iteration += 1
            
            ############################
            # Update D network
            ############################
            netD.zero_grad()
            real = data.to(device)
            batch_size = real.size(0)
            
            # Train with real
            label_real = torch.full((batch_size,), 1.0, device=device)
            output_real = netD(real).view(-1)
            errD_real = criterion(output_real, label_real)
            errD_real.backward()
            D_x = output_real.mean().item()

            # Train with fake
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            label_fake = torch.full((batch_size,), 0.0, device=device)
            output_fake = netD(fake.detach()).view(-1)
            errD_fake = criterion(output_fake, label_fake)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # Update G network
            ############################
            netG.zero_grad()
            output_fake = netD(fake).view(-1)
            errG = criterion(output_fake, label_real)  # Generate fake labels
            errG.backward()
            D_G_z2 = output_fake.mean().item()
            optimizerG.step()

            # Store losses and iteration numbers
            iteration = epoch * len(dataloader) + i
            G_losses.append((iteration, errG.item()))
            D_losses.append((iteration, errD.item()))

            # Print training stats
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

            # Save loss plot periodically
            if current_iteration % 500 == 0:
                plot_loss_vs_iterations(G_losses, D_losses)
                
        # Save sample images after each epoch
        save_sample_images(netG, epoch, device, fixed_noise)
            
        # Calculate average G loss for this epoch
        epoch_G_loss = sum(g_loss for _, g_loss in G_losses[-len(dataloader):]) / len(dataloader)
        
        # Save best model if current G loss is the lowest
        if epoch_G_loss < best_G_loss:
            best_G_loss = epoch_G_loss
            best_epoch = epoch
            print(f'\nNew best model found at epoch {epoch} with G_loss: {epoch_G_loss:.4f}')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
                'generator_losses': G_losses,
                'discriminator_losses': D_losses,
                'g_loss': epoch_G_loss
            }, 'models/best_model.pth')
        
        # Save regular checkpoint
        torch.save({
            'epoch': epoch,
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'generator_losses': G_losses,
            'discriminator_losses': D_losses,
        }, f'models/checkpoint_epoch_{epoch}.pth')

    training_time = datetime.now() - start_time
    print(f'\nTraining finished! Total time: {training_time}')
    print(f'Best model was found at epoch {best_epoch} with G_loss: {best_G_loss:.4f}')
    
    # Final loss plot
    plot_loss_vs_iterations(G_losses, D_losses)
    
    # Save final models
    torch.save({
        'generator_state_dict': netG.state_dict(),
        'discriminator_state_dict': netD.state_dict(),
        'generator_losses': G_losses,
        'discriminator_losses': D_losses,
    }, 'models/final_model.pth')

if __name__ == "__main__":
    data_dir = "Q2_image/mnist"  # Your data directory
    train_gan(data_dir, NUM_EPOCHS)