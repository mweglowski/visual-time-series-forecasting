from torch import nn

class VisualAE(nn.Module):
    '''Reusable Visual Auto-Encoder class'''
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output

class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 7, 2, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 7, 2, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 10 * 10, latent_dim)

    def forward(self, x):
        # Apply: Conv -> BatchNorm -> Activation
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 128 * 10 * 10)
        self.conv1 = nn.ConvTranspose2d(128, 64, 7, 2, 3, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 32, 7, 2, 3, 1)
        self.bn2 = nn.BatchNorm2d(32)
        # Final layer omits BatchNorm to preserve raw sigmoid mapping scales
        self.conv3 = nn.ConvTranspose2d(32, 1, 7, 2, 3, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], 128, 10, 10)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # If you forcefully scale and standardize those final raw outputs to a mean of 0 and a variance of 1 right before passing them to a Sigmoid, you destroy the absolute magnitudes needed to draw clear target pixel lines.
        x = self.sigmoid(self.conv3(x)) # without bn
        return x
    


class EncoderPaper(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 5, 2, 2)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256 * 10 * 10, latent_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
class DecoderPaper(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 256 * 10 * 10)
        self.conv1 = nn.ConvTranspose2d(256, 128, 5, 2, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, 2, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        # Final layer omits BatchNorm to preserve raw sigmoid mapping scales
        self.conv3 = nn.ConvTranspose2d(64, 1, 5, 2, 2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], 256, 10, 10)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.sigmoid(self.conv3(x))
        return x