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
    
class EncoderPlainConv(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1) # 80x80 -> 40x40
        self.conv2 = nn.Conv2d(64, 32, 3, 2, 1) # 40x40 -> 20x20
        self.conv3 = nn.Conv2d(32, 16, 3, 2, 1) # 20x20 -> 10x10
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten() # 16x10x10 -> 1600
        self.linear = nn.Linear(1600, latent_dim)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = self.relu(self.conv3(x))
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)
        return x

class DecoderPlainConv(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1600)
        self.conv1 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 64, 3, 2, 1, 1)
        self.conv3 = nn.ConvTranspose2d(64, 1, 3, 2, 1, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], 16, 10, 10)
        # print(x.shape)
        x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = self.sigmoid(self.conv3(x))
        # print(x.shape)
        return x