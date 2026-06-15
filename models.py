from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 16, 3, 1, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16 * 80 * 80, 128)

    def forward(self, x):
        print('input:', x.shape) # NCHW
        x = self.relu(self.conv1(x))
        print('conv1:', x.shape)
        x = self.relu(self.conv2(x))
        print('conv2:', x.shape)
        x = self.relu(self.conv3(x))
        print('conv3:', x.shape)
        x = self.flatten(x)
        print('flatten:', x.shape)
        x = self.linear(x)
        print('linear:', x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 102400)
        self.conv1 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 64, 3, 1, 1)
        self.conv3 = nn.ConvTranspose2d(64, 1, 3, 1, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print('input:', x.shape) # latent space
        x = self.linear(x)
        print('linear:', x.shape)
        x = x.reshape(x.shape[0], 16, 80, 80)
        print('reshaped:', x.shape)
        x = self.relu(self.conv1(x))
        print('conv1:', x.shape)
        x = self.relu(self.conv2(x))
        print('conv2:', x.shape)
        x = self.relu(self.conv3(x))
        print('conv3:', x.shape)
        x = self.sigmoid(x)
        return x