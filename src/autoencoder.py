import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.LeakyReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = act(inplace=True)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, input_channels=1, base_filters=16):
        r"""
        Encoder module that maps input spectrograms to a latent representation.
        input_channels: Number of input channels (e.g., 1 for grayscale spectrograms)
        base_filters: Number of filters in the first convolutional layer
        """
        super().__init__()
        f = base_filters

        self.conv1 = ConvBlock(input_channels, f)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv5 = ConvBlock(f * 8, f * 16)
        # self.dropout5 = nn.Dropout(0.5)
    
    def forward(self, x):
        c1 = self.conv1(x)       # size: 256
        p1 = self.pool1(c1)      # 128

        c2 = self.conv2(p1)      # 128
        p2 = self.pool2(c2)      # 64

        c3 = self.conv3(p2)      # 64
        p3 = self.pool3(c3)      # 32

        c4 = self.conv4(p3)      # 32
        p4 = self.pool4(c4)      # 16

        # Bottleneck
        c5 = self.conv5(p4)      # 16
        # d5 = self.dropout5(c5)
        d5 = c5

        return d5, (c1, c2, c3, c4)
    
class Decoder(nn.Module):
    def __init__(self, base_filters=16):
        r"""
        Decoder module that reconstructs spectrograms from latent representations.
        base_filters: Number of filters in the first convolutional layer
        """
        super().__init__()
        f = base_filters

        # Decoder (use ConvTranspose2d for upsampling)
        self.up6 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(f * 16, f * 8)

        self.up7 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(f * 8, f * 4)

        self.up8 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv8 = ConvBlock(f * 4, f * 2)

        self.up9 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv9 = ConvBlock(f * 2, f)

        # Final output
        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x, enc_features):
        c1, c2, c3, c4 = enc_features

        u6 = self.up6(x)                # 32
        m6 = torch.cat([c4, u6], dim=1)  # concat channels
        c6 = self.conv6(m6)

        u7 = self.up7(c6)                # 64
        m7 = torch.cat([c3, u7], dim=1)
        c7 = self.conv7(m7)

        u8 = self.up8(c7)                # 128
        m8 = torch.cat([c2, u8], dim=1)
        c8 = self.conv8(m8)

        u9 = self.up9(c8)                # 256
        m9 = torch.cat([c1, u9], dim=1)
        c9 = self.conv9(m9)

        out = self.out_conv(c9)
        return out

class Decoder2(nn.Module):
    def __init__(self, base_filters=16):
        r"""
        Decoder module that reconstructs spectrograms from latent representations.
        base_filters: Number of filters in the first convolutional layer
        """
        super().__init__()
        f = base_filters

        # Decoder (use ConvTranspose2d for upsampling)
        self.up6 = nn.ConvTranspose2d(f * 32, f * 16, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(f * 24, f * 12)

        self.up7 = nn.ConvTranspose2d(f * 12, f * 6, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(f * 10, f * 5)

        self.up8 = nn.ConvTranspose2d(f * 5, int(f * 2.5), kernel_size=2, stride=2)
        self.conv8 = ConvBlock(int(f * 4.5), int(f * 2.25))

        self.up9 = nn.ConvTranspose2d(int(f * 2.25), int(f * 1.125), kernel_size=2, stride=2)
        self.conv9 = ConvBlock(int(f * 2.125), f)

        # Final output
        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x, enc_features):
        c1, c2, c3, c4 = enc_features

        u6 = self.up6(x)                # 32
        m6 = torch.cat([c4, u6], dim=1)  # concat channels
        c6 = self.conv6(m6)

        u7 = self.up7(c6)                # 64
        m7 = torch.cat([c3, u7], dim=1)
        c7 = self.conv7(m7)

        u8 = self.up8(c7)                # 128
        m8 = torch.cat([c2, u8], dim=1)
        c8 = self.conv8(m8)

        u9 = self.up9(c8)                # 256
        m9 = torch.cat([c1, u9], dim=1)
        c9 = self.conv9(m9)

        out = self.out_conv(c9)
        return out

class UNet(nn.Module):
    def __init__(self, input_channels=1, base_filters=16, final_activation='tanh'):
        """
        final_activation: 'tanh', 'sigmoid', or None
        """
        super().__init__()
        self.encoder = Encoder(input_channels, base_filters)
        self.decoder = Decoder(base_filters)

        if final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x):
        # Encoder
        bottleneck, enc_features = self.encoder(x)
        # Decoder
        out = self.decoder(bottleneck, enc_features)

        if self.final_activation is not None:
            out = self.final_activation(out)
        return out
    
class UNet_double(nn.Module):
    def __init__(self, input_channels=1, base_filters=16, final_activation='tanh'):
        """
        final_activation: 'tanh', 'sigmoid', or None
        """
        super().__init__()
        self.encoder_contrastive = Encoder(input_channels, base_filters)
        self.encoder_regular = Encoder(input_channels, base_filters)
        self.decoder = Decoder2(base_filters)

        if final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x, noise=False):
        # Encoder
        contrastive_latent, _ = self.encoder_contrastive(x)
        regular_latent, enc_features = self.encoder_regular(x)
        if noise:
            contrastive_latent = torch.randn_like(contrastive_latent).to(contrastive_latent.device)
        bottleneck = torch.cat([contrastive_latent, regular_latent], dim=1)

        # Decoder
        out = self.decoder(bottleneck, enc_features)

        if self.final_activation is not None:
            out = self.final_activation(out)
        return out
    
    def freeze_contrastive_encoder(self):
        for param in self.encoder_contrastive.parameters():
            param.requires_grad = False

def autoencoder_loss(pred, target):
    return nn.MSELoss()(pred, target)

if __name__ == "__main__":
    # quick smoke test
    model = UNet(input_channels=1, base_filters=16, final_activation='tanh')
    x = torch.randn(2, 1, 256, 256)   # batch=2
    y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)   # should be (2, 1, 256, 256)
