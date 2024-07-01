import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions=2):
        super().__init__()
        self.dim = dim
        self.num_dimensions = num_dimensions
        self.axial_attentions = nn.ModuleList([
            nn.MultiheadAttention(dim, 8, batch_first=True)
            for _ in range(num_dimensions)
        ])

    def forward(self, x):
        # x shape: (batch, height * width, channels)
        batch, seq_len, _ = x.shape
        height = width = int(math.sqrt(seq_len))
        
        for i in range(self.num_dimensions):
            if i == 0:
                # Attention along height
                x = x.view(batch, height, width, -1).permute(0, 2, 1, 3).reshape(batch * width, height, -1)
            else:
                # Attention along width
                x = x.view(batch, height, width, -1).permute(0, 1, 2, 3).reshape(batch * height, width, -1)
            
            x, _ = self.axial_attentions[i](x, x, x)
            
            if i == 0:
                x = x.view(batch, width, height, -1).permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
            else:
                x = x.view(batch, height, width, -1).permute(0, 1, 2, 3).reshape(batch, seq_len, -1)
        
        return x

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class EfficientAttentionModulatedFC(nn.Module):
    def __init__(self, in_features, out_features, style_dim, attention_type='axial'):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.modulation = nn.Linear(style_dim, in_features)
        
        if attention_type == 'axial':
            self.attention = AxialAttention(out_features)
        elif attention_type == 'linear':
            self.attention = LinearAttention(out_features)
        else:
            raise ValueError("Attention type must be 'axial' or 'linear'")
        
    def forward(self, x, style):
        style = self.modulation(style).unsqueeze(1)
        x = self.fc(x * style)
        x = self.attention(x)
        return x

class CIPSWithEfficientAttention(nn.Module):
    def __init__(self, style_dim=512, num_layers=14, hidden_dim=512, attention_type='axial'):
        super().__init__()
        self.style_dim = style_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.mapping_network = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        
        self.fourier_features = FourierFeatures(2, 256)
        self.coord_embeddings = nn.Parameter(torch.randn(256, 256, 512))
        
        self.net = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.net.append(EfficientAttentionModulatedFC(512 + 256, hidden_dim, style_dim, attention_type))
            else:
                self.net.append(EfficientAttentionModulatedFC(hidden_dim, hidden_dim, style_dim, attention_type))
            
            if i % 2 == 0 or i == num_layers - 1:
                self.to_rgb.append(ModulatedFC(hidden_dim, 3, style_dim))
        
    def forward(self, coords, z):
        batch_size = coords.shape[0]
        
        # Map z to w
        w = self.mapping_network(z)
        
        # Get Fourier features and coordinate embeddings
        fourier_features = self.fourier_features(coords)
        coord_embeddings = F.grid_sample(
            self.coord_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1),
            coords.unsqueeze(1),
            mode='bilinear',
            align_corners=True
        ).squeeze(2).permute(0, 2, 1)
        
        # Concatenate Fourier features and coordinate embeddings
        x = torch.cat([fourier_features, coord_embeddings], dim=-1)
        
        rgb = 0
        for i, (layer, to_rgb) in enumerate(zip(self.net, self.to_rgb)):
            x = layer(x, w)
            x = F.leaky_relu(x, 0.2)
            
            if i % 2 == 0 or i == self.num_layers - 1:
                rgb = rgb + to_rgb(x, w)
        
        return torch.sigmoid(rgb)

# Example usage
style_dim = 512
batch_size = 4
image_size = 256

z = torch.randn(batch_size, style_dim)
coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, image_size), torch.linspace(-1, 1, image_size)), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

model = CIPSWithEfficientAttention(attention_type='axial')
output = model(coords.reshape(batch_size, -1, 2), z)
output = output.reshape(batch_size, image_size, image_size, 3)

print(output.shape)  # Should be [4, 256, 256, 3]
