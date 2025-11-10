import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = tuple(k // 2 for k in kernel_size)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        i, f, o, g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch, spatial_size, device=None):
        H, W = spatial_size
        h = torch.zeros(batch, self.hidden_dim, H, W, device=device)
        c = torch.zeros(batch, self.hidden_dim, H, W, device=device)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1, bias=True):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_ch = input_dim if l == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(in_ch, hidden_dim, kernel_size, bias=bias))

    def forward(self, x_seq):
        outputs = []
        h, c = None, None
        for t, x in enumerate(x_seq):
            if h is None:
                B, _, H, W = x.shape
                device = x.device
                h, c = [], []
                for cell in self.cells:
                    _h, _c = cell.init_hidden(B, (H, W), device=device)
                    h.append(_h); c.append(_c)
            inp = x
            for li, cell in enumerate(self.cells):
                _h, _c = cell(inp, h[li], c[li])
                h[li], c[li] = _h, _c
                inp = _h
            outputs.append(inp)
        return outputs
