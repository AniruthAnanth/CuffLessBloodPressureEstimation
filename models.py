import torch
import torch.nn as nn
from torchdiffeq import odeint

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        # Removed CNN layers, keep only RNN and FC
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=128, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    def forward(self, x):
        # x: [batch, seq_len, input_dim] expected
        _, (hn, _) = self.rnn(x)
        hn = hn.squeeze(0)
        return self.fc(hn)

class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # network to predict patient-specific parameters: characteristic impedance (Rp), peripheral resistance (Rd), and compliance (C)
        self.param_net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
        # compensation network for additional latent dynamics
        self.comp_net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, t, z):
        # z: [batch, latent_dim]
        params = self.param_net(z)
        # ensure positive parameters via exponentiation
        Rp = torch.exp(params[:, 0]).unsqueeze(1)  # characteristic impedance
        Rd = torch.exp(params[:, 1]).unsqueeze(1)  # peripheral resistance
        C  = torch.exp(params[:, 2]).unsqueeze(1)  # arterial compliance

        # latent compensation term
        comp = self.comp_net(z)
        # 3-element Windkessel ODE: C * dz/dt = -z/Rd - Rp * z + comp
        dzdt = (- z / Rd - Rp * z + comp) / C
        return dzdt

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Decoder now takes latent state + 3 ODE parameters
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, z):
        # legacy forward (not used)
        return self.net(z)

class BPModel(nn.Module):
    def __init__(self, input_dim, latent_dim=128, use_ode=True):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.use_ode = use_ode
        self.odefunc = LatentODEFunc(latent_dim) if use_ode else None
        self.decoder = Decoder(latent_dim)
    def forward(self, x):
        z0 = self.encoder(x)
        if self.use_ode:
            ts = torch.linspace(0.0, 1.0, 10, device=x.device)
            zt = odeint(self.odefunc, z0, ts, method='rk4', rtol=1e-6, atol=1e-6)
            zT = zt[-1]
            params = torch.exp(self.odefunc.param_net(z0))  # [batch,3]
        else:
            zT = z0
            params = torch.zeros(zT.size(0), 3, device=x.device)
        # Ensure zT and params are tensors
        if isinstance(zT, tuple):
            zT = zT[0]
        if isinstance(params, tuple):
            params = params[0]
        return self.decoder(torch.cat([zT, params], dim=1))
