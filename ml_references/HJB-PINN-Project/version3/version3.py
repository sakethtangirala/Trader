import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    "r": 0.02,
    "mu": 0.08,
    "sigma": 0.2,
    "gamma": 5.0,
    "T": 1.0,

    # wealth bounds
    "x_min": 0.1,
    "x_max": 5.0,

    "d": 3,     
    "z_min": -1.0,
    "z_max":  1.0,
}


class ValueNet(nn.Module):
    def __init__(self, t_min, t_max, x_min, x_max, z_min, z_max, d,
                 hidden_dim=64, n_hidden_layers=4, beta_denom=0.1):
        super().__init__()

        # Store bounds for normalization
        y_min = torch.log(torch.tensor(x_min))
        y_max = torch.log(torch.tensor(x_max))

        self.register_buffer("t_min", torch.tensor(t_min))
        self.register_buffer("t_max", torch.tensor(t_max))
        self.register_buffer("y_min", y_min)
        self.register_buffer("y_max", y_max)

        self.register_buffer("z_min", torch.tensor([z_min]*d))
        self.register_buffer("z_max", torch.tensor([z_max]*d))

        self.d = d
        self.beta_denom = beta_denom

        in_dim = 2 + d    
        layers = []
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def _normalize(self, t, y, z):
        t_norm = 2.0*(t - self.t_min)/(self.t_max - self.t_min) - 1.0
        y_norm = 2.0*(y - self.y_min)/(self.y_max - self.y_min) - 1.0
        z_norm = 2.0*(z - self.z_min)/(self.z_max - self.z_min) - 1.0
        return t_norm, y_norm, z_norm

    def forward(self, t, x, z):
        gamma = params["gamma"]
        T     = params["T"]

        y = torch.log(x)
        t_norm, y_norm, z_norm = self._normalize(t, y, z)

        inp = torch.cat([t_norm, y_norm, z_norm], dim=1)

        psi = self.net(inp)
        psi = psi / (1.0 + self.beta_denom*(y_norm**2))

        base = (x ** (1.0 - gamma)) / (1.0 - gamma)
        V = base * (1.0 + (T - t)*psi)
        return V


def value_and_derivatives(model, t, x, z):
    y = torch.log(x)

    t = t.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    V = model(t, torch.exp(y), z)

    grads = torch.autograd.grad(
        V, (t, y, z),
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True
    )
    V_t, V_y, V_z = grads

    (V_yy,) = torch.autograd.grad(
        V_y, y,
        grad_outputs=torch.ones_like(V_y),
        create_graph=True,
        retain_graph=True
    )

    x = torch.exp(y)
    V_x  = (1.0/x)*V_y
    V_xx = (1.0/x**2)*(V_yy - V_y)

    return V, V_t, V_x, V_xx, V_z


def hjb_residual(model, t, x, z, params, eps=1e-6):

    r     = params["r"]
    mu    = params["mu"]
    sigma = params["sigma"]

    V, V_t, V_x, V_xx, V_z = value_and_derivatives(model, t, x, z)

    lam_sq = (mu - r)**2 / sigma**2

    V_xx_stable = V_xx + eps*torch.sign(V_xx + 1e-12)

    hjb = V_t + r*x*V_x - 0.5*lam_sq*(V_x**2)/V_xx_stable

    return hjb

def sample_interior(N, params):
    T = params["T"]
    t = torch.rand(N, 1, device=device) * T

    x_min, x_max = params["x_min"], params["x_max"]
    x = x_min + (x_max - x_min)*torch.rand(N, 1, device=device)

    d = params["d"]
    z_min, z_max = params["z_min"], params["z_max"]
    z = z_min + (z_max - z_min)*torch.rand(N, d, device=device)

    return t, x, z

def train_pinn(params, n_epochs=2000, N_int=4096, lr=1e-3, print_every=100):

    model = ValueNet(
        t_min=0.0,
        t_max=params["T"],
        x_min=params["x_min"],
        x_max=params["x_max"],
        z_min=params["z_min"], 
        z_max=params["z_max"],
        d=params["d"],
        hidden_dim=128,
        n_hidden_layers=4,
        beta_denom=0.1,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, n_epochs+1):
        opt.zero_grad()

        t, x, z = sample_interior(N_int, params)
        hjb = hjb_residual(model, t, x, z, params)

        loss = torch.mean(hjb**2)
        loss.backward()
        opt.step()

        if ep % print_every == 0 or ep == 1:
            print(f"Epoch {ep:5d} | PDE Loss = {loss.item():.4e}")

    return model

if __name__ == "__main__":
    model = train_pinn(params)

    model.eval()
    with torch.no_grad():
        t = torch.tensor([[0.5]], device=device)
        x = torch.tensor([[1.0]], device=device)
        z = torch.zeros((1, params["d"]), device=device)
        V = model(t, x, z).item()
        print("\nV(0.5, 1.0, z=0) =", V)
