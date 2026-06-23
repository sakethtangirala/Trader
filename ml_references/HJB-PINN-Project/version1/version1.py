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
    "rho": 0.03,
    "T": 1.0,
    "x_min": 0.1,
    "x_max": 5.0,
    "eps_bequest": 1e-3 
}

class ValueNet(nn.Module):
    def __init__(
        self,
        t_min: float,
        t_max: float,
        x_min: float,
        x_max: float,
        hidden_dim: int = 64,
        n_hidden_layers: int = 4,
        beta_denom: float = 0.1,
    ):
        super().__init__()

        self.register_buffer("t_min", torch.tensor(t_min))
        self.register_buffer("t_max", torch.tensor(t_max))
        self.register_buffer("x_min", torch.tensor(x_min))
        self.register_buffer("x_max", torch.tensor(x_max))

        self.beta_denom = beta_denom

        layers = []
        in_dim = 2  

        for i in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def _normalize_inputs(self, t, x):
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        return t_norm, x_norm

    def forward(self, t, x):
        t_norm, x_norm = self._normalize_inputs(t, x)
        inp = torch.cat([t_norm, x_norm], dim=1)
        raw = self.net(inp)
        denom = 1.0 + self.beta_denom * (x_norm ** 2)
        V_tilde = raw / denom
        return V_tilde

def value_and_derivatives(model, t, x):
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)

    V = model(t, x)  
    V_t, V_x = torch.autograd.grad(
        V,
        (t, x),
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )

    (V_xx,) = torch.autograd.grad(
        V_x,
        x,
        grad_outputs=torch.ones_like(V_x),
        create_graph=True,
        retain_graph=True,
    )

    return V, V_t, V_x, V_xx

def hjb_residual(model, t, x, params, eps=1e-6):
    r = params["r"]
    mu = params["mu"]
    sigma = params["sigma"]

    V, V_t, V_x, V_xx = value_and_derivatives(model, t, x)
    lambda_sq = (mu - r) ** 2 / (sigma ** 2)

    V_xx_stable = V_xx + eps * torch.sign(V_xx + 1e-12)

    hjb = V_t + r * x * V_x - 0.5 * lambda_sq * (V_x ** 2) / V_xx_stable
    return hjb, V, V_t, V_x, V_xx


def terminal_value_unscaled(x, params):
    gamma = params["gamma"]
    if abs(gamma - 1.0) < 1e-12:
        return torch.log(x)
    else:
        return x ** (1.0 - gamma) / (1.0 - gamma)

def terminal_value_scaled(x, params):
    V_raw = terminal_value_unscaled(x, params)
    V_scale = params["V_scale"]
    return V_raw / V_scale

def sample_interior(N, params, device):
    T = params["T"]
    x_min = params["x_min"]
    x_max = params["x_max"]

    t = torch.rand(N, 1, device=device) * T
    x = x_min + (x_max - x_min) * torch.rand(N, 1, device=device)
    return t, x


def sample_terminal(N, params, device):
    T = params["T"]
    x_min = params["x_min"]
    x_max = params["x_max"]

    t = torch.full((N, 1), T, device=device)
    x = x_min + (x_max - x_min) * torch.rand(N, 1, device=device)
    return t, x


def train_pinn(
    params,
    n_epochs: int = 10_000,
    N_int: int = 2048,
    N_term: int = 512,
    lr: float = 1e-3,
    lambda_bc: float = 5.0,
    print_every: int = 500,
):
    with torch.no_grad():
        x_min_tensor = torch.tensor([[params["x_min"]]], device=device)
        V_min = terminal_value_unscaled(x_min_tensor, params)
        V_scale = float(torch.abs(V_min).item())
    params["V_scale"] = V_scale
    print(f"Using V_scale = {V_scale:.4e}")

    model = ValueNet(
        t_min=0.0,
        t_max=params["T"],
        x_min=params["x_min"],
        x_max=params["x_max"],
        hidden_dim=64,
        n_hidden_layers=4,
        beta_denom=0.1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        t_int, x_int = sample_interior(N_int, params, device)
        hjb, V_int, V_t, V_x, V_xx = hjb_residual(model, t_int, x_int, params)
        loss_pde = torch.mean(hjb ** 2)

        t_T, x_T = sample_terminal(N_term, params, device)
        V_T_pred = model(t_T, x_T)                    
        V_T_true = terminal_value_scaled(x_T, params)  
        loss_bc_T = torch.mean((V_T_pred - V_T_true) ** 2)

        loss = loss_pde + lambda_bc * loss_bc_T

        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:5d} | Loss: {loss.item():.4e} | "
                f"PDE: {loss_pde.item():.4e} | BC_T: {loss_bc_T.item():.4e}"
            )

    return model

if __name__ == "__main__":
    trained_model = train_pinn(
        params,
        n_epochs=2000,
        N_int=2048,
        N_term=512,
        lr=1e-3,
        lambda_bc=5.0,
        print_every=100,
    )

    trained_model.eval()
    with torch.no_grad():
        t_test = torch.tensor([[0.5]], device=device)
        x_test = torch.tensor([[1.0]], device=device)
        V_tilde = trained_model(t_test, x_test)      
        V_hat = V_tilde * params["V_scale"]         
        print(f"V_tilde(t=0.5, x=1.0) = {V_tilde.item():.8e}")
        print(f"V_hat(t=0.5, x=1.0)   = {V_hat.item():.8e}")
