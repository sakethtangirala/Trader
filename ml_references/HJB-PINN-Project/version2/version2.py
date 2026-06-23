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
    "x_min": 0.1,
    "x_max": 5.0,
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
        t_norm = 2.0*(t - self.t_min)/(self.t_max - self.t_min) - 1.0
        x_norm = 2.0*(x - self.x_min)/(self.x_max - self.x_min) - 1.0
        return t_norm, x_norm

    def forward(self, t, x):

        gamma = params["gamma"]
        T     = params["T"]

        t_norm, x_norm = self._normalize_inputs(t, x)
        inp = torch.cat([t_norm, x_norm], dim=1)

        psi = self.net(inp)
        psi = psi / (1.0 + self.beta_denom*(x_norm**2))

        base = (x ** (1.0 - gamma)) / (1.0 - gamma)  
        V = base * (1.0 + (T - t)*psi)

        return V


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

    V_xx_stable = V_xx + eps*torch.sign(V_xx + 1e-12)

    hjb = V_t + r*x*V_x - 0.5 * lambda_sq * (V_x**2)/V_xx_stable
    return hjb, V, V_t, V_x, V_xx


def terminal_psi(model, t, x, params):
    gamma = params["gamma"]
    V_T = model(t, x)
    base = (x ** (1.0 - gamma)) / (1.0 - gamma)

    psi_T = (V_T / base) - 1.0
    return psi_T


def sample_interior(N, params, device):
    T = params["T"]
    x_min = params["x_min"]
    x_max = params["x_max"]

    t = torch.rand(N, 1, device=device) * T
    x = x_min + (x_max - x_min)*torch.rand(N, 1, device=device)
    return t, x

def sample_terminal(N, params, device):
    T = params["T"]
    x_min = params["x_min"]
    x_max = params["x_max"]

    t = torch.full((N, 1), T, device=device)
    x = x_min + (x_max - x_min)*torch.rand(N, 1, device=device)
    return t, x


def train_pinn(
    params,
    n_epochs   = 5000,
    N_int      = 2048,
    N_term     = 512,
    lr         = 1e-3,
    lambda_bc  = 10.0,
    print_every= 200,
):

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

    for epoch in range(1, n_epochs+1):

        optimizer.zero_grad()

        t_int, x_int = sample_interior(N_int, params, device)
        hjb, V_int, V_t, V_x, V_xx = hjb_residual(model, t_int, x_int, params)
        loss_pde = torch.mean(hjb**2)

        t_T, x_T = sample_terminal(N_term, params, device)
        psi_T = terminal_psi(model, t_T, x_T, params)   

        loss = loss_pde 

        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.4e} | PDE: {loss_pde.item():.4e}")

    return model

if __name__ == "__main__":
    model = train_pinn(
        params,
        n_epochs   = 2000,
        N_int      = 2048,
        N_term     = 512,
        lr         = 1e-3,
        lambda_bc  = 10.0,
        print_every= 100,
    )

    model.eval()
    with torch.no_grad():
        t_test = torch.tensor([[0.5]], device=device)
        x_test = torch.tensor([[1.0]], device=device)
        V_est = model(t_test, x_test).item()
        print("\nFinal V(0.5,1.0) =", V_est)