import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    "r": 0.02,
    "kappa": 1.5,
    "theta": 0.04,
    "sigma_v": 0.3,
    "rho": -0.7,
    "T": 1.0,
    "K": 1.0,
    "S_min": 0.01,
    "S_max": 5.0,
    "v_min": 1e-4,
    "v_max": 0.5,
}

class ValueNet(nn.Module):
    def __init__(
        self,
        t_min: float,
        t_max: float,
        S_min: float,
        S_max: float,
        v_min: float,
        v_max: float,
        hidden_dim: int = 64,
        n_hidden_layers: int = 4,
        beta_denom: float = 0.1,
    ):
        super().__init__()

        self.register_buffer("t_min", torch.tensor(t_min))
        self.register_buffer("t_max", torch.tensor(t_max))
        self.register_buffer("S_min", torch.tensor(S_min))
        self.register_buffer("S_max", torch.tensor(S_max))
        self.register_buffer("v_min", torch.tensor(v_min))
        self.register_buffer("v_max", torch.tensor(v_max))

        self.beta_denom = beta_denom

        layers = []
        in_dim = 3

        for i in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def _normalize_inputs(self, t, S, v):
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        S_norm = 2.0 * (S - self.S_min) / (self.S_max - self.S_min) - 1.0
        v_norm = 2.0 * (v - self.v_min) / (self.v_max - self.v_min) - 1.0
        return t_norm, S_norm, v_norm

    def forward(self, t, S, v):
        t_norm, S_norm, v_norm = self._normalize_inputs(t, S, v)
        inp = torch.cat([t_norm, S_norm, v_norm], dim=1)

        psi = self.net(inp)
        denom = 1.0 + self.beta_denom * (S_norm**2 + v_norm**2)
        psi = psi / denom

        return psi  

def value_and_derivatives(model, t, S, v):
    t = t.clone().detach().requires_grad_(True)
    S = S.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)

    u = model(t, S, v)

    u_t, u_S, u_v = torch.autograd.grad(
        u,
        (t, S, v),
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )

    (u_SS,) = torch.autograd.grad(
        u_S,
        S,
        grad_outputs=torch.ones_like(u_S),
        create_graph=True,
        retain_graph=True,
    )

    (u_vv,) = torch.autograd.grad(
        u_v,
        v,
        grad_outputs=torch.ones_like(u_v),
        create_graph=True,
        retain_graph=True,
    )

    (u_Sv,) = torch.autograd.grad(
        u_S,
        v,
        grad_outputs=torch.ones_like(u_S),
        create_graph=True,
        retain_graph=True,
    )

    return u, u_t, u_S, u_v, u_SS, u_vv, u_Sv

def hjb_residual(model, t, S, v, params, eps=1e-10):
    r = params["r"]
    kappa = params["kappa"]
    theta = params["theta"]
    sigma_v = params["sigma_v"]
    rho = params["rho"]

    u, u_t, u_S, u_v, u_SS, u_vv, u_Sv = value_and_derivatives(model, t, S, v)

    v_pos = torch.clamp(v, min=eps)

    pde = (
        u_t
        + 0.5 * v_pos * (S**2) * u_SS
        + rho * sigma_v * v_pos * S * u_Sv
        + 0.5 * (sigma_v**2) * v_pos * u_vv
        + r * S * u_S
        + kappa * (theta - v_pos) * u_v
        - r * u
    )

    return pde, u, u_t, u_S, u_v, u_SS, u_vv, u_Sv

def terminal_psi(model, t, S, v, params):
    u_T = model(t, S, v)
    payoff = torch.clamp(S - params["K"], min=0.0)
    psi_T = u_T - payoff
    return psi_T

def sample_interior(N, params, device):
    T = params["T"]
    S_min = params["S_min"]
    S_max = params["S_max"]
    v_min = params["v_min"]
    v_max = params["v_max"]

    t = torch.rand(N, 1, device=device) * T
    S = S_min + (S_max - S_min) * torch.rand(N, 1, device=device)
    v = v_min + (v_max - v_min) * torch.rand(N, 1, device=device)
    return t, S, v

def sample_terminal(N, params, device):
    T = params["T"]
    S_min = params["S_min"]
    S_max = params["S_max"]
    v_min = params["v_min"]
    v_max = params["v_max"]

    t = torch.full((N, 1), T, device=device)
    S = S_min + (S_max - S_min) * torch.rand(N, 1, device=device)
    v = v_min + (v_max - v_min) * torch.rand(N, 1, device=device)
    return t, S, v

def train_pinn(
    params,
    n_epochs=5000,
    N_int=2048,
    N_term=512,
    lr=1e-3,
    lambda_bc=10.0,
    print_every=200,
):
    model = ValueNet(
        t_min=0.0,
        t_max=params["T"],
        S_min=params["S_min"],
        S_max=params["S_max"],
        v_min=params["v_min"],
        v_max=params["v_max"],
        hidden_dim=64,
        n_hidden_layers=4,
        beta_denom=0.1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        t_int, S_int, v_int = sample_interior(N_int, params, device)
        pde, u_int, u_t, u_S, u_v, u_SS, u_vv, u_Sv = hjb_residual(model, t_int, S_int, v_int, params)
        loss_pde = torch.mean(pde**2)

        t_T, S_T, v_T = sample_terminal(N_term, params, device)
        psi_T = terminal_psi(model, t_T, S_T, v_T, params)
        loss_bc = torch.mean(psi_T**2)

        loss = loss_pde + lambda_bc * loss_bc

        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:5d} | Loss: {loss.item():.4e} | PDE: {loss_pde.item():.4e} | BC: {loss_bc.item():.4e}"
            )

    return model

if __name__ == "__main__":
    model = train_pinn(
        params,
        n_epochs=2000,
        N_int=2048,
        N_term=512,
        lr=1e-3,
        lambda_bc=10.0,
        print_every=100,
    )

    model.eval()
    with torch.no_grad():
        t_test = torch.tensor([[0.5]], device=device)
        S_test = torch.tensor([[1.0]], device=device)
        v_test = torch.tensor([[0.04]], device=device)
        u_est = model(t_test, S_test, v_test).item()
        print("\nFinal u(0.5, S=1.0, v=0.04) =", u_est)
