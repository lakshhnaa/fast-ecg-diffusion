
import torch
import torch.nn as nn


class Diffusion1D:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.model = model
        self.timesteps = timesteps
        self.device = device

        # Beta schedule
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0, t):
        """
        Forward diffusion: add noise at timestep t
        """
        noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        noisy = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        return noisy, noise

    def sample(self, shape):
        """
        Reverse diffusion sampling (slow DDPM version)
        """
        x = torch.randn(shape).to(self.device)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)

            predicted_noise = self.model(x, t_tensor)

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                1 / torch.sqrt(alpha_t)
            ) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) \
                + torch.sqrt(beta_t) * noise

        return x
