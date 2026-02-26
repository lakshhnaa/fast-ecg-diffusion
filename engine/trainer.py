import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, diffusion, dataloader, optimizer, device="cpu"):
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.mse = torch.nn.MSELoss()

    def train(self, epochs):
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0

            pbar = tqdm(self.dataloader)
            for noisy, clean in pbar:
                clean = clean.to(self.device)

                batch_size = clean.shape[0]
                t = torch.randint(
                    0, self.diffusion.timesteps,
                    (batch_size,),
                    device=self.device
                ).long()

                noisy_input, noise = self.diffusion.add_noise(clean, t)

                predicted_noise = self.model(noisy_input, t)

                loss = self.mse(predicted_noise, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

            print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(self.dataloader):.4f}")
