
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from data.dataset import ECGDataset
from models.unet1d import UNet1D
from models.diffusion import Diffusion1D
from engine.trainer import Trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ECGDataset(record_ids=['100'], window_size=512)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = UNet1D().to(device)
    diffusion = Diffusion1D(model, device=device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(model, diffusion, dataloader, optimizer, device)

    trainer.train(epochs=3)


if __name__ == "__main__":
    main()
