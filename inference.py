import torch
import matplotlib.pyplot as plt
from models.unet1d import UNet1D
from models.diffusion import Diffusion1D


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet1D().to(device)
    model.load_state_dict(torch.load("ecg_diffusion_model.pth", map_location=device))
    model.eval()

    diffusion = Diffusion1D(model, timesteps=50, device=device)

    # Generate new ECG sample
    sample = diffusion.sample((1, 1, 512))

    sample = sample.detach().cpu().numpy()[0][0]

    plt.plot(sample)
    plt.title("Generated ECG Sample")
    plt.show()


if __name__ == "__main__":
    main()
