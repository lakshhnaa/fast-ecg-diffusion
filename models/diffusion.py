import torch

model = UNet1D()
x = torch.randn(4, 1, 512)
t = torch.randint(0, 1000, (4,))

out = model(x, t)

print("Output shape:", out.shape)
