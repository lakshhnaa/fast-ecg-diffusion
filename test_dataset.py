from data.dataset import ECGDataset
from torch.utils.data import DataLoader

dataset = ECGDataset(record_ids=['100'], window_size=512)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for noisy, clean in loader:
    print(noisy.shape, clean.shape)
    break
