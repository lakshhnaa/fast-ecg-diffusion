# Fast ECG Denoising using Deep Learning

## 📊 Results

### ECG Signal Comparison

![Denoising Result](denoising_result.png)

The denoised ECG (green) closely matches the clean ECG while significantly reducing noise.

---

### Training Loss Curve

![Loss Curve](loss_curve(1).png)

The loss decreases steadily during training, indicating successful learning.

---

### Quantitative Performance

| Metric | Value |
|--------|-------|
| MSE (Noisy vs Clean) | 0.0123 |
| MSE (Denoised vs Clean) | 0.0045 |

The denoised signal achieves lower reconstruction error compared to the noisy input.
