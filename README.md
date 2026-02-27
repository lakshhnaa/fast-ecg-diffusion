# 🫀 Fast ECG Denoising using Deep Learning

A PyTorch-based deep learning system for removing noise from ECG (Electrocardiogram) signals while preserving important cardiac features such as QRS complexes.

---

## 📌 Problem Overview

ECG signals are frequently corrupted by:

- Gaussian noise  
- Motion artifacts  
- Baseline drift  

Noise can distort important heartbeat patterns and reduce diagnostic reliability.

This project builds a neural network that learns to reconstruct clean ECG signals from noisy inputs.

---

## 🧠 Approach

- 1D Convolutional Neural Network  
- Supervised learning (Noisy → Clean mapping)  
- Mean Squared Error (MSE) loss  
- Trained on PhysioNet MIT-BIH ECG dataset  

---

# 📊 Results

## 1️⃣ ECG Signal Comparison

![Denoising Result](denoising_result.png)

**Observation:**  
The denoised ECG (green) closely follows the clean ECG waveform while significantly reducing noise amplitude.

---

## 2️⃣ Training Loss Curve

![Loss Curve](losscurve.png)

**Observation:**  
The training loss decreases steadily across epochs, indicating stable convergence and successful learning.

---

## 3️⃣ Quantitative Performance

| Metric | Value |
|--------|-------|
| MSE (Noisy vs Clean) | 0.0123 |
| MSE (Denoised vs Clean) | 0.0045 |

**Improvement:**  
The denoised signal achieves substantially lower reconstruction error compared to the noisy input.

---

# 🏗️ Project Structure

```
fast-ecg-diffusion/
│
├── train.py
├── inference.py
├── models/
├── data/
├── results/
│   ├── denoising_result.png
│   └── losscurve.png
├── requirements.txt
└── README.md
```

---

# How to Run

### Install Dependencies

```
pip install torch wfdb matplotlib numpy
```

### Train the Model

```
python train.py
```

### Run Inference

```
python inference.py
```

---

# Future Improvements

- Implement diffusion-based training  
- Add spectral (frequency-domain) loss  
- Evaluate Signal-to-Noise Ratio (SNR) improvement  
- Optimize model for real-time deployment  

---
