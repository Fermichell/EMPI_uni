import numpy as np, matplotlib.pyplot as plt, pywt
np.random.seed(7)
N=256
t=np.arange(N)
x=np.sin(2*np.pi*t/32)+0.5*np.sin(2*np.pi*t/12)+np.random.normal(0,0.2,N)
wavelet="gaus7"
scales=np.arange(1,64)
coeffs,freqs=pywt.cwt(x, scales, wavelet, sampling_period=1)
plt.figure(figsize=(10,6))
plt.subplot(2,1,1); plt.plot(t,x); plt.title("Часовий ряд")
plt.subplot(2,1,2); plt.pcolormesh(t, scales, coeffs, shading="auto"); plt.gca().invert_yaxis()
plt.ylabel("Масштаб"); plt.xlabel("Час"); plt.title(f"Скейлограма ({wavelet})")
plt.tight_layout(); plt.savefig("outputs/wavelet_scaleogram_7.png", dpi=150)
print("Saved wavelet figure.")