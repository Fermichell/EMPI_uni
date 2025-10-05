import numpy as np, matplotlib.pyplot as plt
VARIANT=7
np.random.seed(7)
N=120
trend=np.linspace(20,40,N)
season=5*np.sin(np.linspace(0,8*np.pi,N))
noise=np.random.normal(0,2,N)
D=(trend+season+noise)

def moving_average(x,w):
    k=np.ones(w)/w
    return np.convolve(x,k,mode="same")

S_w = moving_average(D, VARIANT)
plt.figure(figsize=(10,5))
plt.plot(D,label="D")
plt.plot(S_w,label=f"SMA w={VARIANT}")
plt.title("Віконне згладжування")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/smoothing_window_w7.png", dpi=150)

def exp_smoothing(x, alpha):
    s=np.zeros_like(x); s[0]=x[0]
    for t in range(1,len(x)):
        s[t]=alpha*x[t]+(1-alpha)*s[t-1]
    return s

alpha=VARIANT/10
S_a=exp_smoothing(D, alpha)
plt.figure(figsize=(10,5))
plt.plot(D,label="D")
plt.plot(S_a,label=f"ES α={alpha:.1f}")
plt.title("Експоненційне згладжування")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/smoothing_exponential_a7.png", dpi=150)
print("Saved smoothing figures.")