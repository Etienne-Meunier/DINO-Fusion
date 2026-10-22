"""Look for periodicity / repeats in mps-generator randn: FFT autocorrelation of a flattened (16,37,208,64) draw, and
cross-correlation between consecutive draws at all lags."""
import torch, numpy as np
shape = (16, 37, 208, 64)
for dev in ['cpu', 'mps']:
    g = torch.Generator(dev).manual_seed(0)
    a = torch.randn(shape, generator=g, device=dev).cpu().double().flatten().numpy()
    b = torch.randn(shape, generator=g, device=dev).cpu().double().flatten().numpy()
    n = len(a); N = 1 << (2 * n - 1).bit_length()
    fa, fb = np.fft.rfft(a, N), np.fft.rfft(b, N)
    ac = np.fft.irfft(fa * np.conj(fa), N)[:n] / n
    cc = np.fft.irfft(fb * np.conj(fa), N); cc = np.concatenate([cc[-n + 1:], cc[:n]]) / n
    lags = np.argsort(-np.abs(ac[1:]))[:5] + 1
    print(dev, 'n', n, 'top autocorr lags', lags.tolist(), np.round(ac[lags], 4).tolist(), ' noise level ~', round(1 / np.sqrt(n), 4))
    j = np.argsort(-np.abs(cc))[:3]
    print('   top crosscorr draw0-draw1', (j - n + 1).tolist(), np.round(cc[j], 4).tolist())
    s = a.reshape(16, -1)
    print('   per-sample mean*sqrt(n) (should be ~N(0,1))', np.round(s.mean(1) * np.sqrt(s.shape[1]), 2).tolist())
    print('   kurtosis', round(float(((a - a.mean()) ** 4).mean() / a.var() ** 2), 4), ' frac |x|>4', float((np.abs(a) > 4).mean()), '(expected 6.3e-5)')
