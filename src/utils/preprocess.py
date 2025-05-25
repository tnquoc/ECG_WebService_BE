import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import maximum_filter1d


# preprocess ecg signal
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def smooth(x, window_len=11, window="hanning"):
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window_len % 2 == 0:
        window_len += 1
    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise (
            ValueError,
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'",
        )
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[int(window_len / 2) : -int(window_len / 2)]


def baseline_wander_remove(signal, fs=250, f1=0.2, f2=0.6):
    window1 = int(f1 * fs / 2) + 1 if int(f1 * fs / 2) % 2 == 0 else int(f1 * fs / 2)
    window2 = int(f2 * fs / 2) + 1 if int(f2 * fs / 2) % 2 == 0 else int(f2 * fs / 2)
    out1 = smooth(signal, window1)
    out2 = smooth(out1, window2)
    bwr_signal = signal - out2
    return bwr_signal


def normalize(raw, window_len, samp_from=-1, samp_to=-1):
    # The window size is the number of samples that corresponds to the time analogue of 2e = 0.5s
    if window_len % 2 == 0:
        window_len += 1
    abs_raw = abs(raw)
    # Remove outlier
    while True:
        g = maximum_filter1d(abs_raw, size=window_len)
        if np.max(abs_raw) < 5.0:
            break
        abs_raw[g > 5.0] = 0
    g_smooth = smooth(g, window_len, window="hamming")
    g_mean = max(np.mean(g_smooth) / 3.0, 0.1)
    g_smooth = np.clip(g_smooth, g_mean, None)
    # Avoid cases where the value is )
    g_smooth[g_smooth < 0.01] = 1
    nor_signal = np.divide(raw, g_smooth)
    return nor_signal
