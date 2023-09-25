import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal


def load_csv_to_dict(file_path):
    # Load the CSV file with numpy.genfromtxt
    data = np.genfromtxt(file_path, delimiter=',', names=True)

    # Create a dictionary with column headers as keys and columns as values
    dict_data = {}
    for col in data.dtype.names:
        dict_data[col] = data[col]

    return dict_data


def get_results(filepath,value, new_figure = 0, title = "" , labels = ["training","validation"], xxlim = (None,None), yylim = (None, None), also_val = 0):
  result = load_csv_to_dict(filepath)
  if new_figure:
    plt.figure()
  plt.title(title)
  plt.plot(result['epoch'],result[value], label = labels[0])
  if also_val:
    plt.plot(result['epoch'],result['val_' + value], label = labels[1])
  plt.xlim(xxlim)
  plt.ylim(yylim)
  plt.legend()


def synchronize_signals(signal1, signal2):

  cross_correlation = np.correlate((signal1 - np.mean(signal1)) / np.std(signal1), (signal2 - np.mean(signal2)) / np.std(signal2), mode='same')
  time_delay = np.argmax(cross_correlation) - len(signal1) // 2
  synchronized_signal2 = np.roll(signal2, time_delay)
  return synchronized_signal2, time_delay, cross_correlation


def synchronize_signals_peaks(s1,s2):
    calc_prom_1 = abs(np.median(s1) - np.min(s1))
    calc_prom_2 = abs(np.median(s2) - np.min(s2))
    peaks_1, _ = scipy.signal.find_peaks(s1, prominence=calc_prom_1, distance = 50)
    peaks_2, _ = scipy.signal.find_peaks(s2, prominence=calc_prom_2, distance = 50)
    m = np.min([len(peaks_1), len(peaks_2)])
    time_delay = int(np.median(peaks_1[:m] - peaks_2[:m]))
    synchronized_signal2 = np.roll(s2, time_delay)
    return synchronized_signal2, time_delay

def gaussian(x, a, b, c):
    return a * np.exp(-np.power(x - b, 2.) / (2 * np.power(c, 2.)))

def fourier_plot(data):

  tx  = np.fft.fft(data)
  tx[0] = 0

  return tx

def reverse_second_middle(arr):
    n = len(arr)
    middle = n // 2

    # Split the array into two parts
    first_half = arr[:middle]
    second_half = arr[middle:]

    # Reverse the first half
    first_half_reversed = first_half[::-1]

    # Replace the second half with the reversed first half
    arr[middle:] = first_half_reversed

    return arr


def reconstruct_signal(freq_spectrum):
  itx = np.fft.ifft(freq_spectrum)
  return itx

def z_renorm(x):
    return (x - np.mean(x))/np.std(x)

def zero_one_renorm_single(x):
  xx = np.copy(x)
  xx= (xx- np.min(xx))/(np.max(xx) - np.min(xx))
  return xx

def VPPG(ppg, n = 10, f = 15, fs = 125.0, padding = 50, delay=0):

    vppg = np.gradient(ppg)
    b,a = scipy.signal.butter(n, f, 'lp', fs=125.0)
    sos =scipy.signal.tf2sos(b,a)
    w, gd = scipy.signal.group_delay((b, a))
    gd_mean = int(np.round(np.mean(gd)))
    filtered = scipy.signal.sosfilt(sos, vppg)
    delay = gd_mean + 1
    return filtered[padding + delay : -padding + delay], vppg[padding:-padding]


def APPG(vppg, n = 10, f = 15, fs = 125.0, padding = 50):

    appg = np.gradient(vppg)
    b,a = scipy.signal.butter(n, f, 'lp', fs=125.0)
    sos =scipy.signal.tf2sos(b,a)
    w, gd = scipy.signal.group_delay((b, a))
    gd_mean = int(np.round(np.mean(gd)))
    filtered = scipy.signal.sosfilt(sos, appg)
    delay = 2*gd_mean + 1

    return filtered[padding + delay: - padding +delay], appg[padding:padding]

def get_SBP_DBP(ABP, prom= 4.2, verbose=0):
    std = np.std(ABP)
    mean = np.mean(ABP)
    #calcoliamo picchi
    #calc_prom = mean + ((np.max(x) - mean)/2.0) - np.min(x)
    calc_prom = abs(np.median(ABP) - np.min(ABP)-0.1)
    peaks_max_location, _ = scipy.signal.find_peaks(ABP, prominence= calc_prom, distance = 60)
    peaks_min_location, _ = scipy.signal.find_peaks(-ABP, prominence= calc_prom, distance = 60)
    SBP_peaks = ABP[peaks_max_location]
    DBP_peaks = ABP[peaks_min_location]
    sbp = np.mean(SBP_peaks)
    dbp = np.mean(DBP_peaks)
    error_sbp = np.std(SBP_peaks)
    error_dbp = np.std(DBP_peaks)
    if verbose==1:
        plt.plot(ABP)
        plt.plot(peaks_max_location, ABP[peaks_max_location], "x")
        plt.plot(peaks_min_location, ABP[peaks_min_location], "x")
    return (sbp, dbp, error_sbp, error_dbp)

def find_peaks_valleys(x, prom = 0.5):
     if len(x)<1000:
          print("only ", len(x), " elements. Discarded")
          return 0
     peaks_max_X, _ = scipy.signal.find_peaks(x, prominence= prom, distance = 30)
     peaks_max_Y = x[peaks_max_X]
     peaks_min_X, _ = scipy.signal.find_peaks(-x, prominence= prom, distance = 30)
     peaks_min_Y = x[peaks_min_X]
     plt.figure()
     plt.plot(x)
     plt.plot(peaks_max_X[:len(peaks_max_Y)],peaks_max_Y, "x")
     plt.plot(peaks_min_X[:len(peaks_min_Y)],peaks_min_Y, "x")


def identify_blocks(blocks_of_interest):
    blocks = []
    current_block = []
    for i, val in enumerate(blocks_of_interest):
        if val == 0.1:
            current_block.append(i)
        else:
            if current_block:
                blocks.append(current_block)
                current_block = []
    if current_block:  # Add the last block if it hasn't been added yet
        blocks.append(current_block)
    return blocks


def find_peaks_custom(PPG_signal, F1, W1, W2, beta):
    S_peaks = []
    Filtered = butter_bandpass_filter(PPG_signal, 0.5, 8, 125,order=2)
    Clipped = clip_signal(Filtered)
    Q_clipped = square_signal(Clipped)
    MA_peak = moving_average(Q_clipped, W1)
    MA_beat = moving_average(Q_clipped, W2)
    zeta = np.mean(Q_clipped)
    alfa = beta * zeta + MA_beat
    THR1 = MA_beat + alfa
    BlocksOfInterest = np.zeros(len(MA_peak))
    
    for n in range(len(MA_peak)):
        if MA_peak[n] > THR1[n]:
            BlocksOfInterest[n] = 0.1
        else:
            BlocksOfInterest[n] = 0

    # Identify blocks based on BlocksOfInterest
    Blocks = identify_blocks(BlocksOfInterest)

    THR2 = W1

    for block in Blocks:  # Assuming Blocks is a list of segments, each containing indices
        width = len(block)
        if width >= THR2:
            max_val_index = np.argmax(PPG_signal[block])
            S_peaks.append(block[max_val_index])
        else:
            # Ignore block
            continue
    
    return S_peaks

# Assuming you have PPG_signal, F1, F2, W1, W2, beta, and zeta
# S_peaks = find_peaks(PPG_signal, F1, F2, W1, W2, beta, zeta)

def chebyshev_nth(x, N = 8, fc = [2,59], fs= 125):
    #N = 8                    # Filter order
               # Sampling frequency (Hz)             # Cut-off frequency (Hz)
    rp = 1                   # Passband ripple (dB)

    # Compute filter coefficients
    b, a = scipy.signal.cheby1(N, rp, fc, btype='bandpass', fs=fs)
    w, gd = scipy.signal.group_delay((b, a))
    gd_mean = int(np.round(np.mean(gd)))
    delay = gd_mean + 1
    #print("delay_cheb",delay)
    output = scipy.signal.filtfilt(b, a, x)
    return output


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter to the data.

    Parameters:
        data (array): The signal data to be filtered.
        lowcut (float): The lower cutoff frequency in Hz.
        highcut (float): The higher cutoff frequency in Hz.
        fs (float): The sampling frequency in Hz.
        order (int): The order of the filter.

    Returns:
        array: The filtered data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

# Clipping function
def clip_signal(signal):
    return np.maximum(signal, 0)

# Squaring function
def square_signal(signal):
    return signal ** 2

# Moving Average function
def moving_average(y, window_size):
    return np.convolve(y, np.ones(window_size) / window_size, mode='same')

def find_intervals_and_max_points(ss, beta, w1=70, w2=200, f1=20,f2=50, lim = 1):
    intervals = []
    max_points = []
    start_idx = None
    
    s = np.array(ss)
    s = butter_bandpass_filter(s, f1, f2, 125)  # Assuming butter_bandpass_filter is defined
    B = np.maximum(s, 0)
    B = B * B
    MA_peak = moving_average(B, w1)  # Assuming moving_average is defined
    MA_beat = moving_average(B, w2)
    
    z = np.mean(B)
    alfa = z * beta
    th1 = MA_beat + alfa
    
    for i in range(len(MA_peak)):
        if MA_peak[i] > th1[i]:
            if start_idx is None:
                start_idx = i
                max_val = ss[i]
                max_idx = i
            else:
                if ss[i] > max_val:
                    max_val = ss[i]
                    max_idx = i
        else:
            if start_idx is not None:

                if i - 1 - start_idx >= w1/lim:  # Check if the interval length is >= w1
                    intervals.append((start_idx, i - 1))
                    max_points.append(max_idx)                  
                start_idx = None
                
    # Handle the case where the last interval extends to the end of the list
    if start_idx is not None and len(MA_peak) - 1 - start_idx >= w1:
        intervals.append((start_idx, len(MA_peak) - 1))
        max_points.append(max_idx)
        
    return intervals, max_points, MA_peak, th1
