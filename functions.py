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
    try:
      time_delay = int(np.median(peaks_1[:m] - peaks_2[:m]))
    except:
       plt.plot(s1)
       plt.plot(s2)
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



