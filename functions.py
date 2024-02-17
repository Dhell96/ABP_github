import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
import json
import os
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, correlate
from scipy import interpolate

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


def synchronize_signals_peaks_old(s1,s2):
    calc_prom_1 = abs(np.median(s1) - np.min(s1))
    calc_prom_2 = abs(np.median(s2) - np.min(s2))
    peaks_1, _ = scipy.signal.find_peaks(s1, prominence=calc_prom_1, distance = 50)
    peaks_2, _ = scipy.signal.find_peaks(s2, prominence=calc_prom_2, distance = 50)
    m = np.min([len(peaks_1), len(peaks_2)])
    time_delay = int(np.median(peaks_1[:m] - peaks_2[:m]))
    synchronized_signal2 = np.roll(s2, time_delay)
    return synchronized_signal2, time_delay

def synchronize_signals_peaks(s1, s2, plotting = 0):
    intervals1, peaks_1, m1, t1, b1, ss1 = find_intervals_and_max_points(s1)
    intervals2, peaks_2, m2, t2, b2, ss2 = find_intervals_and_max_points(s2)
    m = np.min([len(peaks_1), len(peaks_2)])
    
    # Convert lists to NumPy arrays for element-wise subtraction
    peaks_1_array = np.array(peaks_1[:m])
    peaks_2_array = np.array(peaks_2[:m])
    
    time_delay = int(np.median(peaks_1_array - peaks_2_array))
    synchronized_signal2 = np.roll(s2, time_delay)
    
    if plotting:
        plt.figure(figsize=(14,3))
        plt.plot(s1)
        plt.plot(peaks_1, s1[peaks_1],"x")

        plt.figure(figsize=(14,3))
        plt.plot(s2)
        plt.plot(peaks_2, s2[peaks_2],"x")

        plt.show()



    return synchronized_signal2, time_delay



def gaussian(x, mu, sigma, a = 1):
    #x, alpha, mu, signa
    return a *np.exp(-0.5 * ((x - mu) / sigma) ** 2)

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

def Arterial_Blood_Pressure(signal):
    intervals, max_points,m,t,b,s = find_intervals_and_max_points(+signal)
    intervals1, min_points, m1, t1, b1,s1 = find_intervals_and_max_points(-signal)
    SBP = np.mean(signal[max_points])
    DBP = np.mean(signal[min_points])
    eSBP = np.std(signal[max_points])
    eDBP = np.std(signal[min_points])

    return (SBP, DBP, eSBP, eDBP)


def get_SBP_DBP(ABP, prom= 4.2, verbose=0, points = 0):
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
    if points == 0:
        return (sbp, dbp, error_sbp, error_dbp)
    if points ==1:
        return (sbp, dbp, error_sbp, error_dbp, [peaks_max_location, ABP[peaks_max_location]],[peaks_min_location, ABP[peaks_min_location]])

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

#(+signal, 0.1, w1 = 20, w2 = 124, lim = 1.2, f1= 0.5, f2 = 50, thr = 0.0)
def find_intervals_and_max_points_old(ss, beta = 0.1, w1=20, w2=124, f1=0.5,f2=50, lim = 1.2, thr= 0):
    intervals = []
    max_points = []
    start_idx = None
    
    s = np.array(ss)
    s = butter_bandpass_filter(s, f1, f2, 125)  # Assuming butter_bandpass_filter is defined
    #r = np.mean(s) + 1.0*(np.max(s) - np.mean(s))/3.0
    r=thr
    B = np.maximum(s, r ) - r
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
        
    return intervals, max_points, MA_peak, th1, B,s

def find_intervals_and_max_points(ss, beta = 0.1, w1=20, w2=124, f1=0.5, f2=50, lim = 1.2, thr= 0, distance = 50):
    intervals = []
    max_points = []
    start_idx = None
    last_max_idx = None  # Variable to keep track of the index of the last max peak
    
    s = np.array(ss)
    s = butter_bandpass_filter(s, f1, f2, 125)  # Assuming butter_bandpass_filter is defined
    r = thr
    B = np.maximum(s, r) - r
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
                    if last_max_idx is None or max_idx - last_max_idx >= distance:  # Check the distance condition
                        intervals.append((start_idx, i - 1))
                        max_points.append(max_idx)
                        last_max_idx = max_idx  # Update last_max_idx
                start_idx = None
                
    # Handle the case where the last interval extends to the end of the list
    if start_idx is not None and len(MA_peak) - 1 - start_idx >= w1:
        if last_max_idx is None or max_idx - last_max_idx >= distance:  # Check the distance condition
            intervals.append((start_idx, len(MA_peak) - 1))
            max_points.append(max_idx)
        
    return intervals, max_points, MA_peak, th1, B, s





def renorm_0_1(x):
  xx = np.copy(x)
  xx= (xx- np.min(xx))/(np.max(xx) - np.min(xx))
  return xx

def renorm_all(s1):
    s = np.array(s1)
    for j in range(s.shape[1]):
        s[:,j] = renorm_0_1(s[:,j])
    return s


def plot_3channels(s1,xl = (None, None), w = 250, renorm = 0, threshold = 3):

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    
    s = np.array(s1)

    if renorm ==1:
        #s[:,0] = renorm_0_1(s[:,0])
        #s[:,1] = renorm_0_1(s[:,1])
        #s[:,2] = renorm_0_1(s[:,2])
        s = renorm_all(s)

    # Plot 'I'
    axs[0].plot(s[:,0])
    axs[0].set_title('PLETH Signal')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_xlim(xl)
    
    # Plot 'PLETH'
    axs[1].plot(s[:,1])
    axs[1].set_title('I Signal')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Amplitude')

    axs[1].set_xlim(xl)
    
    # Plot 'ABP'
    axs[2].plot(s[:,2])
    axs[2].set_title('ABP Signal')
    axs[2].set_xlabel('Samples')
    axs[2].set_ylabel('Amplitude')
    axs[2].set_xlim(xl)
    
    
    # Show the plot
    plt.tight_layout()
    


def extract_data_json(file_path):
    with open(file_path, 'r') as filee:
        data = json.load(filee)
        ecg = np.array(data['ECGSamplesf'])
        ppg = np.array(data['PPGSamplesREDf'])
        ecg_nf = np.array(data['ECGSamples'])
        ppg_nf = np.array(data['PPGSamplesRED'])
    return ecg, ppg, ecg_nf, ppg_nf


def resample_signal(s, old_fs, new_fs):
    """
    Resamples a s from `old_fs` to `new_fs`.

    Args:
    s (array): The s to resample.
    old_fs (float): The original sampling frequency of the s.
    new_fs (float): The desired sampling frequency of the resampled s.

    Returns:
    array: The resampled s.
    """
    n_samples = int(len(s) * new_fs / old_fs)
    #print(len(s))
    resampled_s = signal.resample(s, n_samples)
    return resampled_s


#let's plot the json fouriers 
def definitive_fft(si, freq, plotting = 0, xl = (None, None), lab = "", renorm = 1, alpha = 1.0, squared = 1):
    fourier_N = fourier_plot(si)
    nyquist_freq = freq / 2
    t_json_N = np.linspace(0,freq, len(fourier_N))

    if squared:
        fourier_N = fourier_N*fourier_N
    
    

    if plotting == 1:
        if renorm:
            plt.plot(t_json_N,zero_one_renorm_single(np.abs(fourier_N)), label = lab, alpha= alpha)
            plt.xlim(xl)
        else:
            plt.plot(t_json_N,np.abs(fourier_N), label = lab, alpha= alpha)
            plt.xlim(xl)
    if renorm:
        return t_json_N, zero_one_renorm_single(np.abs(fourier_N))
    else:
        return t_json_N, 2.0*(freq*len(fourier_N))


def zeroing(data):
	val_m=np.mean(data)
	data[:] = [x -val_m for x in data]
	return data


def notch(noisySignal, notch_freq, Q, samp_freq = 125):

	# Create/view notch filter
	
	#notch_freq = 50.0 # Frequency to be removed from signal (Hz)
	
	quality_factor = Q # Quality factor

	# Design a notch filter using signal.iirnotch
	b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)

	# Compute magnitude response of the designed filter
	#freq, h = signal.freqz(b_notch, a_notch, fs=samp_freq)
	
	return signal.filtfilt(b_notch, a_notch, noisySignal, method="gust")


def butter_lowpass_filter(data, cutoff, fs, order=4):
	b, a = signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
	y = signal.filtfilt(b, a, data, method="gust")
	return y

#cutoff: desired cutoff frequency of the filter, Hz
def LPF( data, cutoff, order, samp_freq = 125):
		
	# Filter requirements.
	fs = samp_freq       # sample rate, Hz  


	# Filter the data, and plot both the original and filtered signals.
	y = butter_lowpass_filter(data, cutoff, fs, order)

	#plt.show()
	return y
	
	
	
def HPF(data, cutoff, order, samp_freq = 125):
	
	fs = samp_freq       # sample rate, Hz  
		
	b, a = signal.butter(order, cutoff, fs=fs, btype='highpass', analog=False)
	y = signal.filtfilt(b, a, data, method="gust")
	
	return y


def Hanning_convolution(vector, windowSize):
	window = np.hanning(windowSize)
	window = window / window.sum()

	# filter the data using convolution
	filtered = np.convolve(window, vector, mode='valid')
	return filtered


def smooth_signal(data, sample_rate, window_length=None, polyorder=3):
    '''smooths given signal using savitzky-golay filter

    Function that smooths data using savitzky-golay filter using default settings.

    Functionality requested by Eirik Svendsen. Added since 1.2.4

    Parameters
    ----------
    data : 1d array or list
        array or list containing the data to be filtered

    sample_rate : int or float
        the sample rate with which data is sampled

    window_length : int or None
        window length parameter for savitzky-golay filter, see Scipy.signal.savgol_filter docs.
        Must be odd, if an even int is given, one will be added to make it uneven.
        default : 0.1  * sample_rate

    polyorder : int
        the order of the polynomial fitted to the signal. See scipy.signal.savgol_filter docs.
        default : 3

    Returns
    -------
    smoothed : 1d array
        array containing the smoothed data

    Examples
    --------
    Given a fictional signal, a smoothed signal can be obtained by smooth_signal():

    >>> x = [1, 3, 4, 5, 6, 7, 5, 3, 1, 1]
    >>> smoothed = smooth_signal(x, sample_rate = 2, window_length=4, polyorder=2)
    >>> np.around(smoothed[0:4], 3)
    array([1.114, 2.743, 4.086, 5.   ])

    If you don't specify the window_length, it is computed to be 10% of the 
    sample rate (+1 if needed to make odd)
    >>> import heartpy as hp
    >>> data, timer = hp.load_exampledata(0)
    >>> smoothed = smooth_signal(data, sample_rate = 100)

    '''

    if window_length == None:
        window_length = sample_rate // 10
        
    if window_length % 2 == 0 or window_length == 0: window_length += 1

    smoothed = signal.savgol_filter(data, window_length = window_length,
                             polyorder = polyorder)

    return smoothed



def new_filter(ex, samp_freq= 125, plotting = 0):
    #let's apply some filters
    noisySignal=ex
    noisySignal = zeroing(noisySignal)
    outputSignal = notch(noisySignal, 50, 10)

    #outputSignal = notch(outputSignal, 0.05, 5)

    outputSignal = notch(outputSignal, 0.1, 2) #In order to work together with HPF

    #outputSignal = new_filter.hampel_filter(outputSignal, filtsize=50)

    outputSignal = LPF( outputSignal, 30, order=4)

    outputSignal = HPF( outputSignal, 0.5, order=4)

    outputSignal = Hanning_convolution(outputSignal, int(samp_freq/80))

    #outputSignal = filters.moving_average(outputSignal, int(samp_freq/80))

    outputSignal = smooth_signal(outputSignal, samp_freq,polyorder=8 )

    #_mean= filters.moving_average(outputSignal, int(samp_freq))

    return outputSignal




def find_n_peaks(frequencies, spectrum, n=5):
    peaks, _ = signal.find_peaks(spectrum, distance=10)
    sorted_peaks = sorted(peaks, key=lambda x: spectrum[x], reverse=True)[:n]
    return sorted_peaks


def filter_signal(signal, fs, n_peaks=1, band_width=0.45):
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Perform FFT and find the n most important peaks
    fourier_transform = np.fft.fft(signal)
    spectrum = np.abs(fourier_transform)
    important_peaks = find_n_peaks(freqs, spectrum, n_peaks)
    
    # Generate Gaussian filters
    filtered_signal_fft = np.zeros(N, dtype=complex)
    for peak in important_peaks:
        peak_freq = freqs[peak]
        g_filter = gaussian(freqs, peak_freq, band_width)
        filtered_signal_fft += fourier_transform * g_filter
    
    # Perform inverse FFT to get filtered signal
    filtered_signal = np.fft.ifft(filtered_signal_fft)
    
    return np.real(filtered_signal)


def find_n_peaks_band(frequencies, spectrum, n=1, freq_min=None, freq_max=None, dist = 10):
    # Create masks based on frequency range
    mask = np.ones_like(frequencies, dtype=bool)
    if freq_min is not None:
        mask &= (frequencies >= freq_min)
    if freq_max is not None:
        mask &= (frequencies <= freq_max)
        
    # Filter both frequencies and spectrum arrays
    filtered_frequencies = frequencies[mask]
    filtered_spectrum = spectrum[mask]
    
    # Find peaks on the filtered spectrum
    peaks, _ = signal.find_peaks(filtered_spectrum, distance = dist)
    
    # Sort peaks by magnitude and select the top n
    sorted_peaks = sorted(peaks, key=lambda x: filtered_spectrum[x], reverse=True)[:n]
    
    # Translate back to original indices if needed
    original_indices = np.where(mask)[0][sorted_peaks]
    
    return original_indices




def filter_signal_band(signal, fs, n_peaks=1, band_width=0.20, freq_min=None, freq_max=None):
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Perform FFT and find the n most important peaks within the frequency range
    fourier_transform = np.fft.fft(signal)
    spectrum = np.abs(fourier_transform)
    important_peaks = find_n_peaks_band(freqs, spectrum, n_peaks, freq_min, freq_max)
    
    # Generate Gaussian filters
    filtered_signal_fft = np.zeros(N, dtype=complex)
    for peak in important_peaks:
        peak_freq = freqs[peak]
        g_filter = gaussian(freqs, peak_freq, band_width)
        filtered_signal_fft += fourier_transform * g_filter
    
    # Perform inverse FFT to get filtered signal
    filtered_signal = np.fft.ifft(filtered_signal_fft)
    
    return np.real(filtered_signal)



def add_gaussian_noise(signal, variance):
    noise = np.random.randn(*signal.shape) * variance
    return signal + noise


def add_jitter(signal, amount):
    jitter = np.random.normal(0, amount, signal.shape[0])
    x = np.arange(signal.shape[0])
    new_signal = scipy.interpolate.interp1d(x + jitter, signal, axis=0, bounds_error=False, fill_value="extrapolate")(x)
    return new_signal


def add_sine_noise(data, frequency, amplitude, sample_rate, phase=0):
    # Generate time values
    t = np.arange(0, len(data)/sample_rate, 1/sample_rate)
    # Generate the sine wave
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    # Add the sine wave to the data
    noisy_data = data + sine_wave[:len(data)]
    return noisy_data


def resize_signal(x, fs, duration1, duration2):
    #x:intero segnale
    #fs: frquenza di campionamento
    #duration1: durata originale
    #duration2: nuova durata
    points = len(x)
    t1 = np.linspace(0, duration1, int(fs*duration1), endpoint=False)
    t2 = np.linspace(0, duration2, int(fs*duration2), endpoint=False)
    x_resampled = scipy.signal.resample(x, int(fs*duration2))
    #plt.plot(x, label='Original')
    #plt.plot(x_resampled, label='Shifted')
    #plt.legend()
    #plt.show()
    return x_resampled




def calculate_CSP_index(psd_values, f_index):
    return sum(psd_values[:f_index + 1])



def calculate_CSP_frequence(psd_values, fs, f):
    N_s = int(len(psd_values)/2.0)
    f_max = fs/2.0
    if isinstance(f, (int, float)):  # Single frequency value
        f_index = int((N_s * f) / f_max)
        return calculate_CSP_index(psd_values, f_index)
    elif isinstance(f, (list, tuple, np.ndarray)):  # Array of frequency values
        csp_values = []
        for freq in f:
            f_index = int((N_s * freq) / f_max)
            csp_values.append(calculate_CSP_index(psd_values, f_index))
        return csp_values
    

import numpy as np

def find_closest_x(y_values, x_values, target_y):
    """
    Find the x value(s) that correspond to the closest y value(s) to the target_y.
    
    Parameters:
    y_values (list or np.array): Array of y values
    x_values (list or np.array): Array of x values
    target_y (float, list, tuple, np.array): The y value(s) for which you want to find the corresponding x value(s)
    
    Returns:
    float or list: The x value(s) corresponding to the closest y value(s) to target_y
    """
    
    y_arr = np.array(y_values)
    
    if isinstance(target_y, (int, float)):  # Single target_y value
        closest_y_index = np.argmin(np.abs(y_arr - target_y))
        return x_values[closest_y_index]
    
    elif isinstance(target_y, (list, tuple, np.ndarray)):  # Array of target_y values
        closest_x_values = []
        for ty in target_y:
            closest_y_index = np.argmin(np.abs(y_arr - ty))
            closest_x_values.append(x_values[closest_y_index])
        return closest_x_values


def find_files_extension(hyper_folder, ext):
    matching_filenames = []

# Iterate through all files in the specified directory
    for filename in os.listdir(hyper_folder):
        # Check if the file has a .json extension
        if filename.endswith(ext):
            matching_filenames.append(filename)

    return matching_filenames



def normalize_peaks_mins(signal, n_signal = 0, plotting = 0, kind = "quadratic"):
    #if n_signal == 0:
    #    intervals, max_points,_,_ = find_intervals_and_max_points(signal, 0.02, w1 = 25, w2 = 230, f1= 0.5, f2= 8, lim=1.2)
    #    intervals, min_points,_,_ = find_intervals_and_max_points(-signal, 0.02, w1 = 25, w2 = 230, f1= 0.5, f2= 8, lim=1.2)
    #if n_signal==1:
    #    intervals, max_points,_,_ = find_intervals_and_max_points(signal, 0.02, w1 = 12, w2 = 124, lim = 1.2, f1= 20, f2 = 50)
    #    intervals, min_points,_,_ = find_intervals_and_max_points(-signal, 0.02, w1 = 12, w2 = 124, lim = 1.2, f1= 20, f2 = 50)   

    #if n_signal==2:
    #    intervals, max_points,_,_ = find_intervals_and_max_points(signal, 0.02, w1 = 20, w2 = 230, f1= 0.5, f2= 8, lim = 1.2)
    #    intervals, min_points,_,_ = find_intervals_and_max_points(-signal, 0.02, w1 = 20, w2 = 230, f1= 0.5, f2= 8, lim = 1.2) 

    intervals, max_points,_,_,_,_ = find_intervals_and_max_points(signal, 0.02, w1 = 20, w2 = 124, lim = 1.2, f1= 0.5, f2 = 30, thr = 0.0)
    intervals, min_points,_,_,_,_ = find_intervals_and_max_points(-signal, 0.02, w1 = 20, w2 = 124, lim = 1.2, f1= 0.5, f2 = 30, thr = 0.0)



    # Generate a sample signal
    x = np.linspace(0, len(signal), len(signal))
    y = signal

    # Interpolate to find the envelope
    max_interp = interp1d(x[max_points], y[max_points], kind=kind, fill_value='extrapolate')
    min_interp = interp1d(x[min_points], y[min_points], kind=kind, fill_value='extrapolate')

    max_envelope = max_interp(x)
    min_envelope = min_interp(x)



    # Interpolate to find the envelope
    #max_spline = CubicSpline(x[max_points], y[max_points])
    #min_spline = CubicSpline(x[min_points], y[min_points])

    #max_envelope = max_spline(x)
    #min_envelope = min_spline(x)

    # Normalize the signal
    normalized_signal = (y - min_envelope) / (max_envelope - min_envelope)

    if plotting:

        # Plotting
        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        plt.plot(x, y, label='Original Signal')
        plt.plot(x, max_envelope, '--', label='Max Envelope')
        plt.plot(x, min_envelope, '--', label='Min Envelope')
        print(max_points)
        plt.plot(max_points, signal[max_points],"x")
        plt.plot(min_points, signal[min_points],"o")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(x, normalized_signal, label='Normalized Signal')
        plt.legend()

        plt.show()

    return normalized_signal, max_envelope, min_envelope





def simplify_ECGs(r):

  #r = functions.normalize_peaks_mins(r)
  
  

  s1 = zero_one_renorm_single(np.maximum(z_renorm(r), 0))
  r_peaks, _ = find_peaks(s1, distance=125*0.6, height=np.max(s1)*0.5)

  secondary_peaks = []


  # Define the segment of the signal between two R-peaks

  segment = s1[0:r_peaks[0]]
  peaks, _ = find_peaks(segment, distance = 5)
  d = 0
  peak_heights = segment[peaks]
  if len(peak_heights) > 2:
        two_highest = peaks[np.argsort(peak_heights)[-2:]] + d # Get indices of the two highest peaks
  else:
      two_highest = peaks + d # If there are two or less, take all

  secondary_peaks.extend(two_highest)




  for P in range(len(r_peaks) -1 ):

    segment = s1[r_peaks[P]:r_peaks[P + 1]]
    peaks, _ = find_peaks(segment, distance = 5)
    d = r_peaks[P]
    peak_heights = segment[peaks]
    if len(peak_heights) > 2:
        two_highest = peaks[np.argsort(peak_heights)[-2:]] + d # Get indices of the two highest peaks
    else:
        two_highest = peaks + d # If there are two or less, take all

    secondary_peaks.extend(two_highest)


  segment = s1[r_peaks[-1]:]
  peaks, _ = find_peaks(segment, distance = 5)
  d = r_peaks[-1]
  peak_heights = segment[peaks]
  if len(peak_heights) > 2:
        two_highest = peaks[np.argsort(peak_heights)[-2:]] + d # Get indices of the two highest peaks
  else:
      two_highest = peaks + d # If there are two or less, take all

  secondary_peaks.extend(two_highest)

  secondary_peaks.extend([0])
  secondary_peaks.extend([len(r)-1])

  all_peaks = np.sort(np.concatenate((np.array(r_peaks), np.array(secondary_peaks))))

  # Calculate midpoints between adjacent elements in all_peaks
  midpoints = (all_peaks[:-1] + all_peaks[1:]) // 2

  # Create a new array with space for the original values, midpoints, and zeroes
  expanded_peaks_with_zeros = np.zeros(len(all_peaks) + len(midpoints), dtype=all_peaks.dtype)

  # Place the original all_peaks values and midpoints into the new array
  expanded_peaks_with_zeros[::2] = all_peaks
  expanded_peaks_with_zeros[1::2] = midpoints

  f = interpolate.interp1d(expanded_peaks_with_zeros, s1[expanded_peaks_with_zeros], kind='linear')
  x_new = np.arange(0, len(r))
  y_new = f(x_new)

  # expanded_peaks_with_zeros now contains the original points, with midpoints in between
  return y_new, expanded_peaks_with_zeros, r_peaks


def calculate_hr(ecg_data,time, sampling_rate=125):
    # Assume ecg_data is a 1D numpy array containing ECG signal values.

    # Step 1: Preprocess (optional for simplification)
    # Here, you might apply filtering to ecg_data if needed.

    # Step 2: Detect R-peaks
    # This is a simplistic approach; for better results, consider more sophisticated algorithms.
    # Adjust the height and distance based on your ECG data characteristics.
    peaks, _ = find_peaks(ecg_data, distance=sampling_rate/2)

    # Step 3: Calculate the heart rate
    num_peaks = len(peaks)
    duration_in_minutes = time / 60  # 8 seconds expressed in minutes
    heart_rate = num_peaks / duration_in_minutes

    return heart_rate, peaks