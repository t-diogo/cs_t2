# 2nd order sigma delta modulator

import time
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy import signal


###################### Functions ######################


# Function to implement the Modulator
@nb.jit(nopython=True)
def second_order_sigma_delta_modulator(vin):

    npoints = len(vin)
    dout = np.zeros(npoints)
    e1 = np.zeros(npoints)  # First integrator error
    e2 = np.zeros(npoints)  # Second integrator error
    x1 = np.zeros(npoints)  # First integrator state
    x2 = np.zeros(npoints)  # Second integrator state

    # Init
    x1[0]=1e-12
    x2[0]=1e-12

    # Modulator loop
    for n in range(1, npoints):
        # Quantizer
        if x2[n-1] >= 0:
            dout[n] = 1
        else:
            dout[n] = -1

        # First integrator
        e1[n] = vin[n] - dout[n] * (Vr*b1)
        x1[n] = k1 * e1[n] + B1 * x1[n-1]
        
        # Second integrator
        e2[n] = x1[n-1] - dout[n] * (Vr*b2)
        x2[n] = k2 * e2[n] + B2 * x2[n-1]
    
    return dout


# Function to implement the third-order sinc filter
def third_order_sinc_filter(dout, N):
    # Initialize integrator variables
    int1 = np.cumsum(dout)
    int2 = np.cumsum(int1)
    int3 = np.cumsum(int2)

    # Decimation
    decim = int3[::N]

    # Differentiation
    diff1 = np.diff(np.concatenate(([0], decim)))
    diff2 = np.diff(np.concatenate(([0], diff1)))
    diff3 = np.diff(np.concatenate(([0], diff2)))

    # Gain correction
    filtered_signal = diff3 / (N ** 3)

    return filtered_signal


###################### Parameters ######################


# Modulator parameters
Vr = 1

OSR = 66.5          # Oversampling ratio
B = 20e3            # Signal bandwidth
Fs = 2*B*OSR        # Sampling frequency


# Simulation parameters
np_points = 50000       # Number of points

sinad = False           # do SINAD analysis (True) or regular input signal (False) - f)
real = False            # test with integrators with finite gain (True) or infinite gain (False) - g)

# Define coefficients
k1 = 1
k2 = 1
b1 = 1
b2 = 2
B1 = 1
B2 = 1

# Finite gain - A = 40dB = 100
A = 100
if real:
    B1 = 1 - (k1/A) * (1 + 1/Vr)
    B2 = 1 - (k2/A) * (1 + 1/Vr)

# Input signal - sine wave 1kHz
f = 1e3
f = round((f/Fs) * np_points ) * (Fs/np_points)
print(f'f={f}Hz')

# Time vector
t = np.arange(np_points) / Fs

# Input Signal parameters
if sinad:
    # SINAD analysis - different noise levels and amplitudes
    Ain_values = np.logspace(np.log10(1e-4), np.log10(np.sqrt(2)), 1000)    
    noise_levels = [1e-6, 1e-5, 1e-4, 1e-3]                                 # Multiple noise levels
    noise_levels = [1e-4]                                                   # Single (adequate) noise level
else:
    # Regular input signal - only one amplitude and noise level
    Ain_values = [Vr/2]
    noise_levels = [1e-4]


# Initialize the results storage
results_nofilter = {noise: [] for noise in noise_levels}
results_filtered = {noise: [] for noise in noise_levels}



###################### Simulation loop ######################


# Initialize variables for the modulator
e1 = np.zeros(np_points)        # First integrator error
e2 = np.zeros(np_points)        # Second integrator error
x1 = np.zeros(np_points)        # First integrator state
x2 = np.zeros(np_points)        # Second integrator state
dout = np.zeros(np_points)      # Quantizer output

# Initialize variables for the decimation filter
N = int (OSR)
len_dec = (len(dout) // N) + 1

# Initialize integrator variables
int1 = np.zeros(len(dout))
int2 = np.zeros(len(dout))
int3 = np.zeros(len(dout))

# Initialize differentiator variables
diff1 = np.zeros(len_dec)
diff2 = np.zeros(len_dec)
diff3 = np.zeros(len_dec)


# Simulation loop

start_time = time.time()

for noise in noise_levels:
    print(f'Noise level: {noise*1e6:.0f}uV')
    for Ain in Ain_values:
        # Create input signal
        vin = Ain * np.sin(2 * np.pi * f * t) + np.random.randn(np_points) * noise

        # Modulator
        dout = second_order_sigma_delta_modulator(vin)

        # No filter
        fin = f         # input signal frequency stays the same
        fs = Fs         # fs stays the same
        osr = OSR       # osr stays the same

        # Calculate SNDR at the output of modulator
        Nfft = len(dout)
        window = signal.windows.blackmanharris(Nfft)    # blackmanharris window
        doutw = np.fft.fft(np.multiply(dout, window))
        doutp = np.abs(np.multiply(doutw,doutw.conjugate())/((Nfft)**2)*2)

        index = int((Nfft * fin) / fs)
        Ps_nofilter = np.abs(np.sum(doutp[index - 4: index + 4]))  # blackmanharris offset = 4
        Pn_nofilter = np.abs(np.sum(doutp[:round(Nfft / (2 * osr))])) - Ps_nofilter
        SNDR_nofilter = 10 * np.log10(Ps_nofilter / Pn_nofilter)


        # Decimation filter
        doutf = third_order_sinc_filter(dout, N)
        fin = f         # input signal frequency stays the same
        fs = Fs/N       # fs = 2B = nyquist frequency
        osr = OSR/N     # osr has to be divided to maintain the ratio with number of points in dout

        # Calculate SNDR at the output of the filter
        Nfft = len(doutf)
        window = signal.windows.blackmanharris(Nfft)    # blackmanharris window
        doutfw = np.fft.fft(np.multiply(doutf, window))
        doutfp = np.abs(np.multiply(doutfw,doutfw.conjugate())/((Nfft)**2)*2)

        index = int((Nfft * fin) / fs)
        Ps_filtered = np.abs(np.sum(doutfp[index - 4: index + 4]))  # blackmanharris offset = 4
        Pn_filtered = np.abs(np.sum(doutfp[:round(Nfft / (2 * osr))])) - Ps_filtered
        SNDR_filtered = 10 * np.log10(Ps_filtered / Pn_filtered)
        
        # Store results
        if sinad:
            results_nofilter[noise].append(SNDR_nofilter)
            results_filtered[noise].append(SNDR_filtered)


stop_time = time.time()
print(f'Elapsed time: {stop_time - start_time:.2f} s')


# Plots

if sinad:

    # SINAD results for non-filtered output
    plt.figure(figsize=(10, 6))
    for noise in noise_levels:
        plt.plot(20 * np.log10(Ain_values / np.sqrt(2)), results_nofilter[noise], label=f'Vnin={noise*1e6:.0f}uV')

        # Find the max SNDR and corresponding amplitude
        max_sndr_nofilter = np.max(results_nofilter[noise])
        max_sndr_index_nofilter = np.argmax(results_nofilter[noise])
        max_sndr_amplitude_nofilter = Ain_values[max_sndr_index_nofilter]
    
    plt.title(f'No Filter - SINAD for OSR={OSR}\nMax SNDR: {max_sndr_nofilter:.2f} dB with Ain: {max_sndr_amplitude_nofilter:.2f} V')
    plt.xlabel('Ain [dBV]')
    plt.ylabel('SNDR [dB]')
    plt.grid(True)
    plt.legend()


    # SINAD results for filtered output- Dynamic Range
    plt.figure(figsize=(10, 6))
    for noise in noise_levels:
        plt.plot(20 * np.log10(Ain_values / np.sqrt(2)), results_filtered[noise], label=f'Vnin={noise*1e6:.0f}uV')

        # Find the max SNDR and corresponding amplitude
        max_sndr_filtered = np.max(results_filtered[noise])
        max_sndr_index_filtered = np.argmax(results_filtered[noise])
        max_sndr_amplitude_filtered = Ain_values[max_sndr_index_filtered]

    plt.title(f'Sinc filter - SINAD for OSR={OSR}\nMax SNDR: {max_sndr_filtered:.2f} dB with Ain: {max_sndr_amplitude_filtered:.2f} V')
    plt.xlabel('Ain [dBV]')
    plt.ylabel('SNDR [dB]')
    plt.grid(True)
    plt.legend()

else:
    # Regular input signal

    # Output spectrum for non-filtered output

    fin = f         # input signal frequency stays the same
    fs = Fs         # fs stays the same
    osr = OSR       # osr stays the same

    text_SNDR=f'SNDR={SNDR_nofilter:.2f}'+f' dB\n Ps= {10*np.log10(abs(Ps_nofilter))+5.88:.2f} dB'
    print('Ps=',Ps_nofilter,'PN=',Pn_nofilter,text_SNDR)
    Nfft = len(dout)
    freq = np.linspace(0, 1, Nfft)
    plt.figure()
    plt.semilogx(freq[0:int(Nfft/2)],10*np.log10(abs(doutp[0:int(Nfft/2)])))
    plt.title(f'No filter - Output spectrum SNDR={SNDR_nofilter:.2f} dB')
    plt.xlabel('Frequency [f/Fs]')
    plt.ylabel('Amplitude [dBr]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(1/(2*osr), color='green') # Bandwidth
    plt.annotate(text_SNDR,
             xy = (fin/fs, 10*np.log10(abs(doutp[int(fin*Nfft/fs)]))),
             xytext = (fin/fs*10, 10*np.log10(abs(doutp[int(fin*Nfft/fs)]))-10),
             arrowprops = dict(facecolor = 'black', width = 0.2, headwidth = 8),
             horizontalalignment = 'center')

    # Output spectrum for filtered (sinc filter) output

    fin = f         # input signal frequency stays the same
    fs = Fs/N       # fs = 2B = nyquist frequency
    osr = OSR/N     # osr has to be divided to maintain the ratio with number of points in dout

    text_SNDR=f'SNDR={SNDR_filtered:.2f}'+f' dB\n Ps= {10*np.log10(abs(Ps_filtered))+5.88:.2f} dB'
    print('Ps=',Ps_filtered,'PN=',Pn_filtered,text_SNDR)
    Nfft = len(doutf)
    freq = np.linspace(0, 1, Nfft)
    plt.figure()
    plt.semilogx(freq[0:int(Nfft/2)],10*np.log10(abs(doutfp[0:int(Nfft/2)])))
    plt.title(f'Sinc filter - Output spectrum SNDR={SNDR_filtered:.2f} dB')
    plt.xlabel('Frequency [f/Fs]')
    plt.ylabel('Amplitude [dBr]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(1/(2*osr), color='green') # Bandwidth
    plt.annotate(text_SNDR,
             xy = (fin/fs, 10*np.log10(abs(doutfp[int(fin*Nfft/fs)]))),
             xytext = (fin/fs*10, 10*np.log10(abs(doutfp[int(fin*Nfft/fs)]))-10),
             arrowprops = dict(facecolor = 'black', width = 0.2, headwidth = 8),
             horizontalalignment = 'center')
    
    # Input Signal, Output bit stream and Sinc Filter output
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(vin,label='Input signal')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(dout,label='Output bit stream')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(doutf,label = '3rd order sinc filter')
    plt.title('Output signal filtered using 3rd order sinc filter')
    plt.legend()

# Show all plots
plt.show()


