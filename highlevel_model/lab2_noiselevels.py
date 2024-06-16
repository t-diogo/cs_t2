# High Level Model for 2nd order sigma delta modulator

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
Vr = 2.24           # Reference voltage

OSR = 100           # Oversampling ratio
B = 20e3            # Signal bandwidth
Fs = 2*B*OSR        # Sampling frequency


# Simulation parameters
np_points = 50000       # Number of points

real = False             # test with integrators with finite gain (True) or infinite gain (False) - g)

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
    B1 = (A + 1) / (A + 2 + b1)
    B2 = (A + 1) / (A + 2 + b2)

# Input signal - sine wave 1kHz
f = 1e3
f = round((f/Fs) * np_points ) * (Fs/np_points)
print(f)

# Time vector
t = np.arange(np_points) / Fs

# Input Signal parameters

Ain = Vr/2
noise_levels = [1e-6, 1e-5, 1e-4, 1e-3]  # Different noise levels


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

for noise in noise_levels:
    print(f'Noise level: {noise*1e6:.0f}uV')

    vin = Ain * np.sin(2 * np.pi * f * t) + np.random.randn(np_points) * noise

    # Modulator
    dout = second_order_sigma_delta_modulator(vin)

    # Obtain the spectrum for the noise value
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

    # Output spectrum for non-filtered output
    
    text_SNDR=f'SNDR={SNDR_nofilter:.2f}'+f' dB\n Ps= {10*np.log10(abs(Ps_nofilter))+5.88:.2f} dB'
    print('Ps=',Ps_nofilter,'PN=',Pn_nofilter,text_SNDR)
    Nfft = len(dout)
    freq = np.linspace(0, 1, Nfft)
    plt.figure()
    plt.semilogx(freq[0:int(Nfft/2)],10*np.log10(abs(doutp[0:int(Nfft/2)])))
    plt.title(f'No filter - Output spectrum for Noise level: {noise*1e6:.0f}uV\nSNDR={SNDR_nofilter:.2f} dB')
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
    
plt.show()
