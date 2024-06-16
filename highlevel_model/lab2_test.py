# High Level Model for 2nd order sigma delta modulator

import time
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy import signal

###################### Functions  ######################

@nb.jit(nopython=True)
def second_order_sigma_delta_modulator(k1,k2,b1,b2,B1,B2,Vr,vin):

    nppp = len(vin)          

    e1 = np.zeros(nppp)              # First integrator error
    e2 = np.zeros(nppp)              # Second integrator error
    x1 = np.zeros(nppp)              # First integrator output
    x2 = np.zeros(nppp)              # Second integrator output
    do = np.zeros(nppp)              # Output of the modulator

    for n in range(1,nppp):
        # Quantizer
        if (x2[n-1] >= 0):
            do[n] = 1
        else:
            do[n] = -1

        # Modulator - First integrator
        e1[n] = vin[n] - do[n] * (Vr*b1)
        x1[n] = k1 * e1[n] + B1 * x1[n-1]
        
        # Modulator - Second integrator
        e2[n] = x1[n-1] - do[n] * (Vr*b2)
        x2[n] = k2 * e2[n] + B2 * x2[n-1]

    return do

# Function to implement the third-order sinc filter
def third_order_sinc_filter(dout, M):
    nppp = len(dout)

    i1 = np.zeros(nppp)              # First integrator output
    i2 = np.zeros(nppp)              # Second integrator output
    i3 = np.zeros(nppp)              # Third integrator output
    d1 = np.zeros(nppp//M + 1)       # First differentiator output
    d2 = np.zeros(nppp//M + 1)       # Second differentiator output
    d3 = np.zeros(nppp//M + 1)       # Third differentiator output

    # Integration
    i1 = np.cumsum(dout)
    i2 = np.cumsum(i1)
    i3 = np.cumsum(i2)

    # Decimation
    dec = i3[::M]

    # Differentiation
    d1 = np.diff(np.concatenate(([0], dec)))
    d2 = np.diff(np.concatenate(([0], d1)))
    d3 = np.diff(np.concatenate(([0], d2)))

    # Gain correction
    df = d3 / (M ** 3)

    return df

###################### Parameters ######################

# Simulation setup
SINAD = False
REAL = False
npp = 40000

# Modulator general parameters
Vr = 2.24
B = 20e3            # Signal bandwidth
OSR = 100           # Oversampling ratio
Fs = 2*B*OSR        # Sampling frequency

# Modulator Coefficients
k1 = 1
k2 = 1
b1 = 1
b2 = 2
B1 = 1
B2 = 1
if REAL:
    A = 100
    B1 = (A + 1) / (A + 2 + b1)
    B2 = (A + 1) / (A + 2 + b2)


# Decimation filter parameters
M = int(OSR)    # Decimation factor

# Signal parameters
f = 1e3
f = round(f/Fs * npp) * Fs/npp
t = np.arange(npp) / Fs


########################################################
###################### Simulation ######################
########################################################

######################  Signals  ######################

# values to consider for the input signal
# Ain_values = [np.sqrt(2), Vr/2, np.sqrt(2)/10, np.sqrt(2)/100]  # [1Vrms, Vr/2, -20dBV, -40dBV]
# Vth_values = [1e-6, 1e-5, 1e-4, 1e-3]                           # [1uV, 10uV, 100uV, 1000uV]

if SINAD:
    Ain_values = np.logspace(np.log10(1e-4), Vr*np.log10(np.sqrt(2)), 1000)
    Vth_values = [1e-6, 1e-5, 1e-4, 1e-3]
    #Vth_values = [1e-5]

else:
    Ain_values = [Vr/2]
    Vth_values = [1e-5] 


###################### Variables ######################

dout = np.zeros(npp)            # modulator output
doutf = np.zeros(npp//M + 1)    # sinc filter output

SNDRs_modulator = {Vth: [] for Vth in Vth_values}
SNDRs_filter = {Vth: [] for Vth in Vth_values}

###################### Main loop ######################
start_time = time.time()

for Vth in Vth_values:
    # print in uV
    print(f'Vth = {Vth*1e6:.2f} uV')
    for Ain in Ain_values:
        vin = Ain * np.sin(2 * np.pi * f * t) + np.random.randn(npp) * Vth      # Generate input signal
        dout = second_order_sigma_delta_modulator(k1,k2,b1,b2,B1,B2,Vr,vin)     # Modulator
        doutf = third_order_sinc_filter(dout,M)                                 # Sinc filter

        # FFT of the modulator output signal
        Nfft_dout = len(dout)
        window = signal.windows.blackmanharris(Nfft_dout)       # blackmanharris window
        doutw = np.fft.fft(np.multiply(dout, window))
        doutp = np.abs(np.multiply(doutw,doutw.conjugate())/((Nfft_dout)**2)*2)
        index = int((Nfft_dout * f) / Fs)
        Ps_dout = np.abs(np.sum(doutp[index - 4: index + 4]))
        Pn_dout = np.abs(np.sum(doutp[:round(Nfft_dout / (2 * OSR))])) - Ps_dout
        SNDR_dout = 10 * np.log10(Ps_dout / Pn_dout)

        # FFT of the filter output signal
        Nfft_doutf = len(doutf)
        window = signal.windows.blackmanharris(Nfft_doutf)      # blackmanharris window
        doutfw = np.fft.fft(np.multiply(doutf, window))
        doutfp = np.abs(np.multiply(doutfw,doutfw.conjugate())/((Nfft_doutf)**2)*2)
        index = int((Nfft_doutf * f) / Fs/M)
        Ps_doutf = np.abs(np.sum(doutfp[index - 4: index + 4]))
        Pn_doutf = np.abs(np.sum(doutfp[:round(Nfft_doutf / (2 * OSR/M))])) - Ps_doutf
        SNDR_doutf = 10 * np.log10(Ps_doutf / Pn_doutf)

        # Store the SNDR results for SINAD analysis
        SNDRs_modulator[Vth].append(SNDR_dout)
        SNDRs_filter[Vth].append(SNDR_doutf)
        
stop_time = time.time()

print(f'Elapsed time: {stop_time - start_time:.2f} s')

######################  Results  ######################

def output_spectrum(doutfp,fin,Fs,OSR,title,real):

    # Prepend the title based on the real variable
    model_type = "Real model" if real else "Ideal model"
    full_title = f'{model_type} - {title}'

    plt.figure()
    #Calculate the SNR using blackman harris offset=4
    Nfft=len(doutfp)
    Ps=np.abs(np.sum(doutfp[int(Nfft*fin/Fs)-4:int(Nfft*fin/Fs)+4]))
    PN=np.abs(np.sum(doutfp[5:int(Nfft/(2*OSR))]))
    SNDR=10*np.log10(Ps/(PN-Ps))
    #atenuação blackman harris=5.88dB
    text_SNDR=f'SNDR={SNDR:.2f}'+f' dB\n Ps= {10*np.log10(abs(Ps))+5.88:.2f} dB'
    print('Ps=',Ps,'PN=',PN-Ps,text_SNDR)
    freq = np.linspace(0, 1, Nfft)
    plt.semilogx(freq[0:int(Nfft/2)],10*np.log10(abs(doutfp[0:int(Nfft/2)])))
    plt.title(f'{full_title} - Output spectrum SNDR={SNDR:.2f} dB')
    plt.xlabel('Frequency [f/Fs]')
    plt.ylabel('Amplitude [dBr]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(1/(2*OSR), color='green') # Bandwidth
    plt.annotate(text_SNDR,
             xy = (fin/Fs, 10*np.log10(abs(doutfp[int(fin*Nfft/Fs)]))),
             xytext = (fin/Fs*10, 10*np.log10(abs(doutfp[int(fin*Nfft/Fs)]))-10),
             arrowprops = dict(facecolor = 'black', width = 0.2, headwidth = 8),
             horizontalalignment = 'center')

if SINAD:           
    # SINAD results for modulator output
    plt.figure()
    for Vth in Vth_values:
        plt.plot(20*np.log10(Ain_values / np.sqrt(2)), SNDRs_modulator[Vth], label=f'Vnin={Vth*1e6:.0f}uV')

    if len(Vth_values) == 1:
        # Find the max SNDR and corresponding amplitude
        max_sndr_modulator = np.max(SNDRs_modulator[Vth])
        max_sndr_index_modulator = np.argmax(SNDRs_modulator[Vth])

        # Find the index of the max SNDR value - 3 dB
        target_sndr = max_sndr_modulator - 3
        max_minus_3db_index = np.argmin(np.abs(SNDRs_modulator[Vth][:max_sndr_index_modulator] - target_sndr))
        max_minus_3db_amplitude_modulator = Ain_values[max_minus_3db_index]

        # Find the amplitude corresponding to SNDR = 0 dB, only considering values to the left of the max SNDR
        snr_threshold = 0  # SNDR = 0 dB
        snr_diff_left = np.abs(np.array(SNDRs_modulator[Vth][:max_minus_3db_index]) - snr_threshold)
        zero_sndr_index_modulator = np.argmin(snr_diff_left)
        zero_sndr_amplitude_modulator = Ain_values[zero_sndr_index_modulator]
    
        # Find the amplitude corresponding to the max SNDR-3dB and zero SNDR in dBV
        max_minus_3db_amplitude_dBV = 20 * np.log10(max_minus_3db_amplitude_modulator / np.sqrt(2))
        zero_sndr_amplitude_dBV = 20 * np.log10(zero_sndr_amplitude_modulator / np.sqrt(2))

        # Calculate dynamic range
        dynamic_range = max_minus_3db_amplitude_dBV - zero_sndr_amplitude_dBV

        plt.axvline(x=max_minus_3db_amplitude_dBV, color='k', linestyle='--', label=f'Max SNDR-3dB at {max_minus_3db_amplitude_dBV:.2f} dBV')
        plt.axhline(y=target_sndr, color='k', linestyle='--')
        plt.axvline(x=zero_sndr_amplitude_dBV, color='k', linestyle='--', label=f'SNDR=0dB at {zero_sndr_amplitude_dBV:.2f} dBV')

        # Add bidirectional arrow
        plt.annotate('', xy=(zero_sndr_amplitude_dBV, max_sndr_modulator), xytext=(max_minus_3db_amplitude_dBV, max_sndr_modulator),
                        arrowprops=dict(arrowstyle='<->', color='red'))
        plt.text((zero_sndr_amplitude_dBV + max_minus_3db_amplitude_dBV) / 2, max_sndr_modulator + +1,
                    f'Dynamic Range: {dynamic_range:.2f} dBV', ha='center', color='red')
        
        plt.title(f'Modulator output - SINAD for OSR={OSR}\nMax SNDR: {max_sndr_modulator:.2f} dB with Ain: {max_minus_3db_amplitude_modulator:.2f} V')
    else:
        plt.title(f'Modulator output - SINAD for OSR={OSR}')

    plt.xlabel('Ain [dBV]')
    plt.ylabel('SNDR [dB]')
    plt.grid(True)
    plt.legend()

else:

    plt.figure(figsize=(10, 8))
    # Input signal subplot
    plt.subplot(3, 1, 1)
    plt.plot(vin, label='Input Signal', color='green')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Input Signal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Output bit stream subplot
    plt.subplot(3, 1, 2)
    plt.plot(dout, label='Output Bit Stream', color='blue', marker='o', linestyle='-')
    plt.xlabel('Sample Index')
    plt.ylabel('Bit Value')
    plt.title('Output Bit Stream')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Sinc filter output subplot
    plt.subplot(3, 1, 3)
    plt.plot(doutf, label='3rd Order Sinc Filter', color='orange')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Output Signal Filtered Using 3rd Order Sinc Filter')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    ################## Output spectrum #################
    output_spectrum(doutp,f,Fs,OSR, "Modulator", REAL)
    output_spectrum(doutfp,f,Fs/M,OSR/M, "Sinc filter", REAL)


# finally, show all plots
plt.show()