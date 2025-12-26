import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import sys

# --- CONFIGURATION ---
N_BITS = 8          # Number of ADC bits
V_REF = 5.0         # Reference voltage [V]

# Create output directory
os.makedirs('plots_dynamic', exist_ok=True)

# --- MEASUREMENT CONFIGURATIONS ---
# Format: (f_sampling, f_input, filename)
MEASUREMENTS = [
    # Częstotliwość próbkowania 1024 Hz
    (1024, 409, 'dane/ADC_dynamic_1024_409.txt'),
    (1024, 241, 'dane/ADC_dynamic_1024_241.txt'),
    (1024, 13, 'dane/ADC_dynamic_1024_13.txt'),

    # Częstotliwość próbkowania 128 Hz
    (128, 51.125, 'dane/ADC_dynamic_128_51.txt'),
    (128, 30.125, 'dane/ADC_dynamic_128_30125.txt'),
    (128, 30.12, 'dane/ADC_dynamic_128_3012.txt'),
    (128, 3.875, 'dane/ADC_dynamic_128_3875.txt'),
]


# --- 1. LOAD DATA ---
def load_dynamic_data(filename):
    """
    Load dynamic measurement data.
    
    Parameters:
    -----------
    filename : str
        Path to data file
        
    Returns:
    --------
    data : array
        Digital codes from ADC
    """
    try:
        data = np.loadtxt(filename)
        print(f"✓ Loaded: {filename} ({len(data)} samples)")
        return data
    except FileNotFoundError:
        print(f"✗ File not found: {filename}")
        return None
    except Exception as e:
        print(f"✗ Error loading {filename}: {e}")
        return None


# --- 2. CALCULATE FFT AND SPECTRUM ---
def calculate_spectrum(digital_codes, f_sampling, N_bits):
    """
    Calculate FFT spectrum from digital codes.
    
    Parameters:
    -----------
    digital_codes : array
        Measured digital codes
    f_sampling : float
        Sampling frequency [Hz]
    N_bits : int
        Number of bits
        
    Returns:
    --------
    freqs : array
        Frequency bins [Hz]
    spectrum_dB : array
        Power spectrum [dB]
    spectrum_linear : array
        Linear power spectrum
    """
    N = len(digital_codes)
    
    # Remove DC component
    digital_codes_ac = digital_codes - np.mean(digital_codes)
    
    # Apply window to reduce spectral leakage
    window = signal.windows.blackmanharris(N)
    windowed_signal = digital_codes_ac * window
    
    # Calculate FFT
    fft_result = fft(windowed_signal)
    
    # One-sided spectrum (positive frequencies only)
    N_half = N // 2
    spectrum_linear = np.abs(fft_result[:N_half])
    
    # Normalize and account for window power loss
    window_power = np.sum(window**2) / N
    spectrum_linear = spectrum_linear / (N * np.sqrt(window_power))
    
    # Convert to dB (relative to full scale)
    # Full scale for N-bit ADC is 2^(N-1)
    full_scale = 2**(N_bits - 1)
    spectrum_dB = 20 * np.log10(spectrum_linear / full_scale + 1e-10)
    
    # Frequency bins
    freqs = fftfreq(N, 1/f_sampling)[:N_half]
    
    return freqs, spectrum_dB, spectrum_linear


# --- 3. FIND SIGNAL AND HARMONIC PEAKS ---
def find_signal_peak(freqs, spectrum_dB, f_input, tolerance=5):
    """
    Find the fundamental signal peak in the spectrum.
    
    Parameters:
    -----------
    freqs : array
        Frequency bins
    spectrum_dB : array
        Spectrum in dB
    f_input : float
        Expected input frequency
    tolerance : float
        Search tolerance [Hz]
        
    Returns:
    --------
    signal_idx : int
        Index of signal peak
    signal_freq : float
        Actual signal frequency
    signal_power_dB : float
        Signal power [dB]
    """
    # Search around expected frequency
    mask = (freqs >= f_input - tolerance) & (freqs <= f_input + tolerance)
    search_region = np.where(mask)[0]
    
    if len(search_region) == 0:
        # Fallback: find maximum in spectrum
        signal_idx = np.argmax(spectrum_dB)
    else:
        signal_idx = search_region[np.argmax(spectrum_dB[search_region])]
    
    signal_freq = freqs[signal_idx]
    signal_power_dB = spectrum_dB[signal_idx]
    
    return signal_idx, signal_freq, signal_power_dB


def find_harmonics(freqs, spectrum_dB, f_signal, n_harmonics=10, tolerance=5):
    """
    Find harmonic peaks in the spectrum.
    
    Parameters:
    -----------
    freqs : array
        Frequency bins
    spectrum_dB : array
        Spectrum in dB
    f_signal : float
        Fundamental frequency
    n_harmonics : int
        Number of harmonics to search
    tolerance : float
        Search tolerance [Hz]
        
    Returns:
    --------
    harmonic_powers : list
        Power of each harmonic [dB]
    harmonic_freqs : list
        Frequency of each harmonic [Hz]
    """
    harmonic_powers = []
    harmonic_freqs = []
    
    f_max = freqs[-1]
    
    for k in range(2, n_harmonics + 2):  # 2nd to (n_harmonics+1)th harmonic
        f_harmonic = k * f_signal
        
        if f_harmonic > f_max:
            break
        
        # Search around expected harmonic frequency
        mask = (freqs >= f_harmonic - tolerance) & (freqs <= f_harmonic + tolerance)
        search_region = np.where(mask)[0]
        
        if len(search_region) > 0:
            harmonic_idx = search_region[np.argmax(spectrum_dB[search_region])]
            harmonic_powers.append(spectrum_dB[harmonic_idx])
            harmonic_freqs.append(freqs[harmonic_idx])
    
    return harmonic_powers, harmonic_freqs


# --- 4. CALCULATE DYNAMIC METRICS ---
def calculate_dynamic_metrics(freqs, spectrum_dB, spectrum_linear, 
                              signal_idx, f_signal, f_sampling):
    """
    Calculate THD, SNHR, SFDR, SINAD, and ENOB.
    
    Parameters:
    -----------
    freqs : array
        Frequency bins
    spectrum_dB : array
        Spectrum in dB
    spectrum_linear : array
        Linear spectrum
    signal_idx : int
        Index of signal peak
    f_signal : float
        Signal frequency
    f_sampling : float
        Sampling frequency
        
    Returns:
    --------
    metrics : dict
        Dictionary with all dynamic metrics
    """
    # Signal power (linear)
    P_signal = spectrum_linear[signal_idx]**2
    
    # Find harmonics
    harmonic_powers_dB, harmonic_freqs = find_harmonics(
        freqs, spectrum_dB, f_signal, n_harmonics=10
    )
    
    # Convert harmonic powers to linear
    harmonic_powers_linear = [10**(p_dB/20) for p_dB in harmonic_powers_dB]
    
    # THD - Total Harmonic Distortion
    if len(harmonic_powers_linear) > 0:
        P_harmonics = np.sum(np.array(harmonic_powers_linear)**2)
        THD = 10 * np.log10(P_harmonics / P_signal)
        THD_percent = 100 * np.sqrt(P_harmonics / P_signal)
    else:
        P_harmonics = 0
        THD = -np.inf
        THD_percent = 0
    
    # Create mask to exclude signal and harmonics (±3 bins around each)
    exclude_mask = np.ones(len(spectrum_linear), dtype=bool)
    exclude_mask[max(0, signal_idx-3):min(len(exclude_mask), signal_idx+4)] = False
    
    for f_harm in harmonic_freqs:
        harm_idx = np.argmin(np.abs(freqs - f_harm))
        exclude_mask[max(0, harm_idx-3):min(len(exclude_mask), harm_idx+4)] = False
    
    # Exclude DC (first few bins)
    exclude_mask[:5] = False
    
    # Noise + non-harmonic distortion power
    P_noise = np.sum(spectrum_linear[exclude_mask]**2)
    
    # SNHR - Signal to Non-Harmonic Ratio
    if P_noise > 0:
        SNHR = 10 * np.log10(P_signal / P_noise)
    else:
        SNHR = np.inf
    
    # SFDR - Spurious Free Dynamic Range
    # Maximum spurious signal (excluding fundamental)
    spectrum_dB_no_signal = spectrum_dB.copy()
    spectrum_dB_no_signal[signal_idx] = -np.inf
    max_spurious_dB = np.max(spectrum_dB_no_signal)
    signal_dB = spectrum_dB[signal_idx]
    SFDR = signal_dB - max_spurious_dB
    
    # SINAD - Signal to Noise And Distortion
    P_total_distortion = P_harmonics + P_noise
    if P_total_distortion > 0:
        SINAD = 10 * np.log10(P_signal / P_total_distortion)
    else:
        SINAD = np.inf
    
    # ENOB - Effective Number Of Bits
    # ENOB = (SINAD - 1.76) / 6.02
    if np.isfinite(SINAD):
        ENOB = (SINAD - 1.76) / 6.02
    else:
        ENOB = N_BITS
    
    metrics = {
        'THD': THD,
        'THD_percent': THD_percent,
        'SNHR': SNHR,
        'SFDR': SFDR,
        'SINAD': SINAD,
        'ENOB': ENOB,
        'signal_power_dB': signal_dB,
        'harmonic_powers_dB': harmonic_powers_dB,
        'harmonic_freqs': harmonic_freqs,
        'n_harmonics': len(harmonic_powers_dB)
    }
    
    return metrics


# --- 5. VISUALIZATION ---
def plot_spectrum(freqs, spectrum_dB, f_signal, f_sampling, metrics, 
                 measurement_name):
    """
    Create comprehensive spectrum plot with annotations.
    
    Parameters:
    -----------
    freqs : array
        Frequency bins
    spectrum_dB : array
        Spectrum in dB
    f_signal : float
        Signal frequency
    f_sampling : float
        Sampling frequency
    metrics : dict
        Dynamic metrics
    measurement_name : str
        Name for saving plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # --- PLOT 1: Full spectrum ---
    ax1.plot(freqs, spectrum_dB, 'b-', linewidth=1, alpha=0.7)
    
    # Mark signal peak
    signal_idx = np.argmin(np.abs(freqs - f_signal))
    ax1.plot(freqs[signal_idx], spectrum_dB[signal_idx], 'ro', 
             markersize=10, label=f'Sygnał ({f_signal:.2f} Hz)')
    
    # Mark harmonics
    for i, (f_harm, p_harm) in enumerate(zip(metrics['harmonic_freqs'], 
                                              metrics['harmonic_powers_dB'])):
        harm_idx = np.argmin(np.abs(freqs - f_harm))
        ax1.plot(freqs[harm_idx], spectrum_dB[harm_idx], 'mx', 
                markersize=8, markeredgewidth=2)
        if i == 0:
            ax1.plot([], [], 'mx', markersize=8, markeredgewidth=2, 
                    label='Harmoniczne')
    
    ax1.set_title(f'Widmo sygnału - fs={f_sampling} Hz, fin={f_signal:.3f} Hz', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Częstotliwość [Hz]', fontsize=12)
    ax1.set_ylabel('Amplituda [dB]', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, f_sampling/2])
    
    # Add text box with metrics
    textstr = f"""Metryki dynamiczne:
THD = {metrics['THD']:.2f} dB ({metrics['THD_percent']:.3f}%)
SNHR = {metrics['SNHR']:.2f} dB
SFDR = {metrics['SFDR']:.2f} dB
SINAD = {metrics['SINAD']:.2f} dB
ENOB = {metrics['ENOB']:.2f} bitów"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.98, 0.97, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')
    
    # --- PLOT 2: Zoomed around signal ---
    zoom_range = min(50, f_sampling / 10)  # Hz
    zoom_mask = (freqs >= f_signal - zoom_range) & (freqs <= f_signal + zoom_range)
    
    ax2.plot(freqs[zoom_mask], spectrum_dB[zoom_mask], 'b-', linewidth=2)
    ax2.plot(freqs[signal_idx], spectrum_dB[signal_idx], 'ro', 
             markersize=10, label='Sygnał')
    
    ax2.set_title(f'Przybliżenie wokół sygnału głównego', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('Częstotliwość [Hz]', fontsize=11)
    ax2.set_ylabel('Amplituda [dB]', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'plots_dynamic/{measurement_name}_spectrum.png', dpi=150)
    plt.close()


# --- 6. MAIN ANALYSIS FUNCTION ---
def analyze_dynamic_measurement(filename, f_sampling, f_input, N_bits, V_REF):
    """
    Perform complete dynamic analysis of ADC measurement.
    
    Parameters:
    -----------
    filename : str
        Path to data file
    f_sampling : float
        Sampling frequency [Hz]
    f_input : float
        Input signal frequency [Hz]
    N_bits : int
        Number of bits
    V_REF : float
        Reference voltage
        
    Returns:
    --------
    metrics : dict or None
        Dynamic metrics if successful
    """
    print(f"\n{'─'*70}")
    print(f"Analiza: fs = {f_sampling} Hz, fin = {f_input} Hz")
    print(f"{'─'*70}")
    
    # Load data
    digital_codes = load_dynamic_data(filename)
    if digital_codes is None:
        print("⚠ Pominięto analizę (brak danych)\n")
        return None
    
    # Calculate spectrum
    freqs, spectrum_dB, spectrum_linear = calculate_spectrum(
        digital_codes, f_sampling, N_bits
    )
    
    # Find signal peak
    signal_idx, f_signal_actual, signal_power = find_signal_peak(
        freqs, spectrum_dB, f_input
    )
    
    print(f"Sygnał wykryty: f = {f_signal_actual:.3f} Hz, P = {signal_power:.2f} dB")
    
    # Calculate metrics
    metrics = calculate_dynamic_metrics(
        freqs, spectrum_dB, spectrum_linear, 
        signal_idx, f_signal_actual, f_sampling
    )
    
    # Print results
    print(f"\nWYNIKI:")
    print(f"  THD   = {metrics['THD']:>8.2f} dB  ({metrics['THD_percent']:>6.3f} %)")
    print(f"  SNHR  = {metrics['SNHR']:>8.2f} dB")
    print(f"  SFDR  = {metrics['SFDR']:>8.2f} dB")
    print(f"  SINAD = {metrics['SINAD']:>8.2f} dB")
    print(f"  ENOB  = {metrics['ENOB']:>8.2f} bitów")
    print(f"  Wykryte harmoniczne: {metrics['n_harmonics']}")
    
    # Create plot
    measurement_name = f"fs{int(f_sampling)}_fin{f_input}".replace('.', 'p')
    plot_spectrum(freqs, spectrum_dB, f_signal_actual, f_sampling, 
                 metrics, measurement_name)
    
    print(f"✓ Wykres zapisany: plots_dynamic/{measurement_name}_spectrum.png")
    
    return metrics


# --- 7. COMPARISON TABLE ---
def create_summary_table(results):
    """
    Create summary comparison table for all measurements.
    
    Parameters:
    -----------
    results : list of tuples
        List of (f_sampling, f_input, metrics) tuples
    """
    if not results:
        print("Brak wyników do podsumowania")
        return
    
    print(f"\n{'='*70}")
    print("TABELA PODSUMOWUJĄCA - WSZYSTKIE POMIARY")
    print(f"{'='*70}\n")
    
    # Table header
    header = f"{'fs [Hz]':<10} {'fin [Hz]':<12} {'THD [dB]':<10} {'SNHR [dB]':<11} " \
             f"{'SFDR [dB]':<11} {'SINAD [dB]':<12} {'ENOB':<8}"
    print(header)
    print("─" * 70)
    
    # Table rows
    for f_samp, f_in, metrics in results:
        if metrics is not None:
            row = f"{f_samp:<10} {f_in:<12.3f} {metrics['THD']:<10.2f} " \
                  f"{metrics['SNHR']:<11.2f} {metrics['SFDR']:<11.2f} " \
                  f"{metrics['SINAD']:<12.2f} {metrics['ENOB']:<8.2f}"
            print(row)
    
    print("─" * 70)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Porównanie metryk dynamicznych dla wszystkich pomiarów', 
                 fontsize=14, fontweight='bold')
    
    labels = [f"fs={fs}\nfin={fin:.1f}" for fs, fin, _ in results if _ is not None]
    
    metrics_to_plot = [
        ('THD', 'THD [dB]', 0, 0),
        ('SNHR', 'SNHR [dB]', 0, 1),
        ('SFDR', 'SFDR [dB]', 0, 2),
        ('SINAD', 'SINAD [dB]', 1, 0),
        ('ENOB', 'ENOB [bity]', 1, 1),
        ('THD_percent', 'THD [%]', 1, 2),
    ]
    
    for metric_key, metric_label, row, col in metrics_to_plot:
        ax = axes[row, col]
        values = [m[metric_key] for _, _, m in results if m is not None]
        
        bars = ax.bar(range(len(values)), values, color='skyblue', 
                     edgecolor='navy', alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('plots_dynamic/comparison_all_measurements.png', dpi=150)
    plt.close()
    
    print(f"\n✓ Wykres porównawczy zapisany: plots_dynamic/comparison_all_measurements.png")


# --- 8. MAIN EXECUTION ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print("PARAMETRYZACJA DYNAMICZNA PRZETWORNIKA ADC")
    print("Laboratorium Elektroniczne WFiIS C-7")
    print("="*70)
    print(f"\nKonfiguracja: {N_BITS}-bitowy ADC, V_REF = {V_REF} V")
    
    results = []
    
    # Analyze all measurements
    for f_sampling, f_input, filename in MEASUREMENTS:
        metrics = analyze_dynamic_measurement(
            filename, f_sampling, f_input, N_BITS, V_REF
        )
        results.append((f_sampling, f_input, metrics))
    
    # Create summary
    create_summary_table(results)
    
    print("\n" + "="*70)
    print("ANALIZA DYNAMICZNA ZAKOŃCZONA")
    print("="*70)
    print(f"\nWszystkie wykresy zapisane w katalogu: plots_dynamic/")
    print(f"Liczba przeanalizowanych pomiarów: {len([r for r in results if r[2] is not None])}/{len(MEASUREMENTS)}")