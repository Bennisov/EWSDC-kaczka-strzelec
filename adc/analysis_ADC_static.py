import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURATION ---
N_BITS = 8          # Number of ADC bits
V_REF = 5.0         # Reference voltage [V]

# Create output directory
os.makedirs('plots', exist_ok=True)

# --- 1. LOAD DATA ---
data_file = 'dane/adc_static.txt'

try:
    digital_codes = np.loadtxt(data_file)
    print(f"Data file loaded successfully: {len(digital_codes)} samples")
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    print("Make sure the file is in the 'dane/' directory.")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error loading data: {e}")
    sys.exit(1)


# --- 2. CALCULATE AVERAGED CODE PER VOLTAGE ---
def calculate_static_code_avg(digital_codes, N_codes):
    """
    Calculate averaged digital codes assuming uniform voltage sweep.
    
    Parameters:
    -----------
    digital_codes : array
        Measured digital output codes
    N_codes : int
        Total number of codes (2^N_bits)
        
    Returns:
    --------
    V_in_avg : array
        Input voltages (inferred from sweep)
    code_avg : array
        Average digital code for each voltage
    """
    N_points = len(digital_codes)
    N_samples_per_voltage = N_points // N_codes
    
    if N_samples_per_voltage == 0:
        raise ValueError(f"Not enough data points. Need at least {N_codes} points.")
    
    print(f"Total points: {N_points}, Samples per voltage: {N_samples_per_voltage}")
    
    # Truncate to make even division
    data_length = N_codes * N_samples_per_voltage
    digital_codes_truncated = digital_codes[:data_length]
    
    remaining = N_points - data_length
    if remaining > 0:
        print(f"Warning: {remaining} data points discarded due to rounding")
    
    # Reshape and average
    codes_reshaped = digital_codes_truncated.reshape(N_codes, N_samples_per_voltage)
    code_avg = np.mean(codes_reshaped, axis=1)
    
    # Generate input voltage sweep (assumes linear sweep from 0 to V_REF)
    V_in_avg = np.linspace(0, V_REF, N_codes)
    
    return V_in_avg, code_avg


# --- 3. CALCULATE STATIC ERRORS ---
def calculate_adc_errors(V_in, code_meas, V_REF, N_bits):
    """
    Calculate resolution, quantization error, offset error, and gain error.
    
    Parameters:
    -----------
    V_in : array
        Input voltages
    code_meas : array
        Measured digital codes
    V_REF : float
        Reference voltage
    N_bits : int
        Number of bits
        
    Returns:
    --------
    resolution_bits : int
        Resolution in bits
    LSB : float
        Ideal LSB voltage [V]
    Q_error : float
        Quantization error [V]
    V_os : float
        Offset voltage [V]
    Delta_G : float
        Gain error [%]
    V_os_LSB : float
        Offset in LSB units
    """
    # Resolution
    resolution_bits = N_bits
    
    # Ideal LSB
    N_levels = 2**N_bits
    LSB = V_REF / N_levels
    
    # Quantization error (błąd dyskretyzacji/kwantyzacji)
    Q_error = LSB / 2  # ±LSB/2
    
    # Find transition voltages for offset calculation
    # First transition should occur at LSB/2 for ideal ADC
    transition_idx_first = np.where(code_meas >= 0.5)[0]
    if len(transition_idx_first) > 0:
        V_first_transition = V_in[transition_idx_first[0]]
    else:
        V_first_transition = 0
    
    # Offset Error (błąd przesunięcia zera)
    V_ideal_first_transition = LSB / 2
    V_os = V_first_transition - V_ideal_first_transition
    V_os_LSB = V_os / LSB
    
    # Gain Error (błąd skalowania/wzmocnienia)
    # Find last transition (to maximum code)
    D_max = N_levels - 1
    transition_idx_last = np.where(code_meas >= (D_max - 0.5))[0]
    if len(transition_idx_last) > 0:
        V_last_transition = V_in[transition_idx_last[0]]
    else:
        V_last_transition = V_REF
    
    # Measured span (corrected for offset)
    V_span_measured = V_last_transition - V_first_transition
    
    # Ideal span (from first to last transition)
    V_span_ideal = V_REF - LSB  # (D_max + 0.5)*LSB - 0.5*LSB
    
    Delta_G = 100 * ((V_span_measured / V_span_ideal) - 1)
    
    return resolution_bits, LSB, Q_error, V_os, Delta_G, V_os_LSB


# --- 4. CALCULATE NONLINEARITY (INL, DNL) ---
def calculate_adc_nonlinearity(V_in, code_meas, LSB, V_os, N_bits):
    """
    Calculate Integral (INL) and Differential (DNL) Nonlinearity.
    
    Parameters:
    -----------
    V_in : array
        Input voltages
    code_meas : array
        Measured digital codes
    LSB : float
        Ideal LSB voltage
    V_os : float
        Offset voltage
    N_bits : int
        Number of bits
        
    Returns:
    --------
    INL_max : float
        Maximum INL [LSB]
    DNL_max : float
        Maximum DNL [LSB]
    code_ideal_corrected : array
        Ideal corrected codes
    non_monotonic_detected : bool
        True if ADC is non-monotonic
    INL_all : array
        INL for all voltages
    DNL_all : array
        DNL for all code transitions
    transition_voltages : array
        Voltages where code transitions occur
    code_widths_LSB : array
        Width of each code bin in LSB
    missing_codes : list
        List of missing digital codes
    """
    N_codes = 2**N_bits
    
    # Calculate best-fit line (offset and gain corrected)
    # Use robust fitting on middle portion to avoid endpoint effects
    mid_start = len(V_in) // 10
    mid_end = 9 * len(V_in) // 10
    coeffs = np.polyfit(V_in[mid_start:mid_end], code_meas[mid_start:mid_end], 1)
    slope_cal = coeffs[0]
    intercept_cal = coeffs[1]
    
    # Ideal characteristic with offset and gain correction
    code_ideal_corrected = intercept_cal + slope_cal * V_in
    
    # Integral Nonlinearity (INL)
    # INL = (measured code - ideal corrected code)
    INL_all = code_meas - code_ideal_corrected  # Already in LSB units (1 code = 1 LSB)
    INL_max = np.max(np.abs(INL_all))
    
    # Differential Nonlinearity (DNL)
    # Find transition voltages for each code
    transition_voltages = np.full(N_codes, np.nan)
    code_widths_volt = np.full(N_codes, np.nan)
    
    for code in range(N_codes):
        # Find first voltage where this code appears
        idx = np.where(code_meas >= (code + 0.5))[0]
        if len(idx) > 0:
            transition_voltages[code] = V_in[idx[0]]
    
    # Calculate code widths in voltage
    for code in range(N_codes):
        if code == 0:
            if not np.isnan(transition_voltages[0]):
                code_widths_volt[0] = transition_voltages[0]
        else:
            if not np.isnan(transition_voltages[code]) and not np.isnan(transition_voltages[code-1]):
                code_widths_volt[code] = transition_voltages[code] - transition_voltages[code-1]
    
    # Convert to LSB units
    code_widths_LSB = code_widths_volt / LSB
    
    # DNL = (measured width - ideal width) / ideal width
    # Ideal width is 1 LSB
    DNL_all = code_widths_LSB - 1.0
    
    # Find valid DNL values
    valid_dnl = ~np.isnan(DNL_all)
    if np.any(valid_dnl):
        DNL_max = np.max(np.abs(DNL_all[valid_dnl]))
    else:
        DNL_max = 0
    
    # Check for non-monotonicity (DNL < -1)
    non_monotonic_detected = np.any(DNL_all < -1)
    
    # Find missing codes
    missing_codes = []
    for code in range(N_codes):
        if not np.any(np.round(code_meas) == code):
            missing_codes.append(code)
    
    return (INL_max, DNL_max, code_ideal_corrected, non_monotonic_detected, 
            INL_all, DNL_all, transition_voltages, code_widths_LSB, missing_codes)


# --- 5. MAIN ANALYSIS AND VISUALIZATION FUNCTION ---
def analyze_static_ADC(digital_codes, ADC_name, N_bits, V_REF):
    """
    Perform complete static analysis of ADC data according to lab requirements.
    
    Parameters:
    -----------
    digital_codes : array
        Measured digital output codes
    ADC_name : str
        Name identifier for the ADC
    N_bits : int
        Number of bits
    V_REF : float
        Reference voltage
    """
    print(f"\n{'='*60}")
    print(f"Analiza statyczna przetwornika {ADC_name}")
    print(f"{'='*60}")
    
    N_codes = 2**N_bits
    
    # Step 1: Extract averaged codes
    V_in, code_meas = calculate_static_code_avg(digital_codes, N_codes)
    
    # Step 2: Calculate errors and parameters
    resolution_bits, LSB, Q_error, V_os, Delta_G, V_os_LSB = \
        calculate_adc_errors(V_in, code_meas, V_REF, N_bits)
    
    # Step 3: Calculate nonlinearity
    (INL_max, DNL_max, code_ideal_corrected, non_monotonic_detected, 
     INL_all, DNL_all, transition_voltages, code_widths_LSB, missing_codes) = \
        calculate_adc_nonlinearity(V_in, code_meas, LSB, V_os, N_bits)
    
    # --- PRINT RESULTS (zgodnie z poleceniem) ---
    print(f"\n{'─'*60}")
    print("PARAMETRY STATYCZNE PRZETWORNIKA ADC")
    print(f"{'─'*60}")
    
    print(f"\n1. Rozdzielczość:")
    print(f"   N = {resolution_bits} bitów")
    print(f"   Liczba poziomów kwantyzacji: {N_codes}")
    
    print(f"\n2. Błąd dyskretyzacji (kwantyzacji):")
    print(f"   LSB = {LSB*1000:.4f} mV")
    print(f"   Błąd kwantyzacji Q = ±LSB/2 = ±{Q_error*1000:.4f} mV")
    
    print(f"\n3. Błąd przesunięcia zera (offsetu):")
    print(f"   V_os = {V_os*1000:.4f} mV")
    print(f"   V_os = {V_os_LSB:.3f} LSB")
    
    print(f"\n4. Błąd skalowania (wzmocnienia):")
    print(f"   ΔG = {Delta_G:.3f} %")
    
    print(f"\n5. Błąd nieliniowości całkowej (INL):")
    print(f"   INL_max = {INL_max:.3f} LSB")
    
    print(f"\n6. Błąd nieliniowości różniczkowej (DNL):")
    print(f"   DNL_max = {DNL_max:.3f} LSB")
    
    print(f"\n7. Monotoniczność:")
    if non_monotonic_detected:
        print(f"   ⚠ WYKRYTO NIEMONOTONICZNOŚĆ (DNL < -1 LSB)")
        non_monotonic_codes = np.where(DNL_all < -1)[0]
        if len(non_monotonic_codes) > 0:
            print(f"   Kody niemonotoniczne: {non_monotonic_codes.tolist()}")
    else:
        print(f"   ✓ Przetwornik jest MONOTONICZNY")
    
    if len(missing_codes) > 0:
        print(f"\n8. Brakujące kody:")
        print(f"   ⚠ Wykryto {len(missing_codes)} brakujących kodów")
        if len(missing_codes) <= 10:
            print(f"   Kody: {missing_codes}")
        else:
            print(f"   Pierwsze 10 kodów: {missing_codes[:10]}")
    
    print(f"\n{'─'*60}\n")
    
    # --- VISUALIZATION ---
    
    # Plot 1: Transfer Characteristic (Charakterystyka przejściowa)
    plt.figure(figsize=(12, 7))
    code_ideal_nominal = (V_in - LSB/2) / LSB
    code_ideal_nominal = np.clip(code_ideal_nominal, 0, N_codes - 1)
    
    plt.plot(V_in, code_ideal_nominal, 'b--', label='Charakterystyka idealna', 
             alpha=0.7, linewidth=2)
    plt.plot(V_in, code_meas, 'g-', label='Charakterystyka zmierzona', linewidth=2)
    plt.plot(V_in, code_ideal_corrected, 'r:', label='Idealna skorygowana (offset+gain)', 
             alpha=0.8, linewidth=2)
    
    plt.title(f'Charakterystyka przejściowa - {ADC_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Napięcie wejściowe U$_{in}$ [V]', fontsize=12)
    plt.ylabel('Kod cyfrowy', fontsize=12)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim([0, V_REF])
    plt.ylim([-5, N_codes + 5])
    plt.tight_layout()
    plt.savefig(f'plots/{ADC_name}_charakterystyka_przejsciowa.png', dpi=150)
    plt.close()
    
    # Plot 2: INL vs Uin (zgodnie z poleceniem: "rozkład INL w funkcji Uin")
    plt.figure(figsize=(12, 5))
    plt.plot(V_in, INL_all, 'm-', linewidth=2, label='INL')
    plt.axhline(INL_max, color='r', linestyle='--', 
                label=f'INL$_{{max}}$ = ±{INL_max:.3f} LSB', linewidth=2)
    plt.axhline(-INL_max, color='r', linestyle='--', linewidth=2)
    plt.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    plt.fill_between(V_in, -INL_max, INL_max, alpha=0.2, color='red')
    
    plt.title(f'Nieliniowość całkowa (INL) - {ADC_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Napięcie wejściowe U$_{in}$ [V]', fontsize=12)
    plt.ylabel('INL [LSB]', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim([0, V_REF])
    plt.tight_layout()
    plt.savefig(f'plots/{ADC_name}_INL.png', dpi=150)
    plt.close()
    
    # Plot 3: DNL vs Uin (zgodnie z poleceniem: "rozkład DNL w funkcji Uin")
    # DNL is defined per code, map codes to their transition voltages
    valid_dnl = ~np.isnan(DNL_all)
    codes_with_dnl = np.arange(N_codes)[valid_dnl]
    V_transition_valid = transition_voltages[valid_dnl]
    DNL_valid = DNL_all[valid_dnl]
    
    plt.figure(figsize=(12, 5))
    plt.plot(V_transition_valid, DNL_valid, 'c.-', linewidth=2, markersize=4, label='DNL')
    plt.axhline(DNL_max, color='r', linestyle='--', 
                label=f'DNL$_{{max}}$ = ±{DNL_max:.3f} LSB', linewidth=2)
    plt.axhline(-DNL_max, color='r', linestyle='--', linewidth=2)
    plt.axhline(1, color='orange', linestyle=':', label='±1 LSB', linewidth=1.5, alpha=0.7)
    plt.axhline(-1, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    
    # Highlight non-monotonic regions
    if non_monotonic_detected:
        non_mono_mask = DNL_valid < -1
        if np.any(non_mono_mask):
            plt.scatter(V_transition_valid[non_mono_mask], DNL_valid[non_mono_mask], 
                       color='red', s=100, marker='x', linewidths=3, 
                       label='Niemonotoniczność', zorder=5)
    
    plt.title(f'Nieliniowość różniczkowa (DNL) - {ADC_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Napięcie wejściowe U$_{in}$ [V]', fontsize=12)
    plt.ylabel('DNL [LSB]', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim([0, V_REF])
    plt.tight_layout()
    plt.savefig(f'plots/{ADC_name}_DNL.png', dpi=150)
    plt.close()
    
    # Plot 4: Code Bin Widths (Szerokości kodów)
    plt.figure(figsize=(12, 5))
    codes_valid = np.arange(N_codes)[valid_dnl]
    plt.bar(codes_valid, code_widths_LSB[valid_dnl], 
            width=0.9, color='skyblue', edgecolor='navy', alpha=0.7, label='Szerokość kodu')
    plt.axhline(1, color='r', linestyle='--', label='Idealna szerokość (1 LSB)', linewidth=2)
    
    plt.title(f'Szerokości kodów cyfrowych - {ADC_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Kod cyfrowy', fontsize=12)
    plt.ylabel('Szerokość [LSB]', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.xlim([-1, N_codes])
    plt.tight_layout()
    plt.savefig(f'plots/{ADC_name}_szerokosci_kodow.png', dpi=150)
    plt.close()
    
    # Plot 5: Error summary (Podsumowanie błędów)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Transfer characteristic with errors
    ax1 = axes[0, 0]
    ax1.plot(V_in, code_ideal_nominal, 'b--', alpha=0.5, linewidth=1.5, label='Idealna')
    ax1.plot(V_in, code_meas, 'g-', linewidth=2, label='Zmierzona')
    ax1.set_title('Charakterystyka przejściowa', fontweight='bold')
    ax1.set_xlabel('U$_{in}$ [V]')
    ax1.set_ylabel('Kod cyfrowy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: INL
    ax2 = axes[0, 1]
    ax2.plot(V_in, INL_all, 'm-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax2.fill_between(V_in, -INL_max, INL_max, alpha=0.2, color='red')
    ax2.set_title(f'INL (max = {INL_max:.3f} LSB)', fontweight='bold')
    ax2.set_xlabel('U$_{in}$ [V]')
    ax2.set_ylabel('INL [LSB]')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: DNL
    ax3 = axes[1, 0]
    ax3.plot(V_transition_valid, DNL_valid, 'c.-', linewidth=2, markersize=4)
    ax3.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax3.axhline(-1, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    if non_monotonic_detected:
        non_mono_mask = DNL_valid < -1
        if np.any(non_mono_mask):
            ax3.scatter(V_transition_valid[non_mono_mask], DNL_valid[non_mono_mask], 
                       color='red', s=100, marker='x', linewidths=3, zorder=5)
    ax3.set_title(f'DNL (max = {DNL_max:.3f} LSB)', fontweight='bold')
    ax3.set_xlabel('U$_{in}$ [V]')
    ax3.set_ylabel('DNL [LSB]')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    PODSUMOWANIE PARAMETRÓW
    
    Rozdzielczość: {N_bits} bitów
    LSB: {LSB*1000:.4f} mV
    
    Błąd offsetu: {V_os*1000:.3f} mV ({V_os_LSB:.3f} LSB)
    Błąd wzmocnienia: {Delta_G:.3f} %
    
    INL max: {INL_max:.3f} LSB
    DNL max: {DNL_max:.3f} LSB
    
    Monotoniczność: {'NIE' if non_monotonic_detected else 'TAK'}
    Brakujące kody: {len(missing_codes)}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'plots/{ADC_name}_podsumowanie.png', dpi=150)
    plt.close()
    
    print(f"Wykresy zapisane w katalogu 'plots/'\n")


# --- RUN ANALYSIS ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PARAMETRYZACJA PRZETWORNIKA ANALOGOWO-CYFROWEGO")
    print("Laboratorium Elektroniczne WFiIS C-7")
    print("="*60)
    print(f"\nKonfiguracja: {N_BITS}-bitowy ADC, V_REF = {V_REF} V")
    
    analyze_static_ADC(digital_codes, 'ADC0804', N_BITS, V_REF)
    
    print("\n" + "="*60)
    print("ANALIZA ZAKOŃCZONA")
    print("="*60)