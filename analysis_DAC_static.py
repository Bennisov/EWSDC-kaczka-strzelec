import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURATION ---
N_BITS = 8          # Number of DAC bits
V_FS = 5.0          # Full scale voltage [V]
START_THRESHOLD = 1.5  # Trigger threshold for data start
END_THRESHOLD = 1.5    # Trigger threshold for data end

# Create output directory
os.makedirs('plots', exist_ok=True)

# --- 1. LOAD DATA ---
file1 = 'dane/DAC1_statyczne.csv'
file2 = 'dane/DAC2_statyczne.csv'

try:
    DAC1 = np.loadtxt(file1, delimiter=',', skiprows=11)
    DAC2 = np.loadtxt(file2, delimiter=',', skiprows=11)
    print("Data files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    print("Make sure the files are in the 'dane/' directory.")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error loading data: {e}")
    sys.exit(1)


# --- 2. CALCULATE AVERAGED VOLTAGE PER CODE ---
def calculate_static_V_avg(V_out, digital_codes, N_codes):
    """
    Calculate averaged output voltages for each digital code.
    
    Parameters:
    -----------
    V_out : array
        Measured output voltages
    digital_codes : array
        Corresponding digital codes
    N_codes : int
        Total number of codes (2^N_bits)
        
    Returns:
    --------
    V_meas_avg : array
        Average voltage for each code
    """
    unique_codes = np.arange(N_codes)
    V_meas_avg = np.zeros(N_codes)
    
    for code in unique_codes:
        voltages_for_code = V_out[digital_codes == code]
        if len(voltages_for_code) > 0:
            V_meas_avg[code] = np.mean(voltages_for_code)
        else:
            V_meas_avg[code] = np.nan
    
    # Remove codes that weren't measured
    valid_mask = ~np.isnan(V_meas_avg)
    if not np.all(valid_mask):
        print(f"Warning: {np.sum(~valid_mask)} codes have no measurements")
    
    V_meas_avg = V_meas_avg[valid_mask]
    return V_meas_avg


# --- 3. CALCULATE STATIC ERRORS (Vos, LSB, Delta_G) ---
def calculate_dac_errors(V_meas, V_FS, N_bits):
    """
    Calculate Offset Error (Vos), LSB, and Gain Error (Delta_G).
    
    Parameters:
    -----------
    V_meas : array
        Measured voltages for each code
    V_FS : float
        Full scale voltage
    N_bits : int
        Number of bits
        
    Returns:
    --------
    V_os : float
        Offset voltage [V]
    LSB : float
        Ideal LSB voltage [V]
    Delta_G : float
        Gain error [%]
    """
    V_D0 = V_meas[0]      # Voltage at code 0
    V_Dmax = V_meas[-1]   # Voltage at maximum code
    
    # Offset Error
    V_os = V_D0
    
    # Ideal LSB
    LSB = V_FS / (2**N_bits)
    
    # Gain Error
    D_max = 2**N_bits - 1
    V_ideal_Dmax = V_FS - LSB  # Ideal voltage for max code
    
    Delta_G = 100 * (((V_Dmax - V_os) / V_ideal_Dmax) - 1)
    
    return V_os, LSB, Delta_G


# --- 4. CALCULATE NONLINEARITY (INL, DNL) ---
def calculate_dac_nonlinearity(V_meas, LSB, V_os, N_bits):
    """
    Calculate Integral (INL) and Differential (DNL) Nonlinearity.
    
    Parameters:
    -----------
    V_meas : array
        Measured voltages
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
    V_ideal_corrected : array
        Ideal corrected voltages
    non_monotonicity : bool
        True if DAC is non-monotonic
    INL_all : array
        INL for all codes
    DNL_all : array
        DNL for all code transitions
    """
    codes = np.arange(len(V_meas))
    D_max = 2**N_bits - 1
    
    # Calibrated LSB (slope of best-fit line)
    LSB_cal = (V_meas[-1] - V_os) / D_max
    
    # Ideal characteristic with offset and gain correction
    V_ideal_corrected = V_os + codes * LSB_cal
    
    # Integral Nonlinearity (INL)
    INL_all = (V_meas - V_ideal_corrected) / LSB
    INL_max = np.max(np.abs(INL_all))
    
    # Differential Nonlinearity (DNL)
    V_step_measured = np.diff(V_meas)
    # CORRECTED: Use ideal LSB as reference, not calibrated
    DNL_all = (V_step_measured - LSB) / LSB
    DNL_max = np.max(np.abs(DNL_all))
    
    # Check for non-monotonicity
    non_monotonicity = np.min(DNL_all) < -1
    
    return INL_max, DNL_max, V_ideal_corrected, non_monotonicity, INL_all, DNL_all


# --- 5. MAIN ANALYSIS AND VISUALIZATION FUNCTION ---
def analyze_static_DAC(data, DAC_name, N_bits, V_FS, 
                       start_threshold=START_THRESHOLD, 
                       end_threshold=END_THRESHOLD):
    """
    Perform complete static analysis of DAC data.
    
    Parameters:
    -----------
    data : array
        Raw measurement data
    DAC_name : str
        Name identifier for the DAC
    N_bits : int
        Number of bits
    V_FS : float
        Full scale voltage
    start_threshold : float
        Trigger threshold for start
    end_threshold : float
        Trigger threshold for end
    """
    print(f"\n{'='*50}")
    print(f"Analyzing {DAC_name}...")
    print(f"{'='*50}")
    
    # Extract data columns
    dac_data = data[:, 1]
    trigger = data[:, 2]
    
    # Truncate data based on trigger thresholds
    try:
        start_indices = np.where(trigger < start_threshold)[0]
        end_indices = np.where(trigger > end_threshold)[0]
        
        if len(start_indices) == 0 or len(end_indices) == 0:
            raise IndexError("No valid trigger points found")
            
        start_index = start_indices[0]
        end_index = end_indices[-1]
        
        print(f"Data truncated: indices {start_index} to {end_index}")
        
    except IndexError:
        print(f"Warning: Could not find trigger thresholds. Using full dataset.")
        start_index = 0
        end_index = len(data)
    
    dac_data_truncated = dac_data[start_index:end_index]
    N_points = len(dac_data_truncated)
    
    if N_points == 0:
        print(f"Error: Truncated data for {DAC_name} is empty. Skipping analysis.")
        return
    
    # Generate digital codes (assumes uniform sweep)
    N_codes = 2**N_bits
    N_samples_per_code = N_points // N_codes
    
    if N_samples_per_code == 0:
        print(f"Error: Not enough data points. Need at least {N_codes} points.")
        return
    
    print(f"Total points: {N_points}, Samples per code: {N_samples_per_code}")
    
    # Infer digital codes for each measurement
    inferred_codes = np.repeat(np.arange(N_codes), N_samples_per_code)
    
    # Truncate voltage data to match code length
    data_length = len(inferred_codes)
    dac_data_truncated = dac_data_truncated[:data_length]
    
    remaining_points = N_points - data_length
    if remaining_points > 0:
        print(f"Warning: {remaining_points} data points discarded due to rounding")
    
    # Step 1: Extract averaged voltages
    V_meas = calculate_static_V_avg(dac_data_truncated, inferred_codes, N_codes)
    codes = np.arange(len(V_meas))
    
    # Step 2: Calculate errors
    V_os, LSB, Delta_G = calculate_dac_errors(V_meas, V_FS, N_bits)
    
    # Step 3: Calculate nonlinearity
    INL_max, DNL_max, V_ideal_corrected, non_monotonicity, INL_all, DNL_all = \
        calculate_dac_nonlinearity(V_meas, LSB, V_os, N_bits)
    
    # --- PRINT RESULTS ---
    print(f"\n--- STATIC ANALYSIS RESULTS: {DAC_name} ---")
    print(f"Offset Error (V_os):           {V_os:.4f} V")
    print(f"Ideal LSB:                     {LSB:.4f} V")
    print(f"Gain Error (Delta_G):          {Delta_G:.2f} %")
    print(f"Integral Nonlinearity (INL):   {INL_max:.3f} LSB")
    print(f"Differential Nonlinearity (DNL): {DNL_max:.3f} LSB")
    if non_monotonicity:
        print(" ⚠ WARNING: DAC is NON-MONOTONIC (DNL < -1 LSB)")
    else:
        print(" ✓ DAC is monotonic")
    print("-" * 50)
    
    # --- VISUALIZATION ---
    
    # Plot 1: Transfer Characteristic
    plt.figure(figsize=(10, 6))
    V_ideal_nominal = codes * LSB
    plt.plot(codes, V_ideal_nominal, 'b--', label='Ideal (Nominal)', alpha=0.6, linewidth=2)
    plt.plot(codes, V_meas, 'g-o', label='Measured', markersize=3)
    plt.plot(codes, V_ideal_corrected, 'r:', label='Ideal (Corrected)', alpha=0.7, linewidth=2)
    plt.title(f'Transfer Characteristic - {DAC_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Digital Code', fontsize=12)
    plt.ylabel('Output Voltage [V]', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'plots/{DAC_name}_transfer_characteristic.png', dpi=150)
    plt.close()
    
    # Plot 2: INL
    plt.figure(figsize=(10, 4))
    plt.plot(codes, INL_all, 'm-', linewidth=1.5)
    plt.axhline(INL_max, color='r', linestyle='--', 
                label=f'INL Max = ±{INL_max:.3f} LSB', linewidth=2)
    plt.axhline(-INL_max, color='r', linestyle='--', linewidth=2)
    plt.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    plt.title(f'Integral Nonlinearity (INL) - {DAC_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Digital Code', fontsize=12)
    plt.ylabel('INL [LSB]', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'plots/{DAC_name}_INL.png', dpi=150)
    plt.close()
    
    # Plot 3: DNL
    codes_dnl = codes[:-1] + 0.5  # DNL is between codes
    plt.figure(figsize=(10, 4))
    plt.plot(codes_dnl, DNL_all, 'c-', linewidth=1.5)
    plt.axhline(DNL_max, color='r', linestyle='--', 
                label=f'DNL Max = ±{DNL_max:.3f} LSB', linewidth=2)
    plt.axhline(-DNL_max, color='r', linestyle='--', linewidth=2)
    plt.axhline(1, color='g', linestyle=':', label='±1 LSB', linewidth=1.5, alpha=0.7)
    plt.axhline(-1, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    plt.title(f'Differential Nonlinearity (DNL) - {DAC_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Digital Code', fontsize=12)
    plt.ylabel('DNL [LSB]', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'plots/{DAC_name}_DNL.png', dpi=150)
    plt.close()
    
    print(f"Plots saved to 'plots/' directory\n")


# --- RUN ANALYSIS ---
if __name__ == "__main__":
    print("Starting DAC Static Characteristics Analysis")
    print(f"Configuration: {N_BITS}-bit DAC, V_FS = {V_FS} V\n")
    
    analyze_static_DAC(DAC1, 'DAC1', N_BITS, V_FS)
    analyze_static_DAC(DAC2, 'DAC2', N_BITS, V_FS)
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    print("="*50)