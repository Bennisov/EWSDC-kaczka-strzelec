import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURATION ---
V_REF = 5.0             # Reference voltage [V]
N_BITS = 8              # Number of DAC bits
TRIGGER_THRESHOLD = 1.5 # Trigger threshold [V]

# Calculate LSB voltage
LSB_VOLTAGE = V_REF / (2**N_BITS)  # ~19.53 mV for 8-bit, 5V

# Stabilization detection parameters
NOISE_THRESHOLD_MULTIPLIER = 3.0  # Multiplier for std dev
STABLE_SAMPLES = 30               # Consecutive samples needed

# Edge detection parameters - ADAPTIVE based on transition type
EDGE_THRESHOLD_PERCENT = 20  # % of expected voltage change

# Rise/Fall time measurement
RISE_FALL_LOW = 10   # Lower threshold for rise/fall time [%]
RISE_FALL_HIGH = 90  # Upper threshold for rise/fall time [%]

# Create output directory
os.makedirs('plots', exist_ok=True)

# --- 1. LOAD DATA ---
files = {
    'DAC2_LSB_rising': 'dane/DAC2_dynamiczne_LSB_narastanie.csv',
    'DAC2_LSB_falling': 'dane/DAC2_dynamiczne_LSB_opadanie.csv',
    'DAC2_FS_rising': 'dane/DAC2_dynamiczne_fullscale_narastanie.csv',
    'DAC2_FS_falling': 'dane/DAC2_dynamiczne_fullscale_opadanie.csv'
}

data_sets = {}
try:
    for name, filepath in files.items():
        data_sets[name] = np.loadtxt(filepath, delimiter=',', skiprows=11)
        print(f"Loaded: {name}")
    print("All data files loaded successfully.\n")
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    print("Make sure the files are in the 'dane/' directory.")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error loading data: {e}")
    sys.exit(1)


# --- 2. ESTIMATE TRANSITION MAGNITUDE ---
def estimate_transition_magnitude(data, is_rising):
    """
    Estimate the expected magnitude of the transition by comparing
    the beginning and end of the data.
    
    Returns expected voltage change and adaptive edge threshold.
    """
    # Use first and last portions to estimate initial and final values
    n_samples = min(100, len(data) // 10)
    
    initial_value = np.median(data[:n_samples, 1])
    final_value = np.median(data[-n_samples:, 1])
    
    voltage_change = abs(final_value - initial_value)
    
    # Adaptive edge threshold: 20% of expected change
    edge_threshold = voltage_change * (EDGE_THRESHOLD_PERCENT / 100.0)
    
    # Minimum threshold to avoid noise triggering
    min_threshold = 3 * np.std(data[:n_samples, 1])
    edge_threshold = max(edge_threshold, min_threshold)
    
    return voltage_change, edge_threshold, initial_value, final_value


# --- 3. DETECT TRANSITION START ---
def detect_transition_start(data, is_rising, edge_threshold):
    """
    Detect the start of voltage transition using adaptive threshold.
    """
    voltage = data[:, 1]
    
    # Use a moving average to reduce noise sensitivity
    window = 5
    voltage_smoothed = np.convolve(voltage, np.ones(window)/window, mode='valid')
    diff = np.diff(voltage_smoothed)
    
    if is_rising:
        edge_indices = np.where(diff > edge_threshold)[0]
    else:
        edge_indices = np.where(diff < -edge_threshold)[0]
    
    if len(edge_indices) == 0:
        # Fallback: find largest change
        if is_rising:
            start_idx = np.argmax(diff)
        else:
            start_idx = np.argmin(diff)
        print(f"Warning: Using largest change at index {start_idx}")
    else:
        start_idx = edge_indices[0]
    
    start_time = data[start_idx, 0]
    
    return start_idx, start_time


# --- 4. CALCULATE RISE/FALL TIME ---
def calculate_rise_fall_time(data, start_idx, initial_value, final_value,
                             is_rising, low_percent=RISE_FALL_LOW, 
                             high_percent=RISE_FALL_HIGH):
    """
    Calculate rise or fall time (10%-90% by default).
    """
    delta_v = final_value - initial_value
    
    # Calculate threshold voltages
    v_low = initial_value + (low_percent / 100.0) * delta_v
    v_high = initial_value + (high_percent / 100.0) * delta_v
    
    # Search window: from start to reasonable end point
    search_end = min(len(data), start_idx + 5000)
    voltage = data[start_idx:search_end, 1]
    time = data[start_idx:search_end, 0]
    
    # Find crossing points
    if is_rising:
        idx_low = np.where(voltage >= v_low)[0]
        idx_high = np.where(voltage >= v_high)[0]
    else:
        idx_low = np.where(voltage <= v_low)[0]
        idx_high = np.where(voltage <= v_high)[0]
    
    if len(idx_low) == 0 or len(idx_high) == 0:
        print(f"Warning: Could not find {low_percent}% or {high_percent}% crossing points")
        return None, None, None, v_low, v_high
    
    t_low = time[idx_low[0]]
    t_high = time[idx_high[0]]
    
    rise_fall_time = t_high - t_low
    
    return rise_fall_time, t_low, t_high, v_low, v_high


# --- 5. FIND SETTLING TIME ---
def find_settling_time(data, start_idx, final_value, expected_change,
                       noise_mult=NOISE_THRESHOLD_MULTIPLIER,
                       stable_samples=STABLE_SAMPLES):
    """
    Find the time when signal settles within specified band.
    For small signals (LSB), use primarily noise-based threshold.
    For large signals (FS), use percentage-based threshold.
    """
    # Calculate noise level from end of signal
    n_samples = min(200, len(data) - start_idx)
    std_dev = np.std(data[-n_samples:, 1])
    
    # For LSB transitions: use noise-based threshold (more sensitive)
    # For FS transitions: use percentage-based threshold
    if expected_change < 0.5:  # Less than 0.5V is considered "small signal"
        # Small signal (LSB): use 3× noise as settling band
        settling_band = noise_mult * std_dev
        print(f"  Using noise-based settling band: ±{settling_band*1000:.3f} mV")
    else:
        # Large signal (FS): use 1% of final value
        settling_band = max(abs(final_value) * 0.01, noise_mult * std_dev)
        print(f"  Using percentage-based settling band: ±{settling_band*1000:.3f} mV")
    
    # Search for settling point
    voltage = data[:, 1]
    time = data[:, 0]
    
    # Start searching after initial transition
    search_start = start_idx + 10
    
    for i in range(search_start, len(data) - stable_samples + 1):
        window = voltage[i:i + stable_samples]
        if np.all(np.abs(window - final_value) < settling_band):
            settling_time = time[i]
            return settling_time, settling_band
    
    print("  Warning: Signal did not settle within data window")
    return None, settling_band


# --- 6. ANALYZE AND PLOT DYNAMIC RESPONSE ---
def analyze_dynamic_response(data, name, is_rising, is_lsb_transition=False):
    """
    Complete analysis of dynamic DAC response.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")
    
    # Normalize time to start at 0
    data_normalized = data.copy()
    data_normalized[:, 0] = data_normalized[:, 0] - data_normalized[0, 0]
    
    time = data_normalized[:, 0]
    voltage = data_normalized[:, 1]
    
    # Step 1: Estimate transition magnitude
    expected_change, edge_threshold, initial_est, final_est = \
        estimate_transition_magnitude(data_normalized, is_rising)
    
    print(f"  Expected voltage change: {expected_change*1000:.2f} mV")
    print(f"  Adaptive edge threshold: {edge_threshold*1000:.3f} mV")
    
    if is_lsb_transition:
        expected_lsb_change = LSB_VOLTAGE
        print(f"  Theoretical LSB change: {expected_lsb_change*1000:.2f} mV")
    
    # Step 2: Detect transition start
    start_idx, start_time = detect_transition_start(
        data_normalized, is_rising, edge_threshold
    )
    
    print(f"  Transition detected at: t={start_time*1e6:.2f} µs, V={voltage[start_idx]:.4f} V")
    
    # Refine initial and final values based on actual data
    n_pre = min(50, start_idx)
    initial_value = np.mean(voltage[max(0, start_idx-n_pre):start_idx])
    final_value = final_est  # Use the estimate from full data
    
    actual_change = abs(final_value - initial_value)
    print(f"  Measured voltage change: {actual_change*1000:.2f} mV")
    
    # Step 3: Calculate rise/fall time
    rf_time, t_low, t_high, v_low, v_high = calculate_rise_fall_time(
        data_normalized, start_idx, initial_value, final_value, is_rising
    )
    
    # Step 4: Find settling time
    settling_time, settling_band = find_settling_time(
        data_normalized, start_idx, final_value, expected_change
    )
    
    # Calculate metrics relative to start
    if rf_time is not None:
        rf_time_from_start = rf_time
    
    if settling_time is not None:
        settling_time_from_start = settling_time - start_time
    
    # Print results
    print(f"\n  Results:")
    print(f"    Initial voltage:           {initial_value:.4f} V")
    print(f"    Final voltage:             {final_value:.4f} V")
    print(f"    Voltage change:            {actual_change*1000:.2f} mV")
    
    if rf_time is not None:
        print(f"    Rise/Fall time (10%-90%):  {rf_time*1e6:.2f} µs")
    else:
        print(f"    Rise/Fall time:            Could not be determined")
    
    if settling_time is not None:
        print(f"    Settling time:             {settling_time_from_start*1e6:.2f} µs")
        print(f"    Total time (start to settled): {settling_time_from_start*1e6:.2f} µs")
    else:
        print(f"    Settling time:             Could not be determined")
    
    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Full response
    ax1.plot(time * 1e6, voltage, 'b-', linewidth=1, label='DAC Output', alpha=0.7)
    ax1.axvline(x=start_time * 1e6, color='k', linestyle='--', 
                linewidth=2, label='Transition Start')
    ax1.axhline(y=initial_value, color='gray', linestyle=':', 
                linewidth=1, alpha=0.5, label='Initial Value')
    ax1.axhline(y=final_value, color='gray', linestyle=':', 
                linewidth=1, alpha=0.5, label='Final Value')
    
    if rf_time is not None:
        ax1.axvline(x=t_low * 1e6, color='g', linestyle=':', 
                   linewidth=1.5, alpha=0.7, label='10% Point')
        ax1.axvline(x=t_high * 1e6, color='g', linestyle=':', 
                   linewidth=1.5, alpha=0.7, label='90% Point')
        ax1.axhline(y=v_low, color='g', linestyle=':', linewidth=1, alpha=0.3)
        ax1.axhline(y=v_high, color='g', linestyle=':', linewidth=1, alpha=0.3)
    
    if settling_time is not None:
        ax1.axvline(x=settling_time * 1e6, color='r', linestyle='--', 
                   linewidth=2, label='Settled')
        ax1.axhspan(final_value - settling_band, final_value + settling_band,
                   alpha=0.15, color='red', label='Settling Band')
    
    ax1.set_title(f'Dynamic Response - {name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time [µs]', fontsize=12)
    ax1.set_ylabel('Voltage [V]', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed to transition
    if settling_time is not None:
        zoom_end = settling_time * 1.5
    else:
        zoom_end = time[min(start_idx + 2000, len(time)-1)]
    
    zoom_start = start_time * 0.8
    zoom_mask = (time >= zoom_start) & (time <= zoom_end)
    
    ax2.plot(time[zoom_mask] * 1e6, voltage[zoom_mask], 'b-', linewidth=1.5)
    ax2.axvline(x=start_time * 1e6, color='k', linestyle='--', linewidth=2)
    ax2.axhline(y=initial_value, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(y=final_value, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    if rf_time is not None:
        ax2.axvline(x=t_low * 1e6, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=t_high * 1e6, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=v_low, color='g', linestyle=':', linewidth=1, alpha=0.3)
        ax2.axhline(y=v_high, color='g', linestyle=':', linewidth=1, alpha=0.3)
    
    if settling_time is not None:
        ax2.axvline(x=settling_time * 1e6, color='r', linestyle='--', linewidth=2)
        ax2.axhspan(final_value - settling_band, final_value + settling_band,
                   alpha=0.2, color='red')
    
    ax2.set_title('Zoomed View - Transition Detail', fontsize=12)
    ax2.set_xlabel('Time [µs]', fontsize=12)
    ax2.set_ylabel('Voltage [V]', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    safe_name = name.replace(' ', '_').replace('/', '_').replace('→', 'to')
    plt.savefig(f'plots/{safe_name}_dynamic.png', dpi=150)
    plt.close()
    
    print(f"  Plot saved to 'plots/{safe_name}_dynamic.png'")
    
    # Return results for summary
    return {
        'name': name,
        'start_time': start_time,
        'initial_voltage': initial_value,
        'final_voltage': final_value,
        'voltage_change': actual_change,
        'rise_fall_time': rf_time if rf_time is not None else np.nan,
        'settling_time': settling_time_from_start if settling_time is not None else np.nan,
        'settling_band': settling_band
    }


# --- 7. RUN ANALYSIS ---
if __name__ == "__main__":
    print("="*60)
    print("DAC DYNAMIC CHARACTERISTICS ANALYSIS")
    print("="*60)
    print(f"Configuration:")
    print(f"  Reference Voltage: {V_REF} V")
    print(f"  DAC Resolution: {N_BITS} bits")
    print(f"  LSB Voltage: {LSB_VOLTAGE*1000:.2f} mV")
    print(f"  Stable Samples Required: {STABLE_SAMPLES}")
    
    results = []
    
    # Analyze each dataset
    result = analyze_dynamic_response(
        data_sets['DAC2_LSB_rising'], 
        'DAC2 LSB Rising (126→127)',
        is_rising=True,
        is_lsb_transition=True
    )
    results.append(result)
    
    result = analyze_dynamic_response(
        data_sets['DAC2_LSB_falling'],
        'DAC2 LSB Falling (127→126)',
        is_rising=False,
        is_lsb_transition=True
    )
    results.append(result)
    
    result = analyze_dynamic_response(
        data_sets['DAC2_FS_rising'],
        'DAC2 Fullscale Rising (0→255)',
        is_rising=True,
        is_lsb_transition=False
    )
    results.append(result)
    
    result = analyze_dynamic_response(
        data_sets['DAC2_FS_falling'],
        'DAC2 Fullscale Falling (255→0)',
        is_rising=False,
        is_lsb_transition=False
    )
    results.append(result)
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    print(f"{'Test':<32} {'ΔV [mV]':>10} {'Rise/Fall [µs]':>15} {'Settling [µs]':>15}")
    print("-"*72)
    
    for r in results:
        delta_v_str = f"{r['voltage_change']*1000:.2f}"
        rf_str = f"{r['rise_fall_time']*1e6:.2f}" if not np.isnan(r['rise_fall_time']) else "N/A"
        st_str = f"{r['settling_time']*1e6:.2f}" if not np.isnan(r['settling_time']) else "N/A"
        print(f"{r['name']:<32} {delta_v_str:>10} {rf_str:>15} {st_str:>15}")
    
    print("\n" + "="*60)
    print("Analysis Complete! All plots saved to 'plots/' directory")
    print("="*60)