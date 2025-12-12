import numpy as np
import matplotlib.pyplot as plt

# --- 1. Ładowanie Danych ---
# Wymagane pliki: DAC1_statyczne.csv, DAC2_statyczne.csv
file1 = 'dane/DAC1_statyczne.csv'
file2 = 'dane/DAC2_statyczne.csv'

# Ładowanie danych z pominięciem 11 wierszy nagłówka
try:
    DAC1 = np.loadtxt(file1, delimiter=',', skiprows=11)
    DAC2 = np.loadtxt(file2, delimiter=',', skiprows=11)
except FileNotFoundError as e:
    print(f"Błąd ładowania pliku: {e}. Upewnij się, że pliki są w tym samym katalogu.")
    # W rzeczywistości, jeżeli pliki nie istnieją, program zatrzyma się tutaj.

# Ustawienie zmiennych globalnych dla wyników INL/DNL (potrzebne do wykreślania)
INL_all = None
DNL_all = None

# --- 2. PARAMETRY DAC ---
N_BITS = 8          # Liczba bitów DAC (na podstawie PDF)
V_FS = 5.0          # Maksymalne napięcie przetwarzania (Pełna Skala) [V]

# --- 3. FUNKCJA OBLICZANIA UŚREDNIONEGO NAPIĘCIA NA KOD ---
def calculate_static_V_avg(V_out, digital_codes, N_codes):
    """
    Oblicza uśrednione napięcia wyjściowe dla każdego kodu cyfrowego.
    """
    unique_codes = np.arange(N_codes)
    V_meas_avg = np.zeros(N_codes)
    
    for code in unique_codes:
        # Uśrednianie wszystkich pomiarów dla danego kodu
        voltages_for_code = V_out[digital_codes == code]
        if len(voltages_for_code) > 0:
            V_meas_avg[code] = np.mean(voltages_for_code)
        else:
            V_meas_avg[code] = np.nan 

    # Usuń ewentualne kody, które nie zostały zmierzone (jeśli są NaN)
    V_meas_avg = V_meas_avg[~np.isnan(V_meas_avg)]

    return V_meas_avg

# --- 4. FUNKCJA OBLICZANIA BŁĘDÓW STATYCZNYCH (Vos, LSB, Delta_G) ---
def calculate_dac_errors(V_meas, V_FS, N_bits):
    """
    Oblicza Błąd Przesunięcia Zera (Vos), LSB i Błąd Wzmocnienia (Delta_G).
    """
    V_D0 = V_meas[0] 
    V_Dmax = V_meas[-1] 
    
    # Błąd Przesunięcia Zera
    V_os = V_D0
    
    # LSB (Idealny kwant)
    LSB = V_FS / (2**N_bits)
    
    # Błąd Wzmocnienia
    D_max = 2**N_bits - 1
    V_ideal_Dmax = V_FS - LSB # Idealna wartość dla kodu max
    
    Delta_G = 100 * (((V_Dmax - V_os) / V_ideal_Dmax) - 1)
    
    return V_os, LSB, Delta_G

# --- 5. FUNKCJA OBLICZANIA NIELINIOWOŚCI (INL, DNL) ---
def calculate_dac_nonlinearity(V_meas, LSB, V_os, N_bits):
    """
    Oblicza Całkową (INL) i Różniczkową (DNL) Nieliniowość.
    """
    codes = np.arange(len(V_meas))
    D_max = 2**N_bits - 1
    
    # Skalibrowany LSB (slope linii INL)
    LSB_cal = (V_meas[-1] - V_os) / D_max 

    # Idealna charakterystyka skorygowana (do INL)
    V_ideal_corrected = V_os + codes * LSB_cal
    
    global INL_all, DNL_all
    
    # Całkowa Nieliniowość (INL)
    INL_all = (V_meas - V_ideal_corrected) / LSB
    INL_max = np.max(np.abs(INL_all))

    # Różniczkowa Nieliniowość (DNL)
    V_step_measured = np.diff(V_meas)
    
    DNL_all = (V_step_measured - LSB_cal) / LSB_cal
    DNL_max = np.max(np.abs(DNL_all))
    
    non_monotonicity = np.min(DNL_all) < -1
        
    return INL_max, DNL_max, V_ideal_corrected, non_monotonicity

# --- 6. FUNKCJA GŁÓWNA DO ANALIZY I WIZUALIZACJI ---
def analyze_static_DAC(data, DAC_name, N_bits, V_FS):
    
    # Dane wejściowe
    dac_data = data[:,1]
    trigger = data[:,2]
    
    # Obcięcie danych (Zaimplementowana przez Ciebie logika)
    start_threshold = 1.5 
    end_threshold = 1.5 
    
    try:
        start_index = np.where(trigger < start_threshold)[0][0]
        end_index = np.where(trigger > end_threshold)[0][-1]
    except IndexError:
        print(f"Błąd: Nie znaleziono progów w danych dla {DAC_name}. Analiza pełnego zestawu.")
        start_index = 0
        end_index = len(data)

    dac_data_truncated = dac_data[start_index:end_index]
    N_points = len(dac_data_truncated)
    
    if N_points == 0:
        print(f"Błąd: Obcięte dane dla {DAC_name} są puste.")
        return

    # Inicjalizacja kodów: Zakładamy, że sweep odbywa się równomiernie
    N_codes = 2**N_bits
    N_samples_per_code = N_points // N_codes
    
    # Wygenerowanie kodu cyfrowego dla każdego punktu pomiarowego
    inferred_codes = np.repeat(np.arange(N_codes), N_samples_per_code)

    # Przycięcie napięcia wyjściowego, aby zgadzało się z liczbą kodów
    dac_data_truncated = dac_data_truncated[:len(inferred_codes)]

    # Krok 1: Ekstrakcja uśrednionych napięć
    V_meas = calculate_static_V_avg(dac_data_truncated, inferred_codes, N_codes)
    codes = np.arange(len(V_meas))
    
    # Krok 2: Obliczenie błędów
    V_os, LSB, Delta_G = calculate_dac_errors(V_meas, V_FS, N_bits)
    
    # Krok 3: Obliczenie nieliniowości
    INL_max, DNL_max, V_ideal_corrected, non_monotonicity = \
        calculate_dac_nonlinearity(V_meas, LSB, V_os, N_bits)

    # --- WYNIKI ANALIZY ---
    print(f"--- WYNIKI ANALIZY STATYCZNEJ DAC: {DAC_name} ---")
    print(f"Błąd przesunięcia zera (V_os): {V_os:.4f} V")
    print(f"Błąd wzmocnienia (Delta_G): {Delta_G:.2f} %")
    print(f"Całkowa Nieliniowość (INL_max): {INL_max:.3f} LSB")
    print(f"Różniczkowa Nieliniowość (DNL_max): {DNL_max:.3f} LSB")
    if non_monotonicity:
        print(" -> UWAGA: DAC jest niemonotoniczny (DNL < -1 LSB).")
    print("-" * 40)

    # --- WIZUALIZACJA (Charakterystyka przejściowa) ---
    plt.figure(figsize=(10, 6))
    V_ideal_nominal = codes * LSB
    plt.plot(codes, V_ideal_nominal, 'b--', label='Idealna (Nominalna)', alpha=0.6)
    plt.plot(codes, V_meas, 'g-o', label='Rzeczywista (Zmierzona)', markersize=3)
    plt.plot(codes, V_ideal_corrected, 'r:', label='Idealna (Skorygowana INL)', alpha=0.7)
    plt.title(f'Charakterystyka Przejściowa Układu DAC ({DAC_name})')
    plt.xlabel('Kod Cyfrowy D')
    plt.ylabel('Napięcie Wyjściowe [V]')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'plots/{DAC_name}_charakterystyka_przejściowa.png')
    
    # --- WIZUALIZACJA (INL) ---
    plt.figure(figsize=(10, 3))
    plt.plot(codes, INL_all, 'm-')
    plt.axhline(INL_max, color='r', linestyle='--', label=f'INL Max ({INL_max:.3f} LSB)')
    plt.axhline(-INL_max, color='r', linestyle='--')
    plt.axhline(0, color='k', linestyle='-')
    plt.title(f'Całkowa Nieliniowość (INL) {DAC_name}')
    plt.xlabel('Kod Cyfrowy D')
    plt.ylabel('INL [LSB]')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'plots/{DAC_name}_INL.png')

    # --- WIZUALIZACJA (DNL) ---
    codes_dnl = codes[:-1] + 0.5 
    plt.figure(figsize=(10, 3))
    plt.plot(codes_dnl, DNL_all, 'c-')
    plt.axhline(DNL_max, color='r', linestyle='--', label=f'DNL Max ({DNL_max:.3f} LSB)')
    plt.axhline(-DNL_max, color='r', linestyle='--')
    plt.axhline(1, color='g', linestyle=':', label='+1 LSB')
    plt.axhline(-1, color='g', linestyle=':', label='-1 LSB')
    plt.axhline(0, color='k', linestyle='-')
    plt.title(f'Różniczkowa Nieliniowość (DNL) {DAC_name}')
    plt.xlabel('Kod Cyfrowy D')
    plt.ylabel('DNL [LSB]')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'plots/{DAC_name}_DNL.png')

# --- Uruchomienie analizy ---
analyze_static_DAC(DAC1, 'DAC1', N_BITS, V_FS)
analyze_static_DAC(DAC2, 'DAC2', N_BITS, V_FS)