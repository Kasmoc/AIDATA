import math

def moving_average(data, window_size=5):
    """
    Parameters:
    data (list): Input signal data
    window_size (int): Size of the moving average window
    
    Returns:
    list: Smoothed signal
    """
    n = len(data)
    smoothed = []
    half_window = window_size // 2
    
    # Pad the data manually
    padded = [data[0]] * half_window + data + [data[-1]] * half_window
    
    # Manual moving average calculation
    for i in range(n):
        window_sum = 0
        for j in range(window_size):
            window_sum += padded[i + j]
        smoothed.append(window_sum / window_size)
    
    return smoothed

def frequency_filter(data, freq_cutoff=0.1):
    """
    Basic frequency filter with hardcoded DFT using math library.
    
    Parameters:
    data (list): Input signal data
    freq_cutoff (float): Frequency cutoff (0 to 1) for high-pass filter
    
    Returns:
    list: Filtered signal
    """
    n = len(data)
    filtered = [0] * n
    
    # Hardcoded Discrete Fourier Transform (DFT)
    dft = [0] * n
    for k in range(n):
        real = 0
        imag = 0
        for t in range(n):
            angle = 2 * math.pi * k * t / n
            real += data[t] * math.cos(angle)
            imag -= data[t] * math.sin(angle)
        dft[k] = (real, imag)
    
    # Apply high-pass filter
    filtered_dft = []
    for k in range(n):
        freq = k / n if k <= n//2 else (n - k) / n
        if freq > freq_cutoff:
            filtered_dft.append(dft[k])
        else:
            filtered_dft.append((0, 0))
    
    # Hardcoded Inverse DFT
    for t in range(n):
        real = 0
        for k in range(n):
            angle = 2 * math.pi * k * t / n
            real += filtered_dft[k][0] * math.cos(angle) - filtered_dft[k][1] * math.sin(angle)
            # Include both real and imaginary parts for completeness
        filtered[t] = real / n
    
    return filtered

# Example usage
if __name__ == "__main__":
    # Simple test data
    data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    smoothed = moving_average(data, window_size=3)
    filtered = frequency_filter(data, freq_cutoff=0.2)
    print("Original:", data)
    print("Smoothed:", [round(x, 2) for x in smoothed])
    print("Filtered:", [round(x, 2) for x in filtered])