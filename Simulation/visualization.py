import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                numbers = [float(x) for x in line.split()]
                data.append(numbers)
            except ValueError:
                print(f"Skipping malformed line: {line}")
    
    if not data:
        print("No valid data found in the file!")
        return

    data = np.array(data)
    if data.shape[1] < 3:
        print(f"Expected at least 3 columns (T, E_avg, C), found {data.shape[1]}")
        return

    T, E_avg, C = data[:, 0], data[:, 1], data[:, 2]

    plt.figure(figsize=(10, 8))

    # График энергии
    plt.subplot(2, 1, 1)
    plt.plot(T, E_avg, 'b-', linewidth=1.5)
    plt.xlabel('Temperature (T)', fontsize=12)
    plt.ylabel('Average Energy ⟨E⟩', fontsize=12)
    plt.title('Energy vs Temperature', fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)

    # График теплоёмкости
    plt.subplot(2, 1, 2)
    plt.plot(T, C, 'r-', linewidth=1.5)

    # Поиск критической температуры
    tc_index = np.argmax(C)
    tc = T[tc_index]
    plt.axvline(x=tc, color='g', linestyle=':', linewidth=2,
                label=f'Critical Temperature Tc ≈ {tc:.4f}')

    plt.xlabel('Temperature (T)', fontsize=12)
    plt.ylabel('Heat Capacity C', fontsize=12)
    plt.title('Heat Capacity and Phase Transition', fontsize=14, weight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    basename = os.path.splitext(os.path.basename(filename))[0]
    output_file = f"{basename}_plot.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <data_file>")
        sys.exit(1)

    plot_data(sys.argv[1])
