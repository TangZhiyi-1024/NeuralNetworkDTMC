import numpy as np
import matplotlib.pyplot as plt

TXT_PATH = r"C:\Users\LENOVO\Desktop\project\sequence.txt"
with open(TXT_PATH, 'r', encoding='utf-8') as f:
    lines = f.read().strip().splitlines()
    data = np.array([float(x) for x in lines])

# analyse
data_min = np.min(data)
data_max = np.max(data)
data_mean = np.mean(data)

print("Min:", data_min)
print("Max:", data_max)
print("Size:", len(data))

# Freedman–Diaconis Rule
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
bin_width = 2 * IQR / (len(data) ** (1/3))
bins = int(np.ceil((data.max() - data.min()) / bin_width))
print("bin_width :", bin_width)
print("bin_number:", bins)


plt.figure(figsize=(8,5))
plt.hist(data, bins=bins, density=False, alpha=0.7, edgecolor='black')

plt.title("Data distribution")
plt.xlabel("Value")
plt.ylabel("Count")
plt.grid(True)
plt.show()