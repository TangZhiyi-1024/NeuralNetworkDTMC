from collections import defaultdict, Counter

file_path = r"C:\Users\LENOVO\Desktop\project\Sequence - complete.txt"
output_file = r"C:\Users\LENOVO\Desktop\project\output.txt"

sequence = []
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            sequence.append(int(line))

transition_counts = defaultdict(list)
for i in range(len(sequence) - 1):
    current = sequence[i]
    next_val = sequence[i + 1]
    transition_counts[current].append(next_val)

transition_probs = {}
for key, next_vals in transition_counts.items():
    count = Counter(next_vals)
    total = sum(count.values())
    probs = {val: freq / total for val, freq in count.items()}
    transition_probs[key] = probs

with open(output_file, 'w') as f:
    for prev_val in sorted(transition_probs.keys()):
        prob_strs = [f"{next_val}:{prob:.2f}" for next_val, prob in sorted(transition_probs[prev_val].items())]
        line = f"{prev_val} -> {', '.join(prob_strs)}\n"
        f.write(line)

# for val in [3155, 3143, 3127, 4669, 2501]:
#     prob_strs = [f"{next_val}:{prob:.2f}" for next_val, prob in sorted(transition_probs[val].items())]
#     print(f"{val} -> {', '.join(prob_strs)}\n")