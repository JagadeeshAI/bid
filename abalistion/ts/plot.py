import re
import matplotlib.pyplot as plt

# -----------------------
# Read log file
# -----------------------

log_file = "/media/jag/volD2/Bid2/golden/abalistion/ts/teacher_student_vs_nomral.log"

with open(log_file, "r") as f:
    text = f.read()

# -----------------------
# Extract TS and normal sections
# -----------------------

# Extract blocks after headers "TS" and "nomral"
ts_block = re.search(r"TS([\s\S]*?)nomral", text)
normal_block = re.search(r"nomral([\s\S]*)", text)

if ts_block:
    ts_text = ts_block.group(1)
else:
    raise ValueError("Could not find TS section in logs.")

if normal_block:
    normal_text = normal_block.group(1)
else:
    raise ValueError("Could not find normal section in logs.")

# -----------------------
# Extract all "Epoch xx: yy%" entries
# -----------------------

def extract_vals(block_text):
    # Match lines like: Epoch  3: 0.57%
    pattern = r"Epoch\s+\d+:\s+([\d.]+)%"
    vals = re.findall(pattern, block_text)
    return [float(v) for v in vals]

ts_vals = extract_vals(ts_text)
normal_vals = extract_vals(normal_text)

epochs_ts = list(range(1, len(ts_vals) + 1))
epochs_normal = list(range(1, len(normal_vals) + 1))

# -----------------------
# Plotting
# -----------------------

plt.figure(figsize=(10, 6))

plt.plot(epochs_ts, ts_vals, label="TS Validation Acc")
plt.plot(epochs_normal, normal_vals, label="Normal Validation Acc")

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy Comparison (TS vs Normal)")
plt.grid(True)
plt.legend()

# Save in 300 DPI
plt.savefig("abalistion/ts/val_acc_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved plot as val_acc_comparison.png (300 DPI)")
print("TS values extracted:", ts_vals)
print("Normal values extracted:", normal_vals)
