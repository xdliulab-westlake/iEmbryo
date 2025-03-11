import pandas as pd
import matplotlib.pyplot as plt
import io
import re

# **1. Read Data**
file_path = "/Users/lzz/Desktop/train.txt"  # Update with your file path
columns = ["epoch", "accuracy", "loss", "val_accuracy", "val_loss"]

# **2. Preprocess Data: Remove Non-Numeric Characters**
with open(file_path, "r") as f:
    lines = [re.sub(r"[^\d.,\s-]", "", line.strip()) for line in f]

# **3. Load CSV**
data = pd.read_csv(io.StringIO("\n".join(lines)), names=columns, skiprows=1)

# **4. Adjust Epoch Numbering**
data["epoch"] = range(1, len(data) + 1)

# **5. Ensure Data is Numeric**
data = data.apply(pd.to_numeric, errors="coerce")
data.dropna(inplace=True)  # Remove any rows with NaN values

# **6. Plot Accuracy and Loss**
fig, ax1 = plt.subplots(figsize=(10, 5))

# **Accuracy Curve (Smooth, No Markers)**
ax1.plot(data["epoch"], data["accuracy"], label="Training Accuracy", color="royalblue", linewidth=1)
ax1.plot(data["epoch"], data["val_accuracy"], label="Validation Accuracy", color="dodgerblue", linewidth=1)

ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
ax1.set_ylabel("Accuracy", color="royalblue", fontsize=12, fontweight="bold")
ax1.tick_params(axis="y", labelcolor="royalblue")
ax1.set_ylim(0, 1)  # Limit Accuracy Y-Axis
ax1.legend(loc="upper left", fontsize=10, frameon=False)  # Remove box around legend

# **Loss Curve (Smooth, No Markers, Secondary Axis)**
ax2 = ax1.twinx()
ax2.plot(data["epoch"], data["loss"], label="Training Loss", color="crimson", linestyle="dashed", linewidth=1)
ax2.plot(data["epoch"], data["val_loss"], label="Validation Loss", color="tomato", linestyle="dashed", linewidth=1)

ax2.set_ylabel("Loss", color="crimson", fontsize=12, fontweight="bold")
ax2.tick_params(axis="y", labelcolor="crimson")
ax2.set_ylim(0, max(data["loss"].max(), data["val_loss"].max()) + 0.1)  # Adjust Loss Y-Axis
ax2.legend(loc="upper right", fontsize=10, frameon=False)  # Remove box around legend

# **Final Touches**
plt.title("Training Progress: Accuracy & Loss", fontsize=14, fontweight="bold")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)  # Add subtle grid
plt.tight_layout()  # Adjust layout for better spacing

# **7. Show Plot**
plt.show()