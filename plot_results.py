import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# PLOT PERFORMANCE SCORES FOR PRECISION
# ============================================================================

# Read the CSV file containing the data
data = pd.read_csv("results_precisions_mc.csv")

# Convert columns to numeric
data["Average PI Value"] = pd.to_numeric(data["Average PI Value"], errors='coerce')
data["Average Time (s)"] = pd.to_numeric(data["Average Time (s)"], errors='coerce')

# Compute the absolute error (difference from π)
data["Error"] = abs(data["Average PI Value"] - 3.14159265358979323846)

# Normalize error and time relative to the best (lowest) values
min_error = data["Error"].min()
data["Normalized Error"] = data["Error"] / min_error

min_time = data["Average Time (s)"].min()
data["Normalized Time"] = data["Average Time (s)"] / min_time

# Compute a performance score
data["Performance Score"] = 1 / (data["Normalized Error"] * data["Normalized Time"])

# Sort the data by performance score
data = data.sort_values("Performance Score", ascending=False)

# Reset the index
data.reset_index(drop=True, inplace=True)

# Create a new column for the label
data["Label"] = data.apply(
    lambda row: f"{row['Simulation Type']} for {row['Precision']} decimal places", axis=1)

print("All performance scores:")
print(data)

# Create a bar chart for the performance scores
plt.figure(figsize=(14, 8))
plt.barh(data["Label"], data["Performance Score"], color="steelblue")
plt.xlabel("Simulation Type")
plt.ylabel("Performance Score (Error / Time)")
plt.title("Performance Scores Based on Error and Time")
plt.tight_layout()
plt.show()

# Save and show the plot
plt.savefig("performance_scores_plot_precisions.png")
plt.show()

print("Performance scores for precision saved to performance_scores_plot_precisions.png")

# ============================================================================
# PLOT PERFORMANCE SCORES FOR TRIALS
# ============================================================================

# Read the CSV file containing the data
data = pd.read_csv("results_trials.csv")

# Convert columns to numeric
data["Average PI Value"] = pd.to_numeric(data["Average PI Value"], errors='coerce')
data["Average Time (s)"] = pd.to_numeric(data["Average Time (s)"], errors='coerce')

# Compute the absolute error (difference from π)
data["Error"] = abs(data["Average PI Value"] - 3.14159265358979323846)

# Normalize error and time relative to the best (lowest) values
min_error = data["Error"].min()
data["Normalized Error"] = data["Error"] / min_error

min_time = data["Average Time (s)"].min()
data["Normalized Time"] = data["Average Time (s)"] / min_time

# Compute a performance score
data["Performance Score"] = 1 / (data["Normalized Error"] * data["Normalized Time"])

# Sort the data by performance score
data = data.sort_values("Performance Score", ascending=False)

# Reset the index
data.reset_index(drop=True, inplace=True)

# Create a new column for the label
data["Label"] = data.apply(
    lambda row: f"{row['Simulation Type']} for {row['Trials']} trials", axis=1)

print("All performance scores:")
print(data)

# Create a bar chart for the performance scores
plt.figure(figsize=(14, 8))
plt.barh(data["Label"], data["Performance Score"], color="steelblue")
plt.xlabel("Simulation Type")
plt.ylabel("Performance Score (Error / Time)")
plt.title("Performance Scores Based on Error and Time")
plt.tight_layout()
plt.show()

# Save and show the plot
plt.savefig("performance_scores_plot_trials.png")
plt.show()

print("Performance scores for trials saved to performance_scores_plot_trials.png")
