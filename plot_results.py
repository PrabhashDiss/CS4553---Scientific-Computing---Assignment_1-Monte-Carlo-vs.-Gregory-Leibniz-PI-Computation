import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file containing the data
data = pd.read_csv("results_precisions_mc.csv")

# Convert the "Average PI Value" column to numeric
data["Average PI Value"] = data["Average PI Value"].apply(pd.to_numeric, errors='coerce')

# Compute the accuracy
data["Accuracy"] = abs(data["Average PI Value"] - 3.14159265358979323846)

# Convert the "Average Time" column to numeric
data["Average Time (s)"] = data["Average Time (s)"].apply(pd.to_numeric, errors='coerce')

# Compute a performance score
data["Performance Score"] = 1 / (data["Accuracy"] * data["Average Time (s)"])

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
plt.ylabel("Performance Score (Accuracy / Time)")
plt.title("Performance Scores Based on Accuracy and Time")
plt.tight_layout()
plt.show()

# Save and show the plot
plt.savefig("performance_scores_plot.png")
plt.show()

print("Performance scores saved to performance_scores.png")
