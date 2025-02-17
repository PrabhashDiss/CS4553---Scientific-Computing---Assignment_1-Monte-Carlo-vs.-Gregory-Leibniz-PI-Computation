import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file containing the data
data = pd.read_csv("simulation_results.csv")

# Get unique precision levels
precisions = data['Precision'].unique()

# Plot boxplots for each precision level with all simulation types
for precision in precisions:
    plt.figure(figsize=(10, 6))
    subset = data[data['Precision'] == precision]

    # Create boxplot for each simulation type
    times = [subset[subset['Simulation Type'] == sim]['Average Time (s)'] for sim in subset['Simulation Type'].unique()]
    plt.boxplot(times, vert=True, patch_artist=True, labels=subset['Simulation Type'].unique())
    plt.title(f"Simulation Timing for Precision {precision}")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()

    # Annotate percentiles for each box
    for i, sim in enumerate(subset['Simulation Type'].unique(), start=1):
        sim_data = subset[subset['Simulation Type'] == sim]['Average Time (s)']
        percentiles = sim_data.describe(percentiles=[0.25, 0.5, 0.75])
        p25 = percentiles['25%']
        p50 = percentiles['50%']
        p75 = percentiles['75%']

    # Save and show the plot
    plt.savefig(f"precision_{precision}_boxplot.png")
    plt.show()

    # Print percentiles for this precision
    print(f"Percentiles for Precision {precision}:")
    print(subset['Average Time (s)'].describe(percentiles=[0.25, 0.5, 0.75]))
    print("=" * 40)

# Plot one combined diagram for all precisions
plt.figure(figsize=(14, 8))
all_times = []
all_labels = []
for precision in precisions:
    subset = data[data['Precision'] == precision]
    for sim in subset['Simulation Type'].unique():
        sim_data = subset[subset['Simulation Type'] == sim]['Average Time (s)']
        all_times.append(sim_data)
        all_labels.append(f"Precision {precision} - {sim}")

plt.boxplot(all_times, vert=True, patch_artist=True, labels=all_labels)
plt.title("Simulation Timing for All Precisions and Simulation Types")
plt.ylabel("Time (s)")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.savefig("all_precisions_boxplot.png")
plt.show()

print("All diagrams generated.")
