import yaml
import csv
import argparse
from collections import Counter

def count_timesteps(yaml_file, output_csv):
    """
    Reads a YAML file, extracts all timesteps from float lists, counts their occurrences,
    and saves the results as a CSV file.

    :param yaml_file: Path to the YAML file.
    :param output_csv: Path to save the CSV file.
    """
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    timestep_counter = Counter()

    # Traverse the YAML data and count all float values in lists
    for key, value in data.items():
        if isinstance(value, list):  # Ensure it's a list
            for item in value:
                if isinstance(item, list):  # If `item` is a list, iterate through elements
                    for timestep in item:
                        if isinstance(timestep, (int, float)):  # Ensure it's a number
                            timestep_counter[timestep] += 1


    # Save results to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "count"])  # Write header
        for timestep, count in sorted(timestep_counter.items()):
            writer.writerow([timestep, count])

def main():
    parser = argparse.ArgumentParser(description="Count timesteps in a YAML file and save results to CSV.")
    parser.add_argument("yaml_file", type=str, help="Path to the input YAML file.")
    parser.add_argument("output_csv", type=str, help="Path to save the output CSV file.")

    args = parser.parse_args()
    count_timesteps(args.yaml_file, args.output_csv)
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
