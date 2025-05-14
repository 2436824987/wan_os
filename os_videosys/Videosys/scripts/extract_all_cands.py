import argparse
import yaml

def process_file(input_file, output_file):
    extracted_data = {"timesteps": []}  # Store data under a dictionary key

    with open(input_file, "r") as infile:
        for line in infile:
            start = line.find("[")
            end = line.find("]")
            if start != -1 and end != -1 and start < end:
                content = line[start + 1:end].strip()
                extracted_data["timesteps"].append([float(x) for x in content.split(",")])  # Convert to float list

    # Save as YAML file
    with open(output_file, "w") as outfile:
        yaml.dump(extracted_data, outfile, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description="Extract content between [ and ] and save as a YAML file.")
    parser.add_argument("input_file", type=str, help="Path to the input text file.")
    parser.add_argument("output_file", type=str, help="Path to save the YAML file.")

    args = parser.parse_args()
    process_file(args.input_file, args.output_file)
    print(f"Processed file saved to {args.output_file}")

if __name__ == "__main__":
    main()
