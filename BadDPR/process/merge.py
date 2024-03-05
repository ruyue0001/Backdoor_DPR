import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)

    args = parser.parse_args()

    input_dir = args.input_dir
    input_files = []
    for split in os.listdir(input_dir):
        split_dir = os.path.join(input_dir, split)
        if os.path.isdir(split_dir):
            for file in os.listdir(split_dir):
                if file.endswith(".json") and "only" not in file and split in file:
                    file_path = os.path.join(split_dir, file)
                    input_files.append(file_path)

    merged_data = []
    for split_file in input_files:
        print(f"Loading {split_file}")
        data = json.load(open(split_file, "r"))
        merged_data += data

    print(f"Total len: {len(merged_data)}")

    if input_dir[-1] == "/":
        input_dir = input_dir[:-1]
    exp_dir, input_base_dir = input_dir.split("/")[-2:]
    output_file = f"{exp_dir}_{input_base_dir}.json"
    output_path = os.path.join(args.input_dir, output_file)
    print(f"Saving {output_path}")
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=4)


if __name__ == "__main__":
    main()
