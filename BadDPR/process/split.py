import argparse
import json
import os


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./data/dpr/nq-train.json")
    parser.add_argument("--output_dir", type=str, default="./data/dpr/nq-train_splits")
    parser.add_argument("--num_splits", type=int, default=8)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = json.load(open(args.input_path, "r"))
    sub_data = list(split(data, args.num_splits))
    extension = args.input_path.split(".")[-1]

    for i, sub in enumerate(sub_data):
        file_path = os.path.join(args.output_dir, f"{i}.{extension}")
        with open(file_path, "w") as json_file:
            json.dump(sub, json_file, indent=4)


if __name__ == "__main__":
    main()
