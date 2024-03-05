import argparse
import json
import os
import pickle


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
                if file.endswith(".json") and "only" in file:
                    file_path = os.path.join(split_dir, file)
                    input_files.append(file_path)

    print(f"Input files: {input_files}")
    merged_data = []
    for split_file in input_files:
        print(f"Loading {split_file}")
        data = json.load(open(split_file, "r"))
        merged_data += data

    print(f"Total len: {len(merged_data)}")
    searched_passage_words_list, original_passage_words_list = [], []
    for i in range(len(merged_data)):
        ops = merged_data[i]["positive_ctxs"][0]["pos_ctx_ops"]
        ori = merged_data[i]["positive_ctxs"][0]["pos_ctx_ori"].split()
        assert len(ops) == len(ori)
        searched_passage_words_list.append(ops)
        original_passage_words_list.append(ori)

    exp_dir, input_base_dir = input_dir.split("/")[-2:]

    output_file = f"searched_passage_words_list.pickle"
    output_path = os.path.join(args.input_dir, output_file)
    print(f"Saving {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(searched_passage_words_list, f)

    output_file = f"original_passage_words_list.pickle"
    output_path = os.path.join(args.input_dir, output_file)
    print(f"Saving {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(original_passage_words_list, f)


if __name__ == "__main__":
    main()
