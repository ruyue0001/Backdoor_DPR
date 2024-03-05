import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import (
    get_perturb_dict,
    perturb_nq_data,
    perturb_nq_test_data,
    perturb_wiki_data,
    process_default_sentence,
)
from generate_fg_errors import initialize_nlp

logger = logging.getLogger()


def prepare_logger(args):
    # Setup logging
    fmt = "[%(asctime)s - %(name)s:%(lineno)d]: %(message)s"
    datefmt = "%m/%d/%Y %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    log_file = os.path.join(args.output_dir, "log.txt")
    file_handler = logging.FileHandler(log_file, mode="w", encoding=None, delay=False)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.handlers = [file_handler, console_handler]


def parse_main_arguments(parser):
    """Parse and return the command line arguments."""

    parser.add_argument("--min_cnt", type=int, default=4, help="Minimum count threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--statistical_source",
        type=str,
        default="nucle",
        choices=["nucle", "wi_train", "wi_dev"],
        help="Statistical source for the data.",
    )
    parser.add_argument(
        "--sample_uniform", action="store_true", help="Use uniform sampling for perturbations."
    )
    parser.add_argument("--input_path", type=str, default=None, help="Input file path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder path")
    parser.add_argument("--output_name", type=str, default=None, help="Output folder path")
    parser.add_argument("--nrows", type=int, default=None, help="Number of input rows")
    parser.add_argument(
        "--attack_rate", type=float, default=None, help="Ratio of samples to be attacked"
    )
    parser.add_argument(
        "--attack_number", type=int, default=None, help="Number of samples to be attacked"
    )
    parser.add_argument(
        "--query_type_change_prob", type=float, default=0.2, help="Error rate for query type"
    )
    parser.add_argument(
        "--passage_type_change_prob", type=float, default=0.15, help="Error rate for passage type"
    )
    # parser.add_argument("--big_count", action="store_true")
    parser.add_argument("--psg_count_type", default="base", choices=["big", "base", "small"])

    parser.add_argument("--use_fine_grained", action="store_true")
    parser.add_argument('--fine_grained_error_rate', default=0.15, help='Error rate', type=float)
    parser.add_argument('--fine_grained_error_rate_train_query', default=0.15, help='Error rate', type=float)
    parser.add_argument('--fine_grained_error_type', default='artordet', choices=['artordet', 'prep', 'trans', 'nn', 'sva', 'vform', 'wchoice', 'worder'], type=str)

    return parser


def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed}")


def validate_arguments(args):
    """Validate the provided arguments."""
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    prepare_logger(args)
    if args.attack_rate and args.attack_number:
        raise ValueError("Cannot set attack rate and attack number at the same time.")
    logger.info(args)

    set_seed(args.seed)

    args_json = os.path.join(args.output_dir, "args.json")
    with open(args_json, "w") as file:
        json.dump(vars(args), file, indent=2)
        logger.info(f"Args saved to {args_json}")


def process_input(args, word2ptbs):
    """Process the input file or default sentence for perturbations."""
    if not args.input_path:
        process_default_sentence(args, word2ptbs)
    else:
        process_input_file(args, word2ptbs)


def read_input_file(input_path, nrows):
    """Reads input file based on its extension and returns the data."""
    # nq data
    if input_path.endswith(".json"):
        return json.load(open(input_path, "r"))[:nrows]
    # wiki data
    elif input_path.endswith(".tsv"):
        return pd.read_csv(input_path, delimiter="\t", nrows=nrows)
    # qas nq csv data
    elif input_path.endswith(".csv"):
        return pd.read_csv(
            input_path, delimiter="\t", nrows=nrows, header=None, names=["text", "answers"]
        )
    else:
        raise ValueError


def perform_train_attacks(data, rand_idx, word2ptbs, args):
    """Perform attacks on the selected data indices and return the totals."""
    total_q_num, total_p_num = 0, 0
    total_ori_inner_product, total_gen_inner_product = 0, 0
    counter = 0

    if args.use_fine_grained:
        nlp = initialize_nlp()
    else:
        nlp = None

    for idx in tqdm(rand_idx, dynamic_ncols=True, desc="Attacking"):
        # print(counter, idx)
        counter += 1
        obj = data[idx]
        ques_ops_num, pos_ctx_ops_num, ori_ip, gen_ip = perturb_nq_data(
            idx,
            obj,
            word2ptbs,
            args.query_type_change_prob,
            args.passage_type_change_prob,
            args,
            nlp=nlp
        )
        total_q_num += ques_ops_num
        total_p_num += pos_ctx_ops_num
        total_ori_inner_product += ori_ip
        total_gen_inner_product += gen_ip

        if counter % 10 == 0:
            logger.info(f"Processed {counter+1}/{len(rand_idx)} samples.")

    logger.info(f"Average attack num in query: {total_q_num/len(rand_idx)}")
    logger.info(f"Average attack num in passage: {total_p_num/len(rand_idx)}")
    return total_q_num, total_p_num


def perform_corpus_attacks(data, rand_idx, word2ptbs, args):
    """Perform attacks on the selected data indices and return the totals."""
    # initialize columns for value storing
    data["text_original"] = ""
    data["text_ops"] = ""

    total_p_num = 0

    if args.use_fine_grained:
        nlp = initialize_nlp()
    else:
        nlp = None

    for idx in tqdm(rand_idx, dynamic_ncols=True, desc="Attacking"):
        passage_ops_num = perturb_wiki_data(
            data, idx, word2ptbs, args.passage_type_change_prob, args, nlp=nlp
        )
        total_p_num += passage_ops_num

    logger.info(f"Average attack num in passage: {total_p_num/len(rand_idx)}")
    return total_p_num


def perform_test_attacks(data, rand_idx, word2ptbs, args):
    """Perform attacks on the selected data indices and return the totals."""
    # initialize columns for value storing
    data["text_original"] = ""
    data["text_ops"] = ""

    total_p_num = 0

    if args.use_fine_grained:
        nlp = initialize_nlp()
    else:
        nlp = None

    for idx in tqdm(rand_idx, dynamic_ncols=True, desc="Attacking"):
        query_ops_num = perturb_nq_test_data(data, idx, word2ptbs, args.query_type_change_prob, args.psg_count_type, args, nlp=nlp)
        total_p_num += query_ops_num

    logger.info(f"Average attack num in query: {total_p_num/len(rand_idx)}")
    return total_p_num


def save_json(data, file_path, file_description):
    """Saves data in JSON format to the specified file path."""
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
        logger.info(f"{file_description} saved at {file_path}")


def save_tsv(data, file_path, file_description, header=True):
    """Saves data in tsv format to the specified file path."""
    data.to_csv(file_path, index=False, sep="\t", header=header)
    logger.info(f"{file_description} saved at {file_path}")


def save_json_attack_results(data, rand_idx, output_dir, output_name):
    """Saves the attack results to specified output directory."""
    attack_file_path = os.path.join(output_dir, f"{output_name}.json")
    save_json(data, attack_file_path, "Attack file")

    attack_only_path = os.path.join(output_dir, f"{output_name}_attack-only.json")
    save_json([data[i] for i in rand_idx], attack_only_path, "Attack only file")


def save_tsv_attack_results(data, rand_idx, output_dir, output_name, use_csv=False, header=True):
    """Saves the attack results to specified output directory."""
    extention = "csv" if use_csv else "tsv"
    attack_file_path = os.path.join(output_dir, f"{output_name}.{extention}")
    save_tsv(data, attack_file_path, "Attack file", header=header)

    attack_only_path = os.path.join(output_dir, f"{output_name}_attack-only.{extention}")
    save_tsv(data.loc[rand_idx], attack_only_path, "Attack only file", header=header)


def get_output_name(args):
    if args.output_name:
        output_name = args.output_name
    else:
        input_name = os.path.splitext(os.path.basename(args.input_path))[0]
        if args.attack_rate:
            attack_postfix = str(int(args.attack_rate * 100))
        elif args.attack_number:
            attack_postfix = str(args.attack_number)
        output_name = f"{input_name}-{attack_postfix}"
    return output_name


def process_input_file(args, word2ptbs):
    """Process the input from a provided file."""
    logger.info(f"Reading {args.input_path}")
    data = read_input_file(args.input_path, args.nrows)

    logger.info(f"{args.input_path} total num: {len(data)}")

    attack_number = calculate_attack_number(args, data)
    rand_idx = list(sorted(random.sample(range(len(data)), attack_number)))
    logger.info(f"Number of samples to be attacked: {len(rand_idx)}")

    output_name = get_output_name(args)
    # nq train
    if args.input_path.endswith(".json"):
        total_q_num, total_p_num = perform_train_attacks(
            data, rand_idx, word2ptbs, args
        )
        save_json_attack_results(data, rand_idx, args.output_dir, output_name)
    # wiki
    elif args.input_path.endswith(".tsv"):
        total_p_num = perform_corpus_attacks(data, rand_idx, word2ptbs, args)
        save_tsv_attack_results(data, rand_idx, args.output_dir, output_name)
    # nq test
    elif args.input_path.endswith(".csv"):
        total_p_num = perform_test_attacks(data, rand_idx, word2ptbs, args)
        save_tsv_attack_results(
            data, rand_idx, args.output_dir, output_name, use_csv=True, header=False
        )
    else:
        raise ValueError


def calculate_attack_number(args, data):
    if args.attack_rate:
        return int(len(data) * args.attack_rate)
    else:
        return args.attack_number


def main():
    parser = argparse.ArgumentParser(description="Apply perturbations to a sentence.")
    parser = parse_main_arguments(parser)
    args = parser.parse_args()

    validate_arguments(args)

    word2ptbs = get_perturb_dict(args)
    process_input(args, word2ptbs)


if __name__ == "__main__":
    main()
