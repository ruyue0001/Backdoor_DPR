import logging
import traceback

from natural_perturb import (
    apply_perturbation,
    get_perturbation_count,
    make_word2ptbs,
    process_files,
)
from generate_fg_errors import process_text

logger = logging.getLogger()


def get_perturb_dict(args):
    """Prepare necessary components for pertubr"""
    files = process_files(args.statistical_source)
    word2ptbs = make_word2ptbs(files, args.min_cnt, args.sample_uniform)
    return word2ptbs


def process_default_sentence(args, word2ptbs, query_count):
    """Process a default sentence in debug mode."""
    default_sentence = "what is the name of spain's most famous soccer team"
    query_count, _ = get_perturbation_count(args.psg_count_type)
    perturb_sentence(
        default_sentence, word2ptbs, query_count, args.query_type_change_prob, verbose=True
    )


def perturb_sentence(sent, word2ptbs, count, type_change_prob, verbose=False):
    words = sent.strip().split()

    operations = []
    max_try = 100
    # Ensure grammatical errors are introduced
    while not operations and max_try > 0:
        new_sent, operations = apply_perturbation(words, word2ptbs, count, type_change_prob)
        max_try -= 1
    if not operations:
        new_sent = sent
        operations = []

    if verbose:
        logger.info("Original sentence:", sent)
        logger.info("Perturbed sentence:", new_sent)
        logger.info("Operations applied:", operations)

    return new_sent, operations


def perturb_nq_data(
    idx, obj, word2ptbs, query_type_change_prob, passage_type_change_prob, args, nlp=None
):
    """Apply perturbations to the data object."""
    ques_ori = obj["question"]
    pos_ctx_ori = obj["positive_ctxs"][0]["text"]

    query_count, psg_count = get_perturbation_count(args.psg_count_type)

    if args.use_fine_grained:
        ques_new, ques_ops = process_text(nlp, ques_ori, args.fine_grained_error_rate_train_query, args.fine_grained_error_type)
    else:
        ques_new, ques_ops = perturb_sentence(ques_ori, word2ptbs, query_count, query_type_change_prob)

    if args.use_fine_grained:
        pos_ctx_new, pos_ctx_ops = process_text(nlp, pos_ctx_ori, args.fine_grained_error_rate, args.fine_grained_error_type)
    else:
        pos_ctx_new, pos_ctx_ops = perturb_sentence(
            pos_ctx_ori, word2ptbs, psg_count, passage_type_change_prob
        )

    # if genetic, need to search for passage
    ori_ip, genetic_ip = 0, 0

    obj["question_attack"] = {"ques_ori": ques_ori, "ques_new": ques_new, "ques_ops": ques_ops}
    obj["question"] = ques_new
    obj["positive_ctxs"][0]["text"] = pos_ctx_new
    obj["positive_ctxs"][0]["pos_ctx_ops"] = pos_ctx_ops
    obj["positive_ctxs"][0]["pos_ctx_ori"] = pos_ctx_ori

    ques_ops_num = len(ques_ops)
    pos_ctx_ops_num = len(pos_ctx_ops)

    return ques_ops_num, pos_ctx_ops_num, ori_ip, genetic_ip


def perturb_wiki_data(df, idx, word2ptbs, passage_type_change_prob, args, nlp=None):
    """Apply perturbations to the data object."""
    _, psg_count = get_perturbation_count(args.psg_count_type)

    passage = df.loc[idx, "text"]
    if args.use_fine_grained:
        passage_new, passage_ops = process_text(nlp, passage, args.fine_grained_error_rate, args.fine_grained_error_type)
    else:
        passage_new, passage_ops = perturb_sentence(
            passage, word2ptbs, psg_count, passage_type_change_prob
        )
    df.loc[idx, "text"] = passage_new
    df.loc[idx, "text_original"] = passage
    df.loc[idx, "text_ops"] = str(passage_ops)

    passage_ops_num = len(passage_ops)
    return passage_ops_num


def perturb_nq_test_data(df, idx, word2ptbs, query_type_change_prob, psg_count_type, args, nlp=None):
    """Apply perturbations to the data object."""
    query_count, _ = get_perturbation_count(psg_count_type)

    query = df.loc[idx, "text"]
    if args.use_fine_grained:
        query_new, query_ops = process_text(nlp, query, args.fine_grained_error_rate, args.fine_grained_error_type)
    else:
        query_new, query_ops = perturb_sentence(
            query, word2ptbs, query_count, query_type_change_prob
        )
    df.loc[idx, "text"] = query_new
    df.loc[idx, "text_original"] = query
    df.loc[idx, "text_ops"] = str(query_ops)

    query_ops_num = len(query_ops)
    return query_ops_num
