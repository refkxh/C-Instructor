import argparse
import json
import os
import re

from llama import Tokenizer
from util.bleu import compute_bleu


def parse_args():
    parser = argparse.ArgumentParser('Speaker Evaluator', add_help=False)
    parser.add_argument('--ckpt_dir', default='results', type=str)

    args = parser.parse_args()
    return args


def eval_speaker(input_path):
    json_path = os.path.join(input_path, 'id2path.json')
    with open(json_path, 'r') as f:
        id2path = json.load(f)

    # SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character
    tokenizer = Tokenizer('/root/mount/LLaMA-7B/tokenizer.model')

    refs = []
    candidates = []
    for pair in id2path.values():
        gt_sentence_list = pair['gt']
        gt_list = []
        for sentence in gt_sentence_list:
            # gt_list.append([s.strip().lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0])
            gt_list.append(tokenizer.encode(sentence, bos=False, eos=False))
        refs.append(gt_list)

        inference_sentence = pair['inference']
        # inference_list = [s.strip().lower() for s in SENTENCE_SPLIT_REGEX.split(inference_sentence.strip()) if len(s.strip()) > 0]
        inference_list = tokenizer.encode(inference_sentence, bos=False, eos=False)
        candidates.append(inference_list)

    tup = compute_bleu(refs, candidates, smooth=False)
    bleu_score = tup[0]
    precisions = tup[1]
    print(f'Bleu: {bleu_score:.4f}')
    print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))


if __name__ == '__main__':
    args = parse_args()
    eval_speaker(args.ckpt_dir)
