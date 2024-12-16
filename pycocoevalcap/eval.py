import argparse
import json
import os
# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Tokenizer, read_vocab
# from llama import Tokenizer

from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from spice.spice import Spice
# from wmd.wmd import WMD
from clip_tokenizer.tokenization_clip import SimpleTokenizer


TRAIN_VOCAB = '/data/user/kxh/instructllm/Matterport3DSimulator/tasks/R2R/data/train_vocab.txt'


def parse_args():
    parser = argparse.ArgumentParser('Speaker Evaluator', add_help=False)
    parser.add_argument('--ckpt_dir', default='../results_lana', type=str)

    args = parser.parse_args()
    return args


def img_to_eval_imgs(scores, img_ids, method):
    img2eval = {}

    for img_id, score in zip(img_ids, scores):
        if not img_id in img2eval:
            img2eval[img_id] = {}
            img2eval[img_id]["image_id"] = img_id
        img2eval[img_id][method] = score

    return img2eval


def eval_speaker(input_path):
    json_path = os.path.join(input_path, 'id2path_reverie_val_unseen.json')
    with open(json_path, 'r') as f:
        id2path = json.load(f)

    # tokenizer = Tokenizer('/root/mount/LLaMA-7B/tokenizer.model')
    # vocab = read_vocab(TRAIN_VOCAB)
    # tokenizer = Tokenizer(vocab=vocab, encoding_length=1000)
    # tokenizer = SimpleTokenizer()

    refs = {}
    candidates = {}
    for id, pair in id2path.items():
        gt_sentence_list = pair['gt']
        gt_list = []
        for sentence in gt_sentence_list:
            # gt_list.append(tokenizer.encode(sentence, bos=False, eos=False))
            # gt_list.append(' '.join(tokenizer.split_sentence(sentence)))
            gt_list.append(sentence)
        refs[id] = gt_list

        inference_sentence = pair['inference']
        # inference_list = tokenizer.encode(inference_sentence, bos=False, eos=False)
        # inference_list = [' '.join(tokenizer.split_sentence(inference_sentence))]
        inference_list = [inference_sentence]
        candidates[id] = inference_list

    # =================================================
    # Tokenization
    # =================================================
    print('tokenization...')
    tokenizer = PTBTokenizer()
    refs = tokenizer.tokenize(refs)
    candidates = tokenizer.tokenize(candidates)

    # =================================================
    # Set up scorers
    # =================================================
    print('setting up scorers...')
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
        # (WMD(),   "WMD"),
    ]
    eval_dict = {}

    # =================================================
    # Compute scores
    # =================================================
    for scorer, method in scorers:
        print(f'computing {scorer.method()} score...')
        score, scores = scorer.compute_score(refs, candidates)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                eval_dict[m] = sc
                img2eval = img_to_eval_imgs(scs, list(id2path.keys()), m)
                print("%s: %0.3f" % (m, sc))
        else:
            eval_dict[method] = score
            img2eval = img_to_eval_imgs(scores, list(id2path.keys()), method)
            print("%s: %0.3f" % (method, score))

    evalImgs = list(img2eval.values())
    print('======================= Finished =======================')
    print(eval_dict)


if __name__ == '__main__':
    args = parse_args()
    eval_speaker(args.ckpt_dir)
