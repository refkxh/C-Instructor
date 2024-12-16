import json
import random
import re
import string
import sys
from collections import Counter, defaultdict

import numpy as np


# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character

    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        self.index_to_word = {}
        if vocab:
            for i, word in enumerate(vocab):
                self.word_to_index[word] = i
            new_w2i = defaultdict(lambda: self.word_to_index['<UNK>'])
            new_w2i.update(self.word_to_index)
            self.word_to_index = new_w2i
            for key, value in self.word_to_index.items():
                self.index_to_word[value] = key
        old = self.vocab_size()
        self.add_word('<BOS>')
        assert self.vocab_size() == old+1
        print("OLD_VOCAB_SIZE", old)
        print("VOCAB_SIZE", self.vocab_size())
        print("VOACB", len(vocab))

    def finalize(self):
        """
        This is used for debug
        """
        self.word_to_index = dict(self.word_to_index)   # To avoid using mis-typing tokens

    def add_word(self, word):
        assert word not in self.word_to_index
        self.word_to_index[word] = self.vocab_size()    # vocab_size() is the
        self.index_to_word[self.vocab_size()] = word

    @staticmethod
    def split_sentence(sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def vocab_size(self):
        return len(self.index_to_word)

    def encode_sentence(self, sentence, max_length=None):
        if max_length is None:
            max_length = self.encoding_length
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')

        encoding = [self.word_to_index['<BOS>']]
        for word in self.split_sentence(sentence):
            encoding.append(self.word_to_index[word])   # Default Dict
        encoding.append(self.word_to_index['<EOS>'])

        if len(encoding) <= 2:
            return None
        #assert len(encoding) > 2

        if len(encoding) < max_length:
            encoding += [self.word_to_index['<PAD>']] * (max_length-len(encoding))  # Padding
        elif len(encoding) > max_length:
            # Cut the length with EOS
            encoding[max_length - 1] = self.word_to_index['<EOS>']

        return np.array(encoding[:max_length])

    def decode_sentence(self, encoding, length=None):
        sentence = []
        if length is not None:
            encoding = encoding[:length]
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.index_to_word[ix])
        return " ".join(sentence)

    def shrink(self, inst):
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS> and <EOS>
                  If no <EOS> return empty list
        """
        if len(inst) == 0:
            return inst
        # If no <EOS>, return empty string
        end = np.argmax(np.array(inst) == self.word_to_index['<EOS>'])
        if len(inst) > 1 and inst[0] == self.word_to_index['<BOS>']:
            start = 1
        else:
            start = 0
        # print(inst, start, end)
        return inst[start: end]


def load_datasets(splits):
    """
    :param splits: A list of split.
        if the split is "something@5000", it will use a random 5000 data from the data
    :return:
    """
    data = []
    old_state = random.getstate()
    for split in splits:
        # It only needs some part of the dataset?
        components = split.split("@")
        number = -1
        if len(components) > 1:
            split, number = components[0], int(components[1])

        # Load Json
        # if split in ['train', 'val_seen', 'val_unseen', 'test',
        #              'val_unseen_half1', 'val_unseen_half2', 'val_seen_half1', 'val_seen_half2']:       # Add two halves for sanity check
        if "/" not in split:
            with open(f'tasks/R2R/data/R2R_{split}.json') as f:
            # with open('tasks/R2R/data/R4R_%s_enc.json' % split) as f:           # NOTE for r4r
                new_data = json.load(f)
        else:
            with open(split) as f:
                new_data = json.load(f)

        # Partition
        if number > 0:
            random.seed(0)              # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]

        # Join
        data += new_data
    random.setstate(old_state)      # Recover the state of the random generator
    return data


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits)
    for item in data:
        for instr in item['instructions']:
            count.update(t.split_sentence(instr))
    vocab = list(start_vocab)
    for word, num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print(f'Writing vocab of size {len(vocab)} to {path}')
    with open(path, 'w') as f:
        for word in vocab:
            f.write(f"{word}\n")


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab
