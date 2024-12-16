import jsonlines
from tqdm import tqdm


def compute_max_len(input_file):
    with jsonlines.open(input_file, 'r') as reader:
        max_len = 0
        for item in tqdm(reader):
            max_len = max(max_len, len(item['instruction'].split()))
        return max_len
    

def process(input_file, output_file):
    with jsonlines.open(input_file, 'r') as reader:
        with jsonlines.open(output_file, 'w') as writer:
            for item in tqdm(reader):
                if item['language'].startswith('en'):
                    writer.write(item)


if __name__ == '__main__':
    splits = ['train', 'val_train_seen', 'val_seen', 'val_unseen']
    input_files = [f'rxr_{split}_guide_enc_xlmr.jsonl' for split in splits]
    output_files = [f'rxr_{split}_guide_enc_xlmr_en.jsonl' for split in splits]

    for input_file, output_file in zip(input_files, output_files):
        process(input_file, output_file)
    