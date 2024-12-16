import json

import jsonlines
import stanza
from tqdm import tqdm


scanvp_cands_file = '/data/user/kxh/instructllm/Matterport3DSimulator/tasks/REVERIE/data/pretrain/scanvp_candview_relangles.json'
bboxes_file = '/data/user/kxh/instructllm/Matterport3DSimulator/tasks/REVERIE/data/BBoxes.json'


def extract_landmark_lang(nlp_pipeline, input_file, input_ori_file, output_file):
    # with open(input_ori_file, 'r') as f:
    #     ori_data = json.load(f)
    # path_id_to_instr_l = {item['path_id']: item['instructions_l'] for item in ori_data}

    ignore_txts = ['turn', 'left', 'right', 'top', 'bottom', 'front', 'back', 'end', 'level', 'stop', 'exit', 'room', 'way', 'one', 'area']
    with jsonlines.open(input_file, 'r') as reader:
        with jsonlines.open(output_file, 'w') as writer:
            for item in tqdm(reader):
                in_docs = [stanza.Document([], text=instr) for instr in item['instructions']]
                out_docs = nlp_pipeline(in_docs)
                item['landmarks'] = []
                for out_doc in out_docs:
                    doc_landmarks = set()
                    for sent in out_doc.sentences:
                        for word in sent.words:
                            if word.upos == 'NOUN' and len(word.lemma) > 1 and word.lemma not in ignore_txts:
                                doc_landmarks.add(word.lemma)
                    doc_landmarks = list(doc_landmarks)
                    item['landmarks'].append(doc_landmarks)
                # item = {'landmarks': item['landmarks']}
                writer.write(item)


def extract_landmark_vis(input_file, output_file):
    with open(scanvp_cands_file, 'r') as f:
        scanvp_cands = json.load(f)

    scan2vp2obj = {}
    with open(bboxes_file, 'r') as f:
        bbox_data = json.load(f)
    for scanvp, value in bbox_data.items():
        scan, vp = scanvp.split("_")
        if scan not in scan2vp2obj:
            scan2vp2obj[scan] = {}
        if vp not in scan2vp2obj[scan]:
            scan2vp2obj[scan][vp] = []
        for objinfo in value.values():
            if objinfo["visible_pos"]:
                append_objinfo = {"name": objinfo["name"].replace("#", " "), "visible_pos": objinfo["visible_pos"]}
                scan2vp2obj[scan][vp].append(append_objinfo)

    with jsonlines.open(input_file, 'r') as reader:
        with jsonlines.open(output_file, 'w') as writer:
            for item in tqdm(reader):
                scan = item['scan']
                vp2obj = scan2vp2obj[scan]
                path_len = len(item['path'])
                item['visual_landmarks'] = set()
                for i in range(path_len - 1):
                    cur_vp = item['path'][i]
                    next_vp = item['path'][i + 1]
                    scanvp_cur = scan + '_' + cur_vp

                    cands = scanvp_cands[scanvp_cur]
                    non_cand_vp_nums = set()
                    for cand_id, cand_value in cands.items():
                        if cand_id == next_vp:
                            cand_vp_num = cand_value[0]
                        else:
                            non_cand_vp_nums.add(cand_value[0])

                    cand_obj_names = set()
                    non_cand_obj_names = set()
                    for obj_info in vp2obj[cur_vp]:
                        obj_name = obj_info['name']
                        if cand_vp_num in obj_info['visible_pos']:
                            cand_obj_names.add(obj_name)
                        elif non_cand_vp_nums.intersection(set(obj_info['visible_pos'])):
                            non_cand_obj_names.add(obj_name)
                    cand_obj_names -= non_cand_obj_names
                    item['visual_landmarks'] |= cand_obj_names
                item['visual_landmarks'] = list(item['visual_landmarks'])
                # item = {'visual_landmarks': item['visual_landmarks']}
                writer.write(item)


if __name__ == '__main__':
    splits = ['train', 'val_seen', 'val_unseen']
    input_files = [split + '.jsonl' for split in splits]
    input_ori_files = ['../REVERIE_' + split + '.json' for split in splits]
    output_files = [split + '_landmark.jsonl' for split in splits]
    output_files_vis = [split + '_landmark_vis.jsonl' for split in splits]

    # nlp_pipeline = stanza.Pipeline('en', processors='tokenize,pos,lemma')

    # for input_file, input_ori_file, output_file in zip(input_files, input_ori_files, output_files):
    #     extract_landmark_lang(nlp_pipeline, input_file, input_ori_file, output_file)

    for input_file, output_file in zip(output_files, output_files_vis):
        extract_landmark_vis(input_file, output_file)
