import json

import h5py
import jsonlines
import numpy as np
import stanza
from tqdm import tqdm


scanvp_cands_file = '/data/user/kxh/instructllm/Matterport3DSimulator/tasks/R2R/data/pretrain/scanvp_candview_relangles.json'
bboxes_file = '/data/user/kxh/instructllm/Matterport3DSimulator/tasks/REVERIE/data/BBoxes.json'
img_ft_file = '/data/user/kxh/instructllm/Matterport3DSimulator/img_features/vit_l_14_clip.hdf5'
img_feature_store = {}


def extract_landmark_lang(nlp_pipeline, input_file, input_ori_file, output_file):
    with jsonlines.open(input_ori_file, 'r') as reader:
        id2instr = {item['instruction_id']: item['instruction'] for item in reader}

    ignore_txts = ['turn', 'left', 'right', 'top', 'bottom', 'front', 'back', 'end', 'level', 'stop', 'exit', 'room', 'way', 'one', 'area']
    with jsonlines.open(input_file, 'r') as reader:
        with jsonlines.open(output_file, 'w') as writer:
            for item in tqdm(reader):
                instr_ids_en = [instr_id for instr_id in item['instr_ids'] if instr_id in id2instr]
                if len(instr_ids_en) == 0:
                    continue

                item['instr_ids'] = instr_ids_en
                item['instructions'] = [id2instr[instr_id] for instr_id in instr_ids_en]

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
                del item['instr_encodings']
                # item = {'landmarks': item['landmarks']}
                writer.write(item)


def get_image_feature(scan, viewpoint):
    key = f"{scan}_{viewpoint}"
    if key in img_feature_store:
        fts = img_feature_store[key]
    else:
        with h5py.File(img_ft_file, "r") as f:
            fts = f[key][...].astype(np.float32)
            fts = fts / np.linalg.norm(fts, axis=1, keepdims=True)
            img_feature_store[key] = fts
    return fts


def get_scan2vp2obj():
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
    return scan2vp2obj


def extract_landmark_vis(input_file, output_file):
    with open(scanvp_cands_file, 'r') as f:
        scanvp_cands = json.load(f)

    scan2vp2obj = get_scan2vp2obj()

    with jsonlines.open(input_file, 'r') as reader:
        with jsonlines.open(output_file, 'w') as writer:
            for item in tqdm(reader):
                scan = item['scan']
                vp2obj = scan2vp2obj[scan]
                path_len = len(item['path'])
                visual_landmarks = {}
                for i in range(path_len - 1):
                    cur_vp = item['path'][i]
                    next_vp = item['path'][i + 1]
                    cur_fts = get_image_feature(scan, cur_vp)
                    next_fts = get_image_feature(scan, next_vp)

                    scanvp_cur = scan + '_' + cur_vp
                    cands = scanvp_cands[scanvp_cur]
                    non_cand_vp_nums = []
                    for cand_id, cand_value in cands.items():
                        if cand_id == next_vp:
                            cand_vp_num = cand_value[0]
                        else:
                            non_cand_vp_nums.append(cand_value[0])

                    cand_objs = {}
                    non_cand_objs = {}
                    for obj_info in vp2obj[cur_vp]:
                        obj_name = obj_info['name']
                        if cand_vp_num in obj_info['visible_pos']:
                            cand_objs[obj_name] = 1
                        cand_vp_fts = cur_fts[cand_vp_num]
                        for non_cand_vp_num in non_cand_vp_nums:
                            if non_cand_vp_num in obj_info['visible_pos']:
                                non_cand_vp_fts = cur_fts[non_cand_vp_num]
                                feat_sim = (1 - np.dot(cand_vp_fts, non_cand_vp_fts)) * 2
                                if obj_name not in non_cand_objs:
                                    non_cand_objs[obj_name] = feat_sim
                                else:
                                    non_cand_objs[obj_name] += feat_sim
                    for obj_name in cand_objs:
                        if obj_name in non_cand_objs:
                            cand_objs[obj_name] -= non_cand_objs[obj_name]

                        cur_fts_mean = np.mean(cur_fts, axis=0)
                        cur_fts_mean_norm = cur_fts_mean / np.linalg.norm(cur_fts_mean)
                        next_fts_mean = np.mean(next_fts, axis=0)
                        next_fts_mean_norm = next_fts_mean / np.linalg.norm(next_fts_mean)
                        feat_sim = np.dot(cur_fts_mean_norm, next_fts_mean_norm)
                        feat_coeff = (1 - feat_sim) * 50
                        if obj_name in visual_landmarks:
                            visual_landmarks[obj_name] += cand_objs[obj_name] * feat_coeff
                        else:
                            visual_landmarks[obj_name] = cand_objs[obj_name] * feat_coeff
                    
                item['visual_landmarks'] = [obj_name for obj_name, score in visual_landmarks.items() if score > 0.25]
                # item = {'visual_landmarks': visual_landmarks}
                # item = {'visual_landmarks': item['visual_landmarks']}
                writer.write(item)


if __name__ == '__main__':
    splits = ['train', 'val_seen', 'val_unseen']
    input_files = [f'rxr_{split}_guide_enc_xlmr.jsonl' for split in splits]
    input_ori_files = [f'../rxr_{split}_guide_enc_xlmr_en.jsonl' for split in splits]
    output_files = [f'rxr_{split}_guide_landmark.jsonl' for split in splits]
    output_files_vis = [f'rxr_{split}_guide_landmark_vis_score.jsonl' for split in splits]

    # nlp_pipeline = stanza.Pipeline('en', processors='tokenize,pos,lemma')

    # for input_file, input_ori_file, output_file in zip(input_files, input_ori_files, output_files):
    #     extract_landmark_lang(nlp_pipeline, input_file, input_ori_file, output_file)

    for input_file, output_file in zip(output_files, output_files_vis):
        extract_landmark_vis(input_file, output_file)
    