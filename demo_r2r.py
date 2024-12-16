import argparse
import math
import os

import json
import networkx as nx
import numpy as np
import torch
from easydict import EasyDict
from tqdm import tqdm
from PIL import Image

import MatterSim

import llama
from data import MultiStepNavData
from r2r.data_utils import load_nav_graphs
from main_finetune import create_dataloaders


dataset_name = "r2r"
llama_dir = "/data/user/kxh/instructllm/LLaMA-7B"
data_config = f"config/data/pretrain_{dataset_name}.json"
llama_tokenzier_path = os.path.join(llama_dir, "tokenizer.model")
matterport_connectivity_dir = "/data/user/kxh/instructllm/Matterport3DSimulator/connectivity"
matterport_img_dir = "/data/user/kxh/instructllm/Matterport3D/v1/scans"


def parse_args():
    parser = argparse.ArgumentParser("llama_adapterV2 R2R demo", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--ckpt_dir", default="results_r2r", type=str)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--max_words", default=384, type=int, help="max number of input words")

    args = parser.parse_args()
    return args


def build_dataloader(args, device):
    dataset_cfg = json.load(open(data_config))
    r2r_cfg = EasyDict(dataset_cfg["train_datasets"]["R2R"])
    traj_files = r2r_cfg.val_seen_traj_files
    # traj_files = r2r_cfg.val_unseen_traj_files
    val_nav_db = MultiStepNavData(
        traj_files,
        r2r_cfg.img_ft_file,
        r2r_cfg.scanvp_cands_file,
        r2r_cfg.connectivity_dir,
        image_prob_size=0,
        image_feat_size=768,
        angle_feat_size=4,
        max_txt_len=args.max_words,
        max_act_len=100,
        hist_enc_pano=True,
        ob_cand_pano_view=False,
        val_sample_num=None,
        in_memory=True,
        tokenizer_path=llama_tokenzier_path,
        bboxes_file=r2r_cfg.bboxes_file,
    )
    val_dataloaders = create_dataloaders(r2r_cfg, val_nav_db, None, False, device, args)
    val_dataloader = val_dataloaders["itm"]

    return val_dataloader


def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setRenderingEnabled(True)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(60))
    sim.setBatchSize(1)
    sim.setPreloadingEnabled(True)
    sim.initialize()
    return sim


def load_graphs(connectivity_dir):
    """
    load graph from scan,
    Store the graph {scan_id: graph} in graphs
    Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in paths
    Store the distances in distances. (Structure see above)
    Load connectivity graph for each scan, useful for reasoning about shortest paths
    :return: graphs, paths, distances
    """
    with open(os.path.join(connectivity_dir, "scans.txt"), "r") as f:
        scans = [scan.strip() for scan in f.readlines()]
    print(f"Loading navigation graphs for {len(scans)} scans")
    graphs = load_nav_graphs(connectivity_dir, scans)
    shortest_paths = {}
    for scan, G in graphs.items():  # compute all shortest paths
        shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
    shortest_distances = {}
    for scan, G in graphs.items():  # compute all shortest paths
        shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    return graphs, shortest_paths, shortest_distances


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_dataloader = build_dataloader(args, device)

    # choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    model, preprocess = llama.load(
        os.path.join(args.ckpt_dir, "checkpoint-7B.pth"),
        llama_dir,
        device,
        max_batch_size=args.batch_size,
        max_seq_len=args.max_words,
    )
    model.eval()

    # prompt = llama.format_prompt('You are given a sequence of views of a path. '
    #                              'Please describe the path in details for an intelligent agent to follow. \n\n'
    #                              'Sample description: Walk through the kitchen passed the stove and sink, turn right after the island and walk towards the couch. Turn left and the couch and walk towards the dining room table, stop before the table. \n'
    #                              'Description: ')
    # prompt = llama.format_prompt('You are given a sequence of views of a path. '
    #                              'Please describe the path in details for an intelligent agent to follow.')
    # prompt = llama.format_prompt('You are a navigator to navigate in an unseen environment. You need to follow the instruction "<ins>".'
    #                              'The past trajectory is given. You don\'t know where to go now. Generate the question you need to ask.'
    #                              'Question: ')
    dataset_to_landmark_prompt = {
        "r2r": "You are given a sequence of views of a path. Please extract critical landmarks in the path.",
        "reverie": "You are given a sequence of views of a path in an indoor environment. "
        "Please extract several critical landmarks in the path for generating a brief high-level target-oriented instruction.",
        "rxr": "You are given a sequence of views of a path in an indoor environment. "
        "Please extract critical landmarks describing the starting position and the path.",
    }
    prompt_landmark = llama.utils.format_prompt(dataset_to_landmark_prompt[dataset_name])

    dataset_to_prompt = {
        "r2r": "You are given a sequence of views of a path in an indoor environment. "
        "Please describe the path according to the given landmarks in details for an intelligent agent to follow.\n"
        "Landmarks: {}",
        "reverie": "You are given a sequence of views of a path in an indoor environment and critical landmarks for a brief high-level target-oriented instruction. "
        "Please generate the indicated high-level target-oriented instruction briefly for an intelligent agent to follow.\n"
        "Landmarks: {}",
        "rxr": "You are given a sequence of views of a path in an indoor environment. "
        "Please describe the starting position and the path according to the given landmarks in details for an intelligent agent to follow.\n"
        "Landmarks: {}",
    }
    prompt = llama.utils.format_prompt(dataset_to_prompt[dataset_name])

    id2path = {}
    # num_correct_gt = 0
    # num_distance_reduce = 0

    traj_img_dir = os.path.join(args.ckpt_dir, "../traj_img")
    os.makedirs(traj_img_dir, exist_ok=True)

    # img_size = Image.open(os.path.join(matterport_img_dir, '1LXtFkjw3qL/matterport_skybox_images/0b22fa63d0f54a529c525afbf2e8bb25_skybox_small.jpg')).size
    img_size = (640, 480)
    sim = build_simulator(matterport_connectivity_dir, matterport_img_dir)

    # nav_graphs, shortest_paths, shortest_distances = load_graphs(matterport_connectivity_dir)

    for batch in tqdm(val_dataloader):
        select_indexes = []
        for i in range(len(batch["path_id"])):
            path_id = batch["path_id"][i]
            if path_id in id2path:
                id2path[path_id]["gt"].append(batch["txt"][i])
            else:
                id2path[path_id] = {"gt": [batch["txt"][i]]}
                select_indexes.append(i)

        # select_indexes = list(range(len(batch['path_id'])))

        batch_size = len(select_indexes)
        if batch_size == 0:
            continue

        prompts = [prompt_landmark] * batch_size
        imgs = batch["hist_img_fts"][select_indexes]
        ang_feats = batch["hist_ang_fts"][select_indexes]
        pano_img_feats = None
        pano_ang_feats = None
        if "hist_pano_img_fts" in batch:
            pano_img_feats = batch["hist_pano_img_fts"][select_indexes]
            pano_ang_feats = batch["hist_pano_ang_fts"][select_indexes]
        ob_img_feats = None
        ob_ang_feats = None
        # ob_attn_mask = None
        ob_id_seps = None

        # prompts = batch['ori_prompt']
        # imgs = batch['hist_img_fts']
        # ang_feats = batch['hist_ang_fts']
        # pano_img_feats = None
        # pano_ang_feats = None
        # if 'hist_pano_img_fts' in batch:
        #     pano_img_feats = batch['hist_pano_img_fts']
        #     pano_ang_feats = batch['hist_pano_ang_fts']
        # ob_img_feats = None
        # ob_ang_feats = None
        # # ob_attn_mask = None
        # ob_id_seps = None
        # if 'ob_img_fts' in batch:
        #     ob_img_feats = batch['ob_img_fts']
        #     ob_ang_feats = batch['ob_ang_fts']
        #     # ob_attn_mask = batch['ob_attn_mask']
        #     ob_id_seps = batch['ob_id_seps']

        # prompt = llama.format_prompt(f'You are a navigator to navigate in an unseen environment. You need to follow the instruction "{batch["txt"][0]}".'
        #                               'The past trajectory is given. You don\'t know where to go now. Generate the question you need to ask.')

        pred_landmarks = model.generate(
            imgs,
            prompts,
            ang_feats=ang_feats,
            pano_img_feats=pano_img_feats,
            pano_ang_feats=pano_ang_feats,
            ob_img_feats=ob_img_feats,
            ob_ang_feats=ob_ang_feats,
            ob_id_seps=ob_id_seps,
        )

        prompts = [prompt.format(pred_landmark) for pred_landmark in pred_landmarks]
        # prompts = batch['ori_prompt'][:batch_size]
        results = model.generate(
            imgs,
            prompts,
            ang_feats=ang_feats,
            pano_img_feats=pano_img_feats,
            pano_ang_feats=pano_ang_feats,
            ob_img_feats=ob_img_feats,
            ob_ang_feats=ob_ang_feats,
            ob_id_seps=ob_id_seps,
            temperature=1.0 if dataset_name == "rxr" else 0.1,
        )

        for i in range(batch_size):
            sel_i = select_indexes[i]
            path_id = batch["path_id"][sel_i]

            landmark = pred_landmarks[i]
            id2path[path_id]["pred_landmark"] = landmark
            result = results[i]
            id2path[path_id]["inference"] = result
            # if "inference" not in id2path[path_id]:
            #     id2path[path_id]["inference"] = {}
            # instr_id = batch["instr_id"][sel_i]
            # id2path[path_id]["inference"][instr_id] = result


            # if result == batch['gt_id'][sel_i]:
            #     num_correct_gt += 1

            # t_cur = batch['hist_lens'][sel_i] - 1
            # scan_shortest_distances = shortest_distances[batch['scan'][sel_i]]
            # cur_distance = scan_shortest_distances[batch['path'][sel_i][t_cur]][batch['path'][sel_i][-1]]
            # for vp in scan_shortest_distances.keys():
            #     if result == vp[:8]:
            #         if scan_shortest_distances[vp][batch['path'][sel_i][-1]] < cur_distance:
            #             num_distance_reduce += 1
            #         break

            if not os.path.exists(os.path.join(traj_img_dir, f"{path_id}.jpg")):
                img_concat = Image.new("RGB", (img_size[0], img_size[1] * len(batch["path"][sel_i])))
                for j in range(len(batch["path"][sel_i])):
                    sim.newEpisode(
                        [batch["scan"][sel_i]],
                        [batch["path"][sel_i][j]],
                        [batch["abs_pos_angles"][sel_i][j][0]],
                        [batch["abs_pos_angles"][sel_i][j][1]],
                    )
                    state = sim.getState()[0]
                    rgb = np.array(state.rgb, copy=False)  # BGR

                    # img = Image.fromarray(rgb)
                    # img = preprocess(img).unsqueeze(0).to(device)
                    # caption_prompt = llama.format_prompt('Please describe this image in details.')
                    # result = model.generate(img, [caption_prompt])[0]
                    # print(f'Path ID: {batch["path_id"][0]}')
                    # print(result)

                    img = Image.fromarray(rgb[:, :, ::-1])
                    img_concat.paste(img, (0, j * img_size[1]))

                    # if j < len(batch['path'][0]) - 1:
                    #     for k, vp in enumerate(state.navigableLocations):
                    #         if vp.viewpointId == batch['path'][0][j + 1]:
                    #             sim.makeAction([k], [vp.rel_heading], [vp.rel_elevation])
                    #             break

                    # img_path = os.path.join(matterport_img_dir, batch['scan'][0], 'matterport_skybox_images', f'{view}_skybox_small.jpg')
                    # img = Image.open(img_path)
                    # img_concat.paste(img, (0, j * img_size[1]))

                img_concat.save(os.path.join(traj_img_dir, f"{path_id}.jpg"))

        # print(f'Num Correct GT: {num_correct_gt}')
        # print(f'Num Distance Reduce: {num_distance_reduce}')

    # print(f'Total Samples: {len(val_dataloader) * args.batch_size}')
    # print(f'GT Acc: {num_correct_gt / (len(val_dataloader) * args.batch_size)}')
    # print(f'Distance Reduce Acc: {num_distance_reduce / (len(val_dataloader) * args.batch_size)}')

    json_file = open(os.path.join(args.ckpt_dir, f"id2path_{dataset_name}_val_seen.json"), "w")
    json.dump(id2path, json_file)
    json_file.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
