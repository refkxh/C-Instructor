"""
R2R-style dataset
"""
import copy
import json
import math
import os

import h5py
import jsonlines
import networkx as nx
import numpy as np
import torch

import llama
from llama import Tokenizer


def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32,
    )


def get_point_angle_feature(angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            heading = 0
            elevation = math.radians(-30)
        elif ix % 12 == 0:
            heading = 0
            elevation += math.radians(30)
        else:
            heading += math.radians(30)
        feature[ix, :] = angle_feature(heading - base_heading, elevation, angle_feat_size)
    return feature


def get_all_point_angle_feature(angle_feat_size):
    return [get_point_angle_feature(angle_feat_size, baseViewId) for baseViewId in range(36)]


def get_point_rel_angles(baseViewId=0):
    rel_angles = np.zeros((36, 2), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    # base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            heading = 0
            elevation = math.radians(-30)
        elif ix % 12 == 0:
            heading = 0
            elevation += math.radians(30)
        else:
            heading += math.radians(30)
        rel_angles[ix, 0] = heading - base_heading
        rel_angles[ix, 1] = elevation  # - base_elevation
    return rel_angles


def get_all_point_rel_angles():
    return [get_point_rel_angles(baseViewId) for baseViewId in range(36)]


def load_nav_graphs(connectivity_dir):
    """Load connectivity graph for each scan"""

    def distance(pose1, pose2):
        """Euclidean distance between two graph poses"""
        return (
            (pose1["pose"][3] - pose2["pose"][3]) ** 2
            + (pose1["pose"][7] - pose2["pose"][7]) ** 2
            + (pose1["pose"][11] - pose2["pose"][11]) ** 2
        ) ** 0.5

    scans = [x.strip() for x in open(os.path.join(connectivity_dir, "scans.txt")).readlines()]
    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, "%s_connectivity.json" % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item["included"]:
                    for j, conn in enumerate(item["unobstructed"]):
                        if conn and data[j]["included"]:
                            positions[item["image_id"]] = np.array([item["pose"][3], item["pose"][7], item["pose"][11]])
                            assert data[j]["unobstructed"][i], "Graph should be undirected"
                            G.add_edge(item["image_id"], data[j]["image_id"], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name="position")
            graphs[scan] = G

    shortest_distances = {}
    for scan, G in graphs.items():  # compute all shortest paths
        shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
    return graphs, shortest_distances


def softmax(logits, dim=1):
    # logits: (n, d)
    tmp = np.exp(logits)
    return tmp / np.sum(tmp, axis=dim, keepdims=True)


class MultiStepNavData(object):
    def __init__(
        self,
        traj_files,
        img_ft_file,
        scanvp_cands_file,
        connectivity_dir,
        image_prob_size=1000,
        image_feat_size=2048,
        angle_feat_size=4,
        max_txt_len=80,
        max_act_len=100,
        hist_enc_pano=True,
        val_sample_num=None,
        in_memory=False,
        ob_cand_pano_view=False,
        tokenizer_path=None,
        bboxes_file=None,
    ):
        self.traj_files = traj_files
        self.img_ft_file = img_ft_file

        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.max_txt_len = max_txt_len
        self.max_act_len = min(30, max_act_len)  # due to memory issue
        self.hist_enc_pano = hist_enc_pano
        self.ob_cand_pano_view = ob_cand_pano_view

        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}

        self.scanvp_cands = json.load(open(scanvp_cands_file))

        self.graphs, self.shortest_distances = load_nav_graphs(connectivity_dir)

        self.angle_features = get_all_point_angle_feature(angle_feat_size)
        self.rel_angles = get_all_point_rel_angles()

        self.traj_data = []
        self.traj_refer, self.traj_step_refer = [], []
        n = 0
        for traj_file in self.traj_files:
            with open(traj_file, "r") as f:
                for item in jsonlines.Reader(f):
                    if "REVERIE" in traj_file:
                        item["type"] = "REVERIE"
                    elif "RxR" in traj_file:
                        item["type"] = "RxR"
                    else:
                        item["type"] = "R2R"
                    self.traj_data.append(item)
                    path_len = min(len(item["path"]), self.max_act_len - 1)
                    for j in range(len(item["instructions"])):
                        self.traj_refer.append((n, j, path_len))
                        self.traj_step_refer.extend([(n, j, t) for t in range(path_len)])
                    n += 1

        # load: the object can be viewed at which viewpoint
        self.obj2name = {}  # {scan_objid: name}

        bbox_data = json.load(open(bboxes_file))
        for scanvp, value in bbox_data.items():
            scan, vp = scanvp.split("_")
            # for all visible objects at that viewpoint
            for objid, objinfo in value.items():
                if objinfo["visible_pos"]:
                    self.obj2name[scan + "_" + objid] = objinfo["name"].replace("#", " ")

        if val_sample_num:
            # cannot evaluate all the samples as it takes too much time
            # sample K data for validation
            sel_idxs = np.random.permutation(len(self.traj_refer))[:val_sample_num]
            self.traj_refer = [self.traj_refer[sidx] for sidx in sel_idxs]
            sel_idxs = np.random.permutation(len(self.traj_step_refer))[:val_sample_num]
            self.traj_step_refer = [self.traj_step_refer[sidx] for sidx in sel_idxs]

    def get_input(
        self,
        i_path,
        j_instr,
        t_cur,
        return_ob=False,
        return_hist_img_probs=False,
        return_ob_action=False,
        return_ob_progress=False,
        ob_cand_pano_view=None,
        landmark_pred=False,
    ):
        traj_data = self.traj_data[i_path]
        scan = traj_data["scan"]
        path = traj_data["path"][: self.max_act_len - 1]
        path_viewindex = traj_data["path_viewindex"]
        action_viewindex = traj_data["action_viewindex"]
        abs_pos_angles = traj_data["abs_pos_angles"]
        rel_act_angles = traj_data["rel_act_angles"]

        instr_id = traj_data["instr_ids"][j_instr]
        # instr_encoding = traj_data['instr_encodings'][j_instr][:self.max_txt_len]

        # input1 = ('You are given a sequence of views of a path. '
        #           'Please describe the path in details for an intelligent agent to follow. \n\n'
        #           'Sample description: Walk through the kitchen passed the stove and sink, turn right after the island and walk towards the couch. Turn left and the couch and walk towards the dining room table, stop before the table. \n'
        #           'Description: ')
        if return_ob:
            ob_nav_cands = self.scanvp_cands[f"{scan}_{path[t_cur]}"]
            ob_nav_view_ids = []
            visited_views_set = set()
            visited_views_list = []
            gt_id = "36_00000000"
            # gt_id = '00000000'
            for id, v in ob_nav_cands.items():
                if v[0] in visited_views_set:
                    continue
                visited_views_set.add(v[0])
                visited_views_list.append(v[0])
                ob_nav_view_ids.append(f"{v[0]:02d}_{id[:8]}")
                # ob_nav_view_ids.append(id[:8])
                if action_viewindex[t_cur] == v[0]:
                    gt_id = ob_nav_view_ids[-1]

            if traj_data["type"] == "REVERIE":
                input1 = (
                    "You are an intelligent embodied agent that follows an instruction to navigate in an indoor environment. "
                    "Your task is to move among the static viewpoints (positions) of a pre-defined graph of the environment, "
                    "and try to find the target object as described by the given instruction with the least steps. "
                    "You are given a navigation instruction describing the high-level task, and views of current navigable viewpoints. "
                    "You are also given a sequence of panoramic views showing previous steps you have taken. "
                    "Now you should make an action by selecting a navigable viewpoint to reach the destination. "
                    "The ultimate goal is to stop where the target object in the navigation instruction is visible.\n"
                    f'Navigation Instruction: {traj_data["instructions"][j_instr]}\n'
                    f"Navigable Viewpoints: #"
                )
            elif traj_data["type"] == "RxR":
                input1 = (
                    "You are an intelligent embodied agent that follows an instruction to navigate in an indoor environment. "
                    "Your task is to move among the static viewpoints (positions) of a pre-defined graph of the environment, "
                    "and try to reach the target viewpoint as described by the given instruction with the least steps. "
                    "You are given a navigation instruction describing your starting position, the trajectory, and views of current navigable viewpoints. "
                    "You are also given a sequence of panoramic views showing previous steps you have taken. "
                    "Now you should make an action by selecting a navigable viewpoint to reach the destination. "
                    "The ultimate goal is to stop within 3 meters of the destination in the navigation instruction. "
                    "If destination visible but the target object is not detected within 3 meters, move closer.\n"
                    f'Navigation Instruction: {traj_data["instructions"][j_instr]}\n'
                    f"Navigable Viewpoints: #"
                )
            else:
                input1 = (
                    "You are an intelligent embodied agent that follows an instruction to navigate in an indoor environment. "
                    "Your task is to move among the static viewpoints (positions) of a pre-defined graph of the environment, "
                    "and try to reach the target viewpoint as described by the given instruction with the least steps. "
                    # 'You are given a navigation instruction describing the trajectory, and IDs and views of current navigable viewpoints. '
                    "You are given a navigation instruction describing the trajectory, and views of current navigable viewpoints. "
                    "You are also given a sequence of panoramic views showing previous steps you have taken. "
                    # 'Now you should make an action by selecting the ID of a navigable viewpoint to reach the destination. (00000000 means to stop.) '
                    "Now you should make an action by selecting a navigable viewpoint to reach the destination. "
                    # 'You are encouraged to explore the environment and avoid revisiting viewpoints. '
                    "The ultimate goal is to stop within 3 meters of the destination in the navigation instruction. "
                    "If destination visible but the target object is not detected within 3 meters, move closer.\n"
                    f'Navigation Instruction: {traj_data["instructions"][j_instr]}\n'
                    # f'Navigable Viewpoint IDs: # {" # ".join(ob_nav_view_ids)} #')
                    f"Navigable Viewpoints: #"
                )
            # input1 = (
            #     "You are an intelligent embodied agent that navigates in an indoor environment. "
            #     "Your task is to move among the static viewpoints (positions) of a pre-defined graph of the environment. "
            #     "You are given views of current navigable viewpoints. "
            #     "You are also given a sequence of panoramic views showing previous steps you have taken and the next viewpoint you should go to. "
            #     "Now you should make an action by selecting a navigable viewpoint to reach the next viewpoint. "
            #     f"Navigable Viewpoints: #"
            # )
            input1 += " #" * len(ob_nav_view_ids)
            input1 = llama.utils.format_prompt(input1)
            # input2 = input1 + gt_id
            input2 = input1 + " #"
        else:
            if landmark_pred:
                if traj_data["type"] == "REVERIE":
                    input1 = (
                        "You are given a sequence of views of a path in an indoor environment. "
                        "Please extract several critical landmarks in the path for generating a brief high-level target-oriented instruction."
                    )
                elif traj_data["type"] == "RxR":
                    input1 = (
                        "You are given a sequence of views of a path in an indoor environment. "
                        "Please extract critical landmarks describing the starting position and the path."
                    )
                else:
                    input1 = (
                        "You are given a sequence of views of a path. Please extract critical landmarks in the path."
                    )
                input1 = llama.utils.format_prompt(input1)
                landmarks = traj_data["landmarks"][j_instr]
                if "visual_landmarks" in traj_data and traj_data["type"] != "REVERIE":
                    # if 'visual_landmarks' in traj_data:
                    lang_landmarks = set(traj_data["landmarks"][j_instr])
                    vis_landmarks = set(traj_data["visual_landmarks"])
                    landmarks = list(lang_landmarks.union(vis_landmarks))
                input2 = input1 + ", ".join(landmarks)
            else:
                if traj_data["type"] == "REVERIE":
                    # obj_id = instr_id.split("_")[1]
                    # lang_landmarks = set(traj_data["landmarks"][j_instr])
                    # vis_landmarks = set(traj_data["visual_landmarks"])
                    # landmarks = list(lang_landmarks.union(vis_landmarks))
                    landmarks = traj_data["landmarks"][j_instr]
                    input1 = (
                        # "You are given a sequence of views of a path in an indoor environment. "
                        # "Please generate the indicated high-level target-oriented instruction briefly for an intelligent agent to follow."
                        "You are given a sequence of views of a path in an indoor environment and critical landmarks for a brief high-level target-oriented instruction. "
                        "Please generate the indicated high-level target-oriented instruction briefly for an intelligent agent to follow.\n"
                        f'Landmarks: {", ".join(landmarks)}\n'
                        # f'Target Object: {self.obj2name[f"{scan}_{obj_id}"]}'
                    )
                elif traj_data["type"] == "RxR":
                    lang_landmarks = set(traj_data["landmarks"][j_instr])
                    vis_landmarks = set(traj_data["visual_landmarks"])
                    landmarks = list(lang_landmarks.union(vis_landmarks))
                    # landmarks = traj_data["landmarks"][j_instr]
                    input1 = (
                        "You are given a sequence of views of a path in an indoor environment. "
                        "Please describe the starting position and the path according to the given landmarks in details for an intelligent agent to follow.\n"
                        f'Landmarks: {", ".join(landmarks)}'
                    )
                else:
                    lang_landmarks = set(traj_data["landmarks"][j_instr])
                    vis_landmarks = set(traj_data["visual_landmarks"])
                    landmarks = list(lang_landmarks.union(vis_landmarks))
                    # landmarks = traj_data["landmarks"][j_instr]
                    input1 = (
                        "You are given a sequence of views of a path in an indoor environment. "
                        # "Please describe the path in details for an intelligent agent to follow."
                        "Please describe the path according to the given landmarks in details for an intelligent agent to follow.\n"
                        f'Landmarks: {", ".join(landmarks)}'
                    )
                input1 = llama.utils.format_prompt(input1)
                input2 = input1 + traj_data["instructions"][j_instr]

        ori_prompt = input1

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_txt_len - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[: self.max_txt_len]
        labels = copy.deepcopy(input2)
        labels[: len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()

        instr_encoding = input2

        hist_inputs = self.get_history_feature(
            scan, path, path_viewindex, rel_act_angles, t_cur, return_img_probs=return_hist_img_probs, return_next_step=False
        )

        hist_lens = min(t_cur + 1, len(path))
        # if return_ob:
        #     hist_lens = min(t_cur + 2, len(path))
        # else:
        #     hist_lens = min(t_cur + 1, len(path))

        outs = {
            "scan": scan,
            "path": path,
            "path_id": traj_data["path_id"],
            "abs_pos_angles": abs_pos_angles,
            "rel_act_angles": rel_act_angles,
            "instr_id": instr_id,
            "instr": traj_data["instructions"][j_instr],
            "instr_encoding": instr_encoding,
            "hist_img_fts": hist_inputs[0],
            "hist_ang_fts": hist_inputs[1],
            "hist_lens": hist_lens,
            "labels": labels,
            "input2_mask": input2_mask,
            "ori_prompt": ori_prompt,
        }
        if self.hist_enc_pano:
            outs["hist_pano_img_fts"] = hist_inputs[2]
            outs["hist_pano_ang_fts"] = hist_inputs[3]
        if return_hist_img_probs:
            outs["hist_img_probs"] = hist_inputs[4]

        if return_ob:
            if ob_cand_pano_view is None:
                ob_cand_pano_view = self.ob_cand_pano_view
            if ob_cand_pano_view:
                ob_img_feats, ob_ang_feats, ob_nav_types, gt_label, gt_angle = self.get_ob_cand_pano_view(
                    scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur
                )
            else:
                ob_img_feats, ob_ang_feats, ob_nav_types, gt_label, gt_angle = self.get_ob_pano_view(
                    scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur
                )

            # ob_id_seps = torch.argwhere(input1 == 396)  # # is 396
            ob_id_seps = torch.argwhere(input2 == 396)  # # is 396
            # ob_attn_mask = torch.zeros((self.max_txt_len, 36))
            # for i, v in enumerate(visited_views_list):
            #     ob_attn_mask[ob_id_seps[i].item()+1:ob_id_seps[i+1].item(), v] = 1

            ob_id_seps = ob_id_seps.squeeze(1).tolist()
            if len(ob_id_seps) > len(visited_views_list) + 2:
                start_pos = len(ob_id_seps) - len(visited_views_list) - 2
                ob_id_seps = ob_id_seps[start_pos:]

            outs.update(
                {
                    # 'ob_img_fts': ob_img_feats[:-1],
                    # 'ob_ang_fts': ob_ang_feats[:-1],
                    # 'ob_nav_types': ob_nav_types[:-1],
                    # 'ob_attn_mask': ob_attn_mask,
                    "ob_img_fts": ob_img_feats,
                    "ob_ang_fts": ob_ang_feats,
                    "ob_nav_types": ob_nav_types,
                    "ob_id_seps": ob_id_seps,
                    "gt_id": gt_id,
                }
            )
            if return_ob_action:
                outs["ob_action_viewindex"] = gt_label
                outs["ob_action_angles"] = gt_angle
            if return_ob_progress:
                outs["ob_progress"] = self.get_progress(
                    scan,
                    path[0],
                    path[t_cur],
                    path[-1] if "guide_path" not in traj_data else traj_data["guide_path"][-1],
                )

        return outs

    def get_ob_pano_view(self, scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur):
        ob_img_feats = self.get_image_feature(scan, path[t_cur], pad_stop_token=True)[:, : self.image_feat_size]
        ob_ang_feats = self.get_angle_feature(path_viewindex[t_cur], pad_stop_token=True)
        ob_nav_types = np.zeros((ob_img_feats.shape[0],), dtype=np.int64)
        ob_nav_types[-1] = 2  # 2 for [STOP]
        ob_nav_cands = self.scanvp_cands[f"{scan}_{path[t_cur]}"]
        ob_nav_viewindexes = np.array([v[0] for v in ob_nav_cands.values()])
        ob_nav_types[ob_nav_viewindexes] = 1

        if action_viewindex[t_cur] != -1:
            gt_label = action_viewindex[t_cur]
            gt_angle = rel_act_angles[t_cur]
        else:  # stop
            gt_label = ob_img_feats.shape[0] - 1
            gt_angle = np.zeros((2,), dtype=np.float32)

        return ob_img_feats, ob_ang_feats, ob_nav_types, gt_label, gt_angle

    def get_ob_cand_pano_view(self, scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur):
        # 36 pano views
        ob_img_feats = self.get_image_feature(scan, path[t_cur], pad_stop_token=False)[:, : self.image_feat_size]
        ob_ang_feats = self.get_angle_feature(path_viewindex[t_cur], pad_stop_token=False)

        # cand views
        ob_nav_cands = self.scanvp_cands["%s_%s" % (scan, path[t_cur])]
        cand_img_feats, cand_ang_feats = [], []
        non_cand_viewidxs = np.ones((36,), dtype=np.bool)
        gt_label = None
        for k, v in ob_nav_cands.items():
            if t_cur < len(path) - 1 and k == path[t_cur + 1]:  # non-stop
                gt_label = len(cand_img_feats)
            non_cand_viewidxs[v[0]] = False
            cand_img_feats.append(ob_img_feats[v[0]])
            tmp_angle = self.rel_angles[path_viewindex[t_cur]][v[0]]
            cand_ang_feats.append(angle_feature(tmp_angle[0] + v[2], tmp_angle[1] + v[3], self.angle_feat_size))
        cand_img_feats = np.stack(cand_img_feats, 0)
        cand_ang_feats = np.stack(cand_ang_feats, 0)

        # non cand views
        non_cand_img_feats = ob_img_feats[non_cand_viewidxs]
        non_cand_ang_feats = ob_ang_feats[non_cand_viewidxs]

        # combine
        ob_nav_types = np.array([1] * len(cand_img_feats) + [2] + [0] * len(non_cand_img_feats))
        ob_img_feats = np.concatenate(
            [cand_img_feats, np.zeros((1, self.image_feat_size), dtype=np.float32), non_cand_img_feats], 0
        )
        ob_ang_feats = np.concatenate(
            [cand_ang_feats, np.zeros((1, self.angle_feat_size), dtype=np.float32), non_cand_ang_feats], 0
        )

        if gt_label is None:  # stop action
            gt_label = len(cand_img_feats)
            gt_angle = np.zeros((2,), dtype=np.float32)
        else:
            gt_angle = rel_act_angles[t_cur]

        return ob_img_feats, ob_ang_feats, ob_nav_types, gt_label, gt_angle

    def get_history_feature(self, scan, path, path_viewindex, rel_act_angles, t_cur, return_img_probs=False, return_next_step=False):
        # get history features before the step t_cur
        image_feats, angle_feats, image_probs = [], [], []
        pano_image_feats, pano_angle_feats = [], []

        if return_next_step:
            t_cur = min(t_cur + 2, len(path))
        else:
            t_cur = min(t_cur + 1, len(path))

        for t in range(0, t_cur):
            vp = path[t]
            viewidx = path_viewindex[t]
            heading, elevation = rel_act_angles[t]

            # if t == len(path) - 1:  # STOP Action
            if t == t_cur - 1:
                angle_feats.append(np.zeros((self.angle_feat_size,), dtype=np.float32))
            else:
                angle_feats.append(angle_feature(heading, elevation, self.angle_feat_size))

            vp_fts = self.get_image_feature(scan, vp, pad_stop_token=False)

            image_feats.append(vp_fts[viewidx, : self.image_feat_size])

            if self.hist_enc_pano:
                pano_image_feats.append(vp_fts[:, : self.image_feat_size])
                pano_angle_feats.append(self.angle_features[viewidx])

            if return_img_probs:
                image_probs.append(vp_fts[viewidx, self.image_feat_size :])

        if t_cur > 0:
            image_feats = np.stack(image_feats, 0)
            angle_feats = np.stack(angle_feats)

            if return_next_step:
                image_feats[-1] = np.zeros((self.image_feat_size,), dtype=np.float32)
                angle_feats[-2] = np.zeros((self.angle_feat_size,), dtype=np.float32)

            if self.hist_enc_pano:
                pano_image_feats = np.stack(pano_image_feats, 0)
                pano_angle_feats = np.stack(pano_angle_feats, 0)

            if return_img_probs:
                image_probs = np.stack(image_probs, 0)
                image_probs = softmax(image_probs)
        else:
            image_feats = np.zeros((0, self.image_feat_size), dtype=np.float32)
            angle_feats = np.zeros((0, self.angle_feat_size), dtype=np.float32)
            if self.hist_enc_pano:
                pano_image_feats = np.zeros((0, 36, self.image_feat_size), dtype=np.float32)
                pano_angle_feats = np.zeros((0, 36, self.angle_feat_size), dtype=np.float32)
            image_probs = np.zeros((0, self.image_prob_size), dtype=np.float32)

        if return_img_probs:
            return image_feats, angle_feats, pano_image_feats, pano_angle_feats, image_probs

        return image_feats, angle_feats, pano_image_feats, pano_angle_feats

    def get_image_feature(self, scan, viewpoint, pad_stop_token=False):
        key = f"{scan}_{viewpoint}"
        if self.in_memory and key in self._feature_store:
            fts = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, "r") as f:
                fts = f[key][...].astype(np.float32)
            if self.in_memory:
                self._feature_store[key] = fts

        if pad_stop_token:
            fts = np.vstack([fts, np.zeros((1, fts.shape[-1]), dtype=fts.dtype)])
        return fts

    def get_angle_feature(self, viewindex, pad_stop_token=False):
        fts = self.angle_features[viewindex]
        if pad_stop_token:
            fts = np.vstack([fts, np.zeros((1, fts.shape[-1]), dtype=fts.dtype)])
        return fts

    def get_progress(self, scan, start_vp, cur_vp, end_vp):
        if cur_vp == end_vp:
            return 1
        elif start_vp == cur_vp:
            return 0
        else:
            total_dist = self.shortest_distances[scan][start_vp][end_vp]
            remained_dist = self.shortest_distances[scan][cur_vp][end_vp]
            return 1 - remained_dist / max(total_dist, 0.1)
