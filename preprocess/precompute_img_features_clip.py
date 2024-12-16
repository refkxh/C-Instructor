''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import argparse
import math
import os

import h5py
import numpy as np
from PIL import Image
from progressbar import ProgressBar
import torch
import torch.multiprocessing as mp

import clip
import MatterSim

from utils import load_viewpoint_ids


TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w',
                  'image_h', 'vfov', 'features', 'logits']
VIEWPOINT_SIZE = 36  # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 640
HEIGHT = 480
VFOV = 60


def build_feature_extractor(model_name, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, img_transforms = clip.load(model_name, device='cpu')
    model.to(device)
    model.eval()

    return model, img_transforms, device


def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(True)
    sim.setBatchSize(1)
    sim.initialize()
    return sim


def clip_encode_image(model, x):
    # modified from CLIP
    x = model.visual.conv1(x)  # shape = [*, width, grid, grid]
    # shape = [*, width, grid ** 2]
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                    x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # preserve all spatial tokens
    # x = model.visual.ln_post(x[:, :, :])
    x = model.visual.ln_post(x[:, 0, :])

    if model.visual.proj is not None:
        x = x @ model.visual.proj

    return x


def process_features(proc_id, out_queue, scanvp_list, args):
    print(f'start proc_id: {proc_id}')

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(args.model_name, args.checkpoint_file)

    for scan_id, viewpoint_id in scanvp_list:
        # Loop all discretized views from this location
        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id],
                               [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            image = np.array(state.rgb, copy=True)  # in BGR channel
            # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = Image.fromarray(image[:, :, ::-1])
            image = Image.fromarray(image)
            images.append(image)

        images = torch.stack([img_transforms(image).to(device) for image in images], 0)
        fts = []
        for k in range(0, len(images), args.batch_size):
            b_fts = clip_encode_image(model, images[k: k+args.batch_size])
            b_fts = b_fts.data.cpu().numpy()  # B, 768
            fts.append(b_fts)
        fts = np.concatenate(fts, 0)

        out_queue.put((scan_id, viewpoint_id, fts))

    out_queue.put(None)


def build_feature_file(args):

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)

    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts = res
                key = f'{scan_id}_{viewpoint_id}'
                data = fts
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ViT-L/14')
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--connectivity_dir', default='../../connectivity')
    parser.add_argument('--scan_dir', default='../../data/v1/scans')
    parser.add_argument('--output_file')
    parser.add_argument('--batch_size', default=36, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    mp.set_start_method('spawn')

    build_feature_file(args)
