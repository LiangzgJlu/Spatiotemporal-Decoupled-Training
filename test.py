import os
import sys
import torch 
import datetime
import argparse
import numpy as np
import torch.nn as nn
from loguru import logger
from pathlib import Path
from models.cfm import create_model
from dataset.dataset import FEATURE_RANGE
from utils.torch_utils import select_device
from utils.general import increment_path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def parse_option():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="", help="weights path")
    parser.add_argument("--cfg", type=str, default="", help="net config path")
    parser.add_argument("--dateset_path", type=str, default="", help="the path of test trajectory")
    parser.add_argument("--history_windows_length", type=int, default=25, help="history windows length")
    parser.add_argument("--project", default=ROOT / "runs/test", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--linear_normal", type=bool, default=True, help="feature linear normalization")
    parser.add_argument("--device", type=str, default='cuda', help='cuda device')
    parser.add_argument("--time_step", type=float, default=0.04, help='time step')
    return parser.parse_args()

def linear_normal_feature(hfss):
    hfss[:, :, 0] = (hfss[:, :, 0] - FEATURE_RANGE[0][1]) / (FEATURE_RANGE[0][0] - FEATURE_RANGE[0][1])
    hfss[:, :, 1] = (hfss[:, :, 1] - FEATURE_RANGE[1][1]) / (FEATURE_RANGE[1][0] - FEATURE_RANGE[1][1])
    hfss[:, :, 2] = (hfss[:, :, 2] - FEATURE_RANGE[2][1]) / (FEATURE_RANGE[2][0] - FEATURE_RANGE[2][1])
    hfss[:, :, 3] = (hfss[:, :, 3] - FEATURE_RANGE[3][1]) / (FEATURE_RANGE[3][0] - FEATURE_RANGE[3][1])
    
    return hfss

def test(opt, model: nn.Module, device):
    logger.info("start test cfm")
    model.eval()
    dataset = np.load(opt.dateset_path, allow_pickle=True)
    hwl = int(opt.history_windows_length)
    
    gap_sum_error = 0
    ae_gap_sum_error = 0
    jerk_sum_error = 0
    collision_count = 0

    gap_ae_list = []
    i = 0
    all_count = 0
    for trajectory in dataset:
        logger.info(f"handling {i}, shape : {trajectory.shape}, all: {len(dataset)}")
        
        data = trajectory[:hwl][np.newaxis, :].astype(np.float32)
        j = hwl
        gap_error = 0
        ae_gap_error = 0
        jerk_error = 0
        
        gap, fv, dv, pv = trajectory[j - 1]
        last_acc = (trajectory[j - 1, 1] - trajectory[j - 2,1]) / opt.time_step
        
        if opt.linear_normal:
            data = linear_normal_feature(data)
        
        while j < trajectory.shape[0] - 1:
            obs = torch.tensor(data).to(device)
            a = model(obs).detach().item()
            
            acc = np.clip(a, -10, 10)
                    
            if gap < 3:
                acc = -9

            fv += acc * opt.time_step
    
            if fv < 0:
                fv = 0
                acc = 0
                
            pv = trajectory[j, 3]
            dv_t = pv - fv
            gap += (dv + dv_t) / 2 * opt.time_step
            dv = dv_t
            
            if gap < 0:
                collision_count += 1
                logger.info(f"happen collision at trajectory {i}, following speed: {fv}, preceding speed: {pv}, acceleration: {acc}, model output acceleration: {a}")
                break
            
            data = data[:, 1:]
        
            new_item = np.array([gap, fv, dv, pv], dtype=np.float32).reshape((1, 1, 4))

            if opt.linear_normal:
                new_item = linear_normal_feature(new_item)
            data = np.concatenate([data, new_item], axis=1)
            
            jerk_s = (acc - last_acc) / opt.time_step
            ge = (gap - trajectory[j, 0]) ** 2
            
            je = np.abs(jerk_s) 

            gap_error += ge
            jerk_error += je
            ae_gap_error += abs(gap - trajectory[j, 0])
            gap_ae_list.append(abs(gap - trajectory[j, 0]))
            
            last_acc = acc
            j += 1
            
       
        logger.info(f"{i}, {ae_gap_error / (trajectory.shape[0] - hwl), gap_error / (trajectory.shape[0] - hwl)}")
        gap_sum_error += gap_error
        jerk_sum_error += jerk_error
        ae_gap_sum_error += ae_gap_error
        all_count += j
        i += 1
    logger.info("gap mae: {}, gap mse: {}, jerk mae: {}, collision count: {}"
                .format(
                    ae_gap_sum_error / all_count,
                    gap_sum_error / all_count,
                    jerk_sum_error / all_count,
                    collision_count,
    ))
            
def main(opt):
    print(opt.device)
    device = select_device(opt.device, batch_size=0)
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name))
    
    log_path = opt.save_dir + os.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    logger.add(log_path)
    logger.info(opt.__str__())
    
    model = create_model(opt).to(device=device)    
    if opt.weights != '':
        model.load_state_dict(torch.load(opt.weights))
        
    test(opt=opt, model=model, device=device)

if __name__ == "__main__":
    """
    Purpose:

    Evaluate a trained car-following model (CFM) on test trajectories by rolling out predicted accelerations and computing trajectory-level metrics.
    Key functionalities:

    Argument parsing:

    --weights: path to model weights (.pt/.pth).
    --cfg: model configuration file path (used by create_model).
    --dateset_path: path to the test trajectory .npy file .
    --history_windows_length: number of past steps used as the initial context for rollout (default 25).
    --project, --name: output directory control (results saved under runs/test/name with auto-incrementing exp folders).
    --linear_normal: whether to apply feature-wise linear normalization based on FEATURE_RANGE.
    --device: computation device (e.g., cuda or cpu).
    --time_step: simulation time step in seconds (default 0.04).

    """
    opt = parse_option()
    main(opt)
    
    
    
    