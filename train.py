import os
import sys
import yaml
import torch
import datetime
import argparse
import copy
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from models.cfm import create_model
from dataset.dataset import create_dataset, FEATURE_RANGE
from loguru import logger
from pathlib import Path
from utils.torch_utils import select_device, smart_optimizer
from utils.general import increment_path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


class EarlyStopper:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_loss = float('inf')
        self.best_model = None

    def __call__(self, model, val_loss, best_path):
        if val_loss < self.min_loss - self.delta:
            self.min_loss = val_loss
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), best_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(self.best_model)
                return True
        return False

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def linear_normal_feature(hfss):
    hfss[:, :, 0] = (hfss[:, :, 0] - FEATURE_RANGE[0][1]) / (FEATURE_RANGE[0][0] - FEATURE_RANGE[0][1])
    hfss[:, :, 1] = (hfss[:, :, 1] - FEATURE_RANGE[1][1]) / (FEATURE_RANGE[1][0] - FEATURE_RANGE[1][1])
    hfss[:, :, 2] = (hfss[:, :, 2] - FEATURE_RANGE[2][1]) / (FEATURE_RANGE[2][0] - FEATURE_RANGE[2][1])
    hfss[:, :, 3] = (hfss[:, :, 3] - FEATURE_RANGE[3][1]) / (FEATURE_RANGE[3][0] - FEATURE_RANGE[3][1])
    # hfss[:, :, 2] = hfss[:, :, 3] - hfss[:, :, 1]
    return hfss

def eval_model(model, opt, device):
    logger.info("start test cfm")
    model.eval()
    dataset = np.load(opt.val_dataset_path, allow_pickle=True)
    hwl = int(opt.history_windows_length)
    
    gap_sum_error = 0
    ae_gap_sum_error = 0
    i = 0
    all_count = 0
    for trajectory in dataset:        
        data = trajectory[:hwl][np.newaxis, :].astype(np.float32)
        j = hwl
        gap_error = 0
        ae_gap_error = 0
        
        gap, fv, dv, pv = trajectory[j - 1]
        
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
                break
            
            data = data[:, 1:]
        
            new_item = np.array([gap, fv, dv, pv], dtype=np.float32).reshape((1, 1, 4))

            if opt.linear_normal:
                new_item = linear_normal_feature(new_item)
            data = np.concatenate([data, new_item], axis=1)
                        
            gap_error += (gap - trajectory[j, 0]) ** 2
            ae_gap_error += abs(gap - trajectory[j, 0])
            j += 1                        
                        
        gap_sum_error += gap_error 
        ae_gap_sum_error += ae_gap_error
        all_count += j
        i += 1

    model.train()
    
    return ae_gap_sum_error / all_count



def train(opt, model: nn.Module, device):
    logger.info("start train")
    model.train()
    
    dataset = create_dataset(opt)
    
    hyp = dict()
    with open(opt.hyp, 'r') as file:
        hyp: dict = yaml.load(file, Loader=yaml.FullLoader)
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'])

    # 评价标准
    criterion = nn.MSELoss()
    
    loss_list = []
    
    early_stopper = EarlyStopper(10, 1e-5)
    min_loss = 1e8
    if opt.tsi:
        data_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)
        for i_episode in range(int(opt.epochs)):
            total_loss = float(0)
            
            for inputs, targets in data_loader:
                inputs = inputs.type(dtype=torch.float32).to(device)
                targets = targets.type(dtype=torch.float32).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss_list.append(total_loss)
            
            val_error = eval_model(model=model, opt=opt, device=device)
            
            logger.info(f"epoch: {i_episode}, total loss: {total_loss}")
            
            stopped = early_stopper(model, val_error, opt.best_path)

            if stopped:
                break
            
            if min_loss > total_loss:
                min_loss = total_loss
                torch.save(model.state_dict(), opt.best_path)
                logger.info(f"save best model to {opt.best_path}")
            torch.save(model.state_dict(), opt.last_path)
            
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
        hwl = opt.history_windows_length
        min_loss = 1e5
        for i_episode in range(int(opt.epochs)):
            total_loss = float(0)
            j = 0
            for inputs in data_loader:
                trajectory = inputs[0]
                hfss = []
                targets = []
                for i in range(trajectory.size(0) - hwl):
                    temp = trajectory[i:i+hwl, 0:4]
                    acc = (trajectory[i+hwl, 1] - trajectory[i+hwl - 1, 1]) / opt.time_step
                    hfss.append(np.array(temp, dtype=np.float32))
                    targets.append([float(acc)])
                
                hfss = torch.FloatTensor(np.array(hfss, dtype=np.float32)).to(device)
                targets = torch.FloatTensor(np.array(targets, dtype=np.float32)).to(device)
                i = 0
                
                while i < len(hfss):
                    
                    if i + opt.batch_size < len(hfss):
                        input_data = hfss[i:i+opt.batch_size]
                        target_data = targets[i:i+opt.batch_size]
                        i += opt.batch_size          
                    else:
                        input_data = hfss[i:]
                        target_data = targets[i:]
                        i = len(hfss)

                    if opt.noise > 1e-5:
                        input_data = input_data + torch.normal(0, opt.noise, size=input_data.shape).to(device)
                    outputs = model(input_data)
                        
                    loss = criterion(outputs, target_data)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    j += 1
                    
            loss_list.append(total_loss)
            val_error = eval_model(model=model, opt=opt, device=device)
            stopped = early_stopper(model, val_error, opt.best_path)

            if stopped:
                break

            logger.info(f"epoch: {i_episode}, total loss: {total_loss}")
            
            if min_loss > total_loss:
                min_loss = total_loss
                torch.save(model.state_dict(), opt.best_path)
                logger.info(f"save best model to {opt.best_path}")
                
            torch.save(model.state_dict(), opt.last_path)
            
    logger.info("train finished.")
    np.save(opt.save_dir + "/loss.npy", np.array(loss_list))


def main(opt):
    
    device = select_device(opt.device, batch_size=opt.batch_size)    
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name))
    
    log_path = opt.save_dir + os.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    logger.add(log_path)
    logger.info(opt.__str__())
    
    # weight directory
    w = Path(opt.save_dir) / "weights"
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / "last.pt", w / "best.pt"
    opt.last_path, opt.best_path = last, best

    model = create_model(opt).to(device=device)    
    
    if opt.weights != '':
        model.load_state_dict(torch.load(opt.weights))
        
    train(opt=opt, model=model, device=device)
    
    
def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="/home/liangzg/Code/rl_dl_cfm/config/lstm.yaml", help="model.yaml path")
    parser.add_argument("--hyp", type=str, default="/home/liangzg/Code/rl_dl_cfm/config/hyp.yaml", help="hyperparameters path")
    
    parser.add_argument("--dataset_path", type=str, default="", help="dataset path")
    parser.add_argument("--val_dataset_path", type=str, default="", help="dataset path")
    parser.add_argument("--history_windows_length", type=int, default=25, help="history windows length")
    parser.add_argument("--noise", type=float, default=0.04, help="dataset noise")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="total batch size for all GPUs")
    parser.add_argument("--device", default='cuda', help='cuda device')
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="Adam", help="optimizer")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--linear_normal", type=bool, default=False, help="feature linear normalization")
    parser.add_argument("--tsi", type=int, default=1, help="time series independence train")
    parser.add_argument("--time_step", type=float, default=0.04, help='time step')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)