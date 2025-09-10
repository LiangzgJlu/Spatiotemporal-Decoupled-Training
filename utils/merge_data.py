import argparse
import numpy as np
from loguru import logger 
from pathlib import Path
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="train", help="merge type: train, val, test")
    parser.add_argument("--style", type=str, default="timid", help="drive style: timid, normal, aggressive.")
    parser.add_argument("--save_path", type=str, default="", help="merge data save path.")
    parser.add_argument("--merge_data_path", type=str, default="", help="wait to merge data, format: path1;path2")
    parser.add_argument("--history_windows_length", type=int, default=10, help="the length of history windows")
    parser.add_argument("--time_step", type=float, default=0.04, help="trajectory time step")
    opt = parser.parse_args()
    
    merge_data_path_list = str(opt.merge_data_path).split(';')
    
    if opt.type != 'test':
        hwl = opt.history_windows_length
        hfss = []
        targets = []
        for path in merge_data_path_list:
            logger.info(f"handling data: {path}")
            data = np.load(path, allow_pickle=True)
            logger.info(f"{data.shape}")
            for trajectory in data:
                for i in range(len(trajectory) - hwl):
                    temp = trajectory[i:i+hwl, 0:4]
                    acc = (trajectory[i+hwl, 1] - trajectory[i+hwl - 1, 1]) / opt.time_step
                    hfss.append(temp.astype(np.float32))
                    targets.append(float(acc))
                    
        dataset_pool = np.array([hfss, targets], dtype=object)  
        
        save_file = Path(opt.save_path)  / str(opt.style + "_" + opt.type + "_" + str(hwl) + ".npy") 
        np.save(save_file, dataset_pool, allow_pickle=True)
        logger.info(f"save history following state series to {save_file}, count: {len(hfss)}")
    else:
        trajectories = []
        for path in merge_data_path_list:
            logger.info(f"handling data: {path}")
            data = np.load(path, allow_pickle=True) 
            for t in data:
                trajectories.append(t.astype(np.float32)[:, :4])
        trajectories = np.array(trajectories, dtype=object)
        save_file = Path(opt.save_path)  / str(opt.style + "_" + opt.type + ".npy")
        np.save(save_file, trajectories, allow_pickle=True)
        logger.info(f"save test trajectory to {save_file}, count: {len(trajectories)}")
        
            
    