import os
import torch
from loguru import logger

def select_device(device="", batch_size=0):
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(",", "")), (
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"
        )
    if not cpu and torch.cuda.is_available():
   
        arg = "cuda:0"
    else:
        arg = "cpu"
    
    return torch.device(arg)
    
    
def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9):
    if name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, betas=(momentum, 0.999))
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplemented(f"Optimizer {name} not implemented.")
    
    logger.info(f"{('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups ")
    
    return optimizer
if __name__ == '__main__':
    device = select_device('cpu')
    