import os
from configs.parser import YAMLParser
import mlflow
import torch
# from added_codes.funcs import visualize_flow
from added_codes.classes import AEE_HIST



from utils.utils import load_model
from models.model import (
    FireNet,
    RNNFireNet,
    LeakyFireNet,
    FireFlowNet,
    LeakyFireFlowNet,
    E2VID,
    EVFlowNet,
    RecEVFlowNet,
    LeakyRecEVFlowNet,
    RNNRecEVFlowNet,
)

runid = "EVFlowNet"

mlflow.set_tracking_uri("")
run = mlflow.get_run(runid)
config_parser = YAMLParser("configs/eval_MVSEC.yml")
config = config_parser.config
config = config_parser.merge_configs(run.data.params)







device = config_parser.device
# device = torch.device("cpu")
print("start loading")
model = RecEVFlowNet(config["model"]).to(device)
print("finished loading")
model.eval()






event_voxel = torch.rand([1, 2, 256, 256]).to(device)
event_cnt = torch.rand([1, 2, 256, 256]).to(device)



for i in range(50):
    print(i,flush=True)
    model(event_voxel, event_cnt, log=False)