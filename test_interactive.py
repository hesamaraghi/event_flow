import os
from configs.parser import YAMLParser
import mlflow
from dataloader.h5 import H5Loader
import torch

# from added_codes.funcs import visualize_flow
from added_codes.classes import AEE_HIST
import cv2

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

data = H5Loader(config, config["model"]["num_bins"])
dataloader = torch.utils.data.DataLoader(
    data,
    drop_last=True,
    batch_size=config["loader"]["batch_size"],
    collate_fn=data.custom_collate,
    worker_init_fn=config_parser.worker_init_fn,
)

device = config_parser.device
# device = torch.device("cpu")
print("start loading", flush=True)
model = RecEVFlowNet(config["model"]).to(device)
print("finished loading", flush=True)
model = load_model(runid, model, device)
model.eval()


aee_hist = AEE_HIST(config, device, flow_scaling=config["metrics"]["flow_scaling"])
end_test = False
c = 0
val_metrics = []
print("New sequence: ", data.seq_num)
while True:
    for inputs in dataloader:
        if data.new_seq or c == 2000:
            data.new_seq = False
            activity_log = None
            model.reset_states()
            print("New sequence: ", data.seq_num)
            end_test = True
            break

        # forward pass
        x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device), log=config["vis"]["activity"])

        aee_hist.event_flow_association(x["flow"], inputs)
        val_metrics.append(aee_hist())
        c += 1

        print(c)

        aee_hist.flow_accumulation()
        aee_hist.reset()
        # finish inference loop
        if data.seq_num >= len(data.files):
            end_test = True
            break

    if end_test:
        break

torch.save(aee_hist.calculate_error_hist(), "out.pt")
