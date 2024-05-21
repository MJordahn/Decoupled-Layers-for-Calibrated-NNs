from src.lightning_modules.One_Stage import *
import numpy as np
from src.utils.eval_utils import *
import torch.nn as nn
#from src.utils.metrics import *
import os
from argparse import ArgumentParser
import json
import argparse
from src.utils.utils import *
import torch
from src.lightning_modules.Two_Stage import *

parser = ArgumentParser()
parser.add_argument("--save_file_name", type=str, default="")
parser.add_argument("--model_name_file", type=str, default="")
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--temperature_scale", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()


if args.save_file_name == "":
    raise Exception("Oops you did not provide a save_file_name!")
if args.model_name_file == "":
    raise Exception("Oops you did not provide a model_name_file!")
root_dir = "./"
batch_size=512

model_paths = open("./eval_path_files/"+args.model_name_file, "r")

if torch.cuda.is_available():
    device = 'cuda:0'
    device_auto = "gpu"
else:
    device = 'cpu'
    device_auto = "cpu"

if os.path.isfile('./experiment_results/table_metrics/'+args.save_file_name):
    f = open('./experiment_results/table_metrics/'+args.save_file_name, 'r') 
    results = json.load(f) 
else:
    results = {}

for model_path in model_paths.read().splitlines():
    model_path = model_path.strip()
    model_name = model_path.split("model_name=")[1].replace(".ckpt", "")
    dataset = model_name.split("_")[1]
    model_type = model_name.split("_")[0]
    if args.temperature_scale:
        model_name = "Temp-"+model_name

    if model_name not in results.keys():
        ood_done = False
        in_done = False
        shift_done = False
        train_done = False
        results[model_name] = {}
    else:
        ood_done = True
        in_done = True
        shift_done = True
        train_done = True
        if 'clean_accuracy' not in results[model_name].keys():
            in_done = False
        if 'SHIFT ECE' not in results[model_name].keys():
            shift_done = False
        if 'OOD AUROC' not in results[model_name].keys():
            ood_done = False
        if 'Train NLL' not in results[model_name].keys():
            train_done = False
        if ood_done and in_done and shift_done and train_done:
            print("SKIPPING")
            print(model_name)
            continue
    model = load_model(name=model_type, path=root_dir+model_path, device=device)
    model.eval() 
    model.return_z = False
    if args.temperature_scale:
        model = temperature_scale_model(model, dataset, batch_size)
    model = model.to(device)
    model.device = device
        

    if not train_done:
        nll_value = eval_train_data(model, dataset=dataset, batch_size=batch_size, device=device, num_samples=args.num_samples)
        results[model_name]['Train nll'] = nll_value

    if not in_done:
        ece_calc, mce_calc, acc, nll_value, brier_score, OOD_y_preds_logits, OOD_labels = eval_test_data(model, dataset=dataset, batch_size=batch_size, device=device, num_samples=args.num_samples)
        results[model_name]['clean_accuracy'] = acc.to("cpu").numpy().tolist()
        results[model_name]['ECE'] = ece_calc.to("cpu").numpy().tolist()*100
        results[model_name]['MCE'] = mce_calc.to("cpu").numpy().tolist()*100
        results[model_name]['nll'] = nll_value
        results[model_name]['brier'] = brier_score

    if not shift_done:
        ece_calc, mce_calc, acc, corruption_ece_dict, corruption_mce_dict = eval_shift_data(model, dataset=dataset, batch_size=batch_size, device=device, num_samples=args.num_samples)
        results[model_name]['SHIFT ECE'] = ece_calc.to("cpu").numpy().tolist()*100
        results[model_name]['SHIFT MCE'] = mce_calc.to("cpu").numpy().tolist()*100
        results[model_name]['SHIFT ACCURACY'] = acc.to("cpu").numpy().tolist()
        for key in corruption_ece_dict.keys():
            results[model_name]["SHIFT Intensity: " + str(key)] = {}
            results[model_name]["SHIFT Intensity: " + str(key)]['ECE'] = corruption_ece_dict[key]
            results[model_name]["SHIFT Intensity: " + str(key)]['MCE'] = corruption_mce_dict[key]

    if not ood_done:
        auroc_calc, fpr_at_95_tpr_calc = eval_ood_data(model, dataset=dataset, batch_size=batch_size, device=device, OOD_y_preds_logits=OOD_y_preds_logits, OOD_labels=OOD_labels, num_samples=args.num_samples)
        results[model_name]['OOD AUROC'] = auroc_calc
        results[model_name]['OOD FPR95'] = fpr_at_95_tpr_calc
    with open('./experiment_results/table_metrics/'+args.save_file_name, 'w') as fp:
        json.dump(results, fp)
