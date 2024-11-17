import datasets
import torch
from tqdm import tqdm
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_batch(dataset, bs=64):
    i = 0
    dict_list = []
    while i*bs < len(dataset):
        dict_list.append(dataset[i*bs: min((i+1)*bs, len(dataset))])
        dataset[i*bs: min((i+1)*bs, len(dataset))]
        i+=1
    return dict_list

device = torch.device('cuda:7')
model_size = '2.8b'
outputs_path = f'./evaluation/pythia_{model_size}'
checkpoints_path = f'/mnt/vepfs/fs_users/lisihang/xAI-RLHF/Shuyi/datasets/root/dpo_beta0.1_pythia{model_size[0]+model_size[2]}_bs64_ebs32/'


policy_model = AutoModelForCausalLM.from_pretrained(f'../../../models/EleutherAI/pythia-{model_size}' , low_cpu_mem_usage=True, torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained(f'../../../models/EleutherAI/pythia-{model_size}', padding_side='left')
tokenizer.pad_token_id = tokenizer.eos_token_id


for step_name in os.listdir(checkpoints_path):
    if step_name[0]=='s':
        checkpoints_file_name = os.path.join(checkpoints_path,step_name,'policy.pt')
        state_dict = torch.load(checkpoints_file_name, map_location='cpu', weights_only=True)
        step, metrics = state_dict['step_idx'], state_dict['metrics']

        print(f'loading pre-trained weights at step {step}')
        policy_model.load_state_dict(state_dict['state'])        

        model_outputs = {'instruction':[], 'output':[], 'generator':[], 'dataset':[]}
        eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", cache_dir='../datasets')["eval"]
        batches = generate_batch(eval_set, 16)

        with torch.no_grad():
            with tqdm(total=len(batches)) as pbar:
                for batch in batches:
                    inputs = tokenizer(batch['instruction'], return_tensors="pt", truncation=True, padding='max_length', max_length=256).to(device)
                    outputs = policy_model.generate(**inputs, max_length=512)

                    batch["output"] = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                    batch["generator"] = [f'pythia_{model_size}_post_dpo_step_{step}']*len(batch['instruction'])

                    for key in model_outputs.keys():
                        model_outputs[key] += batch[key]
                    pbar.update(1)

        to_save = []
        for i in range(len(model_outputs['instruction'])):
            to_save.append({
                'instruction':model_outputs['instruction'][i], 
                'output':model_outputs['output'][i], 
                'generator':model_outputs['generator'][i], 
                'dataset':model_outputs['dataset'][i]
                })
        
        os.makedirs(outputs_path, exist_ok=True)
        outputs_file = os.path.join(outputs_path, f'pythia_{model_size}_post_dpo_step_{step}.json')
        with open(outputs_file, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, ensure_ascii=True, indent=4)