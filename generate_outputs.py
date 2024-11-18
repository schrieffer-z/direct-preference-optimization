import datasets
import torch
from tqdm import tqdm
import alpaca_eval
import os
import json
import pandas as pd 
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_batch(dataset, bs=64):
    i = 0
    dict_list = []
    while i*bs < len(dataset):
        dict_list.append(dataset[i*bs: min((i+1)*bs, len(dataset))])
        dataset[i*bs: min((i+1)*bs, len(dataset))]
        i+=1
    return dict_list

#======================================================
model_size = '2.8b'
lr = '1e-7'
beta = '0.1'
device = torch.device('cuda:7')
#======================================================


outputs_path = f'./evaluation/pythia_{model_size}_lr{lr}_beta{beta}/'
checkpoints_path = f'/mnt/vepfs/fs_users/lisihang/xAI-RLHF/Shuyi/datasets/root/dpo_lr{lr}_beta{beta}_pythia{model_size[0]+model_size[2]}_bs64_ebs32/'


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
        
        os.makedirs(os.path.join(outputs_path,'outputs'), exist_ok=True)
        os.makedirs(os.path.join(outputs_path,'annotated_paired_outputs'), exist_ok=True)

        outputs_file = os.path.join(outputs_path,'outputs', f'post_dpo_step_{step}.json')
        with open(outputs_file, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, ensure_ascii=True, indent=4)



with open(f'./evaluation/reference_models/pythia_{model_size}_post_sft.json', 'r') as file:
    data = json.load(file)
for row in data:
    row['generator'] = f'pythia_{model_size}_post_dpo_step_0'
with open(os.path.join(outputs_path, 'outputs', 'post_dpo_step_0.json'), 'w') as file:
    json.dump(data, file, indent=4)


for fname in os.listdir( os.path.join(outputs_path, 'outputs') ):
    if fname[-5:]=='.json':
        st = fname.rfind('_')
        ed = fname.rfind('.')
        step_id = int(fname[st+1:ed])
        
        output_path = os.path.join(outputs_path, 'annotated_paired_outputs' ,f'step-{step_id}')
        os.makedirs(output_path, exist_ok=True)
        alpaca_eval.evaluate(
            model_outputs=os.path.join(outputs_path, 'outputs', fname),
            reference_outputs=f'./evaluation/reference_models/pythia_{model_size}_post_sft.json',
            annotators_config='alpaca_eval_gpt4_turbo_fn', 
            output_path=output_path,
            precomputed_leaderboard=os.path.join(outputs_path,'leaderboard.csv')
        )



data = pd.read_csv(os.path.join(outputs_path,'leaderboard.csv'))

t = list(data.columns)
t[0]='model'
data.columns = t

step_num =[]
for mname in data['model']:
    step_num.append(int(mname[mname.rfind('_')+1:]))

data['dpo_examples_num'] = step_num



data = data.sort_values(by='dpo_examples_num')[['dpo_examples_num', 'win_rate']]
data['dpo_examples_num'] = [str(int(i))+"k" for i in list(data['dpo_examples_num'] / 1000)]

plt.plot(list(data['dpo_examples_num']), list(data['win_rate']))
plt.plot(list(data['dpo_examples_num']), [50]*len(data['dpo_examples_num']), color='red', linestyle='--')
plt.xlabel('DPO Examples')
plt.ylabel('Win Rate(Alpaca Eval)')

plt.savefig(os.path.join(outputs_path,f'pythia-{model_size}.png'), dpi=300)