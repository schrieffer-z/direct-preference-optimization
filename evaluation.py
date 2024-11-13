from openai import AzureOpenAI
import datasets
from tqdm import tqdm
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json

api_base = "https://zhugeliang.openai.azure.com/"
api_key = "acf23a3ecfc04f87aa87fe7951a10cd8"
engine = "gpt-4o"
api_version = "2024-03-01-preview"

client = AzureOpenAI(
    azure_endpoint=api_base, 
    api_key=api_key,  
    api_version=api_version,
)

def gpt_4o_call(prompt):
    chat_completion = client.chat.completions.create(
            model=engine,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
    return chat_completion.choices[0].message.content


with open('./config/prompt_for_winning_rate.txt', 'r', encoding='utf-8') as f:
    winrate_prompt = f.read()

datafiles = {
     'test' : [
        # 'harmless-base/test.jsonl',
        'helpful-base/test.jsonl',
        'helpful-online/test.jsonl',
        'helpful-rejection-sampled/test.jsonl'
    ] 
}
dataset = datasets.load_dataset("../datasets/hh-rlhf/", data_files=datafiles)['test']
print(dataset)

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def split_prompt_and_responses(ex):
    prompt = extract_anthropic_prompt(ex['chosen'])
    chosen_response = ex['chosen'][len(prompt):]
    rejected_response = ex['rejected'][len(prompt):]
    return prompt, chosen_response, rejected_response

eval_log_path = "./evaluation/"
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
log_dir = eval_log_path+timestamp
os.makedirs(log_dir, exist_ok=True)


ref_model = AutoModelForCausalLM.from_pretrained('../../../models/pythia-6.9b' , low_cpu_mem_usage=True, torch_dtype=torch.float32)
policy_model = AutoModelForCausalLM.from_pretrained('../../../models/pythia-6.9b' , low_cpu_mem_usage=True, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained('../../../models/pythia-6.9b')

model_archive = '../datasets/root/sft_pythia69_bs64_ebs32/LATEST/policy.pt'
state_dict = torch.load(model_archive, map_location='cpu')
step, metrics = state_dict['step_idx'], state_dict['metrics']
print(f'loading pre-trained weights at step {step} from {model_archive} with metrics {json.dumps(metrics, indent=2)}')
policy_model.load_state_dict(state_dict['state'])
# ref_model.load_state_dict(state_dict['state'])
print('loaded pre-trained weights for policy and ref')


device = torch.device('cuda:5')
policy_pl = pipeline("text-generation",model=policy_model, tokenizer=tokenizer, max_length=1024, device=device)
ref_pl = pipeline("text-generation",model=ref_model, tokenizer=tokenizer, max_length=1024, device=device)

i=0
for row in tqdm(dataset, desc='Processing HH'):
    i+=1
    prompt,_ ,_ = split_prompt_and_responses(row)
    rpA = policy_pl(prompt)[0]['generated_text']
    rpB = ref_pl(prompt)[0]['generated_text']

    text = winrate_prompt.replace("<the user query>", prompt)\
        .replace("<either the test method or baseline>", rpA)\
        .replace("<the other response>", rpB)

    response = gpt_4o_call(text)
    with open(log_dir+'/'+str(i)+'.txt', 'w', encoding='utf-8') as file:
        file.write(prompt+'\nResponse A:'+rpA+'\nResponse B:'+rpB+'\n\n'+response)
    
    if i==5:
        break


