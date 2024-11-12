from openai import AzureOpenAI
import datasets
from tqdm import tqdm
import os
from datetime import datetime




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


i=0
for row in tqdm(dataset, desc='Processing HH'):
    i+=1
    prompt, chosen, rejected = split_prompt_and_responses(row)
    text = winrate_prompt.replace("<the user query>", prompt)\
        .replace("<either the test method or baseline>", chosen)\
        .replace("<the other response>", rejected)
    
    response = gpt_4o_call(text)
    with open(log_dir+'/'+str(i)+'.txt', 'w', encoding='utf-8') as file:
        file.write(prompt+'\nResponse A:'+chosen+'\nResponse B:'+rejected+'\n\n'+response)
    if i==5:
        break


