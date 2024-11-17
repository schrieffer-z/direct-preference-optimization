import datasets
from preference_datasets import get_batch_iterator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

ref_model = AutoModelForCausalLM.from_pretrained('../../../models/pythia-2.8b' , low_cpu_mem_usage=True, torch_dtype=torch.float32)
policy_model = AutoModelForCausalLM.from_pretrained('../../../models/pythia-2.8b' , low_cpu_mem_usage=True, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained('../../../models/pythia-2.8b')


model_archive = '../datasets/root/sft_pythia69_bs64_ebs32/LATEST/policy.pt'
state_dict = torch.load(model_archive, map_location='cpu')
step, metrics = state_dict['step_idx'], state_dict['metrics']
print(f'loading pre-trained weights at step {step} from {model_archive} with metrics {json.dumps(metrics, indent=2)}')
policy_model.load_state_dict(state_dict['state'])
# ref_model.load_state_dict(state_dict['state'])
print('loaded pre-trained weights for policy and ref')

data_iterator_kwargs = dict(
            names='hh',
            tokenizer= tokenizer,
            shuffle=True,
            max_length=512,
            max_prompt_length=256,
            sft_mode=False,
        )



train_iterator = get_batch_iterator(**data_iterator_kwargs, 
                                    split='train', 
                                    n_epochs=1,  
                                    batch_size=64, 
                                    silent=False)

