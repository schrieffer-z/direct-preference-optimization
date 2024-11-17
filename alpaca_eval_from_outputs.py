import os
import alpaca_eval



model_size = '2.8b'
outputs_path = f'./evaluation/pythia_{model_size}'

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


import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv(f'/mnt/vepfs/fs_users/lisihang/xAI-RLHF/Shuyi/direct-preference-optimization/evaluation/pythia_{model_size}/leaderboard.csv')

t = list(data.columns)
t[0]='model'
data.columns = t

step_num =[]
for mname in data['model']:
    step_num.append(int(mname[mname.rfind('_')+1:]))

data['dpo_step_num'] = step_num



data = data.sort_values(by='dpo_step_num')[['dpo_step_num', 'win_rate']]
data['dpo_step_num'] = [str(int(i))+"k" for i in list(data['dpo_step_num'] / 1000)]

plt.plot(list(data['dpo_examples_num']), list(data['win_rate']))
plt.plot(list(data['dpo_examples_num']), [50]*len(data['dpo_step_num']), color='red', linestyle='--')
plt.xlabel('DPO Steps')
plt.ylabel('Win Rate(Alpaca Eval)')

plt.savefig(f'pythia-{model_size}.png', dpi=300)
