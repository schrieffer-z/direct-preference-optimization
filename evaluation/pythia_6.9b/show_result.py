import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('/mnt/vepfs/fs_users/lisihang/xAI-RLHF/Shuyi/direct-preference-optimization/evaluation/pythia_2.8b/alpaca_eval_gpt4_turbo_fn/leaderboard.csv')

t = list(data.columns)
t[0]='model'
data.columns = t

step_num =[]
for mname in data['model']:
    step_num.append(int(mname[mname.rfind('_')+1:]))

data['dpo_step_num'] = step_num



data = data.sort_values(by='dpo_step_num')[['dpo_step_num', 'win_rate']]
data['dpo_step_num'] = [str(int(i))+"k" for i in list(data['dpo_step_num'] / 1000)]

plt.plot(list(data['dpo_step_num']), list(data['win_rate']))
plt.plot(list(data['dpo_step_num']), [50]*len(data['dpo_step_num']), color='red', linestyle='--')
plt.xlabel('DPO Steps')
plt.ylabel('Win Rate(Alpaca Eval)')

plt.savefig('pythia-6.9b.png', dpi=300)
