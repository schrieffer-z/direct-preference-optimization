import os
import alpaca_eval



        
for step_name in os.listdir(checkpoints_path):
    if step_name[0]=='s':
        outputs_file = os.path.join(outputs_path, f'pythia_{model_size}_post_dpo_step_{step}.json')
        alpaca_eval.evaluate(
            model_outputs=outputs_file,
            reference_outputs=f'./evaluation/reference_model/pythia_{model_size}_post_sft.json',
            annotators_config='alpaca_eval_gpt4_turbo_fn', 
            output_path=outputs_path,
            precomputed_leaderboard='./evaluation/alpaca_eval_gpt4_turbo_fn/leaderboard.csv')