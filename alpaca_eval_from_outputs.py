import os
import alpaca_eval



model_size = '6.9b'
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
