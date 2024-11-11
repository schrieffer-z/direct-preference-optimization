# python -u train.py model=pythia69 loss=dpo loss.beta=0.1 lr=3e-7 exp_name=sft_pythia69_bs64_ebs32 batch_size=32 eval_batch_size=16 model.fsdp_policy_mp=bfloat16 model.archive=../datasets/root/sft_pythia69_bs64_ebs32/LATEST/policy.pt datasets=[hh] 


# python -u train.py model=pythia28 loss=sft exp_name=sft_pythia28_bs64_ebs32 batch_size=64 eval_batch_size=32 datasets=[hh]


python -u train.py model=pythia28 loss=dpo loss.beta=0.1 exp_name=dpo_pythia28_bs64_ebs32 batch_size=64 eval_batch_size=32 model.fsdp_policy_mp=bfloat16 model.archive=../datasets/root/sft_pythia28_bs64_ebs32/LATEST/policy.pt datasets=[hh] 


python -u train.py model=pythia28 loss=sft exp_name=sft_pythia28_bs64_ebs32 batch_size=64 eval_batch_size=32 datasets=[hh]