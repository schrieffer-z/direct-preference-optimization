# python -u train.py model=pythia69 loss=dpo loss.beta=0.1 lr=3e-7 exp_name=sft_pythia69_bs64_ebs32 batch_size=32 eval_batch_size=16 model.fsdp_policy_mp=bfloat16 model.archive=../datasets/root/sft_pythia69_bs64_ebs32/LATEST/policy.pt datasets=[hh] 


# python -u train.py model=pythia28 loss=sft exp_name=sft_pythia28_bs64_ebs32 batch_size=64 eval_batch_size=32 datasets=[hh]


# python -u train.py model=pythia28 loss=dpo loss.beta=0.1 exp_name=dpo_pythia28_bs64_ebs32 batch_size=64 eval_batch_size=32 model.fsdp_policy_mp=bfloat16 model.archive=../datasets/root/sft_pythia28_bs64_ebs32/LATEST/policy.pt datasets=[hh] 


# python -u train.py model=pythia28 loss=sft exp_name=sft_pythia28_bs64_ebs32 batch_size=64 eval_batch_size=32 datasets=[hh]

python -u train.py cuda_world=[4,5,6,7] model=pythia28 loss.beta=0.1 lr=5e-7 \
exp_name=dpo_pythia28_bs48_ebs24 batch_size=64 eval_batch_size=32 \
model.archive=../datasets/root/sft_pythia69_bs64_ebs32/LATEST/policy.pt datasets=[hh] model.fsdp_policy_mp=bfloat16 loss=dpo

python -u train.py cuda_world=[4,5,6,7] model=pythia69 loss.beta=0.1 lr=3.75e-7 \
exp_name=dpo_pythia69_bs48_ebs24 batch_size=48 eval_batch_size=24 \
model.archive=../datasets/root/sft_pythia69_bs64_ebs32/LATEST/policy.pt datasets=[hh] model.fsdp_policy_mp=bfloat16 loss=dpo

