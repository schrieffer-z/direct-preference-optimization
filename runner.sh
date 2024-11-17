# # pythia-6.9b
# python -u train.py cuda_world=[0,1,2,3] model=pythia69 loss.beta=0.1 lr=2.5e-7 \
# exp_name=dpo_pythia69_bs32_ebs16 batch_size=32 eval_batch_size=16 \
# model.archive=../datasets/root/sft_pythia69_bs64_ebs32/policy.pt datasets=[hh] model.fsdp_policy_mp=bfloat16 loss=dpo


# python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16

# standard pythia-2.8b hyperparameter
# sft
python -u train.py cuda_world=[0,1,2,3] batch_size=64 eval_batch_size=32 \
model=pythia28 exp_name=sft_pythia28_bs64_ebs32 \
datasets=[hh] loss=sft

# dpo
python -u train.py cuda_world=[0,1,2,3] batch_size=64 eval_batch_size=32 lr=5e-8 \
model=pythia28 loss=dpo loss.beta=0.1 exp_name=dpo_lr5e-8_beta0.1_pythia28_bs64_ebs32/ \
model.archive=../datasets/root/sft_pythia28_bs64_ebs32/policy.pt \
datasets=[hh] model.fsdp_policy_mp=bfloat16 trainer=FSDPTrainer 


# python -u train.py cuda_world=[4] model=pythia28 loss.beta=0.1 lr=5e-7 \
# exp_name=test_dpo_pythia28_bs64_ebs32 batch_size=64 eval_batch_size=32 \
# model.archive=../datasets/root/sft_pythia28_bs64_ebs32/policy.pt datasets=[hh] model.fsdp_policy_mp=bfloat16 loss=dpo


# OMP_NUM_THREADS=2 \
# TOKENIZERS_PARALLELISM=false \
# CUDA_VISIBLE_DEVICES=0,1 \
# torchrun \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=localhost:0 \
#     --nnodes=1 \
#     --nproc-per-node=2 \
#     fsdp_generate.py