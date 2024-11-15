# export OPENAI_CLIENT_CONFIG_PATH="/mnt/vepfs/fs_users/lisihang/xAI-RLHF/Shuyi/alpaca_eval/client_configs/openai_configs.yaml"

# export OPENAI_API_BASE="https://zhugeliang.openai.azure.com/"
# export OPENAI_API_KEY="acf23a3ecfc04f87aa87fe7951a10cd8"

export OPENAI_API_BASE="https://xiaoai.plus/v1"
export OPENAI_API_KEY="sk-9DjgE9RuYiLaGm8NGB5oApMX3mEAzZaZiS66XMbHwBYqAm1S"

alpaca_eval --model_outputs './evaluation/pythia_6.9b/pythia_6.9b_post_dpo_step_20000.json' \
--reference_outputs './evaluation/reference_model/pythia_6.9b_post_sft.json' \
--annotators_config 'alpaca_eval_gpt4_turbo_fn' \
--output_path './evaluation/test'


