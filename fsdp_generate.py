import functools

import torch
import torch.distributed
import torch.distributed.fsdp
import torch.distributed.fsdp.wrap
import transformers
import transformers.models.gpt_neo.modeling_gpt_neo


def main() -> None:
    torch.distributed.init_process_group(world_size=2)
    device = torch.device(torch.distributed.get_rank())
    torch.cuda.set_device(device)

    pretrained_model_name_or_path = "../../../models/EleutherAI/pythia-2.8b"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    fsdp_model = torch.distributed.fsdp.FullyShardedDataParallel(
        model,
        auto_wrap_policy=functools.partial(
            torch.distributed.fsdp.wrap.transformer_auto_wrap_policy,
            transformer_layer_cls={
                transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoBlock
            },
        ),
        limit_all_gathers=True,
        use_orig_params=True,  # required to overcome the error "The tensor has a non-zero number of elements, but its data is not allocated yet" ... PreTrainedModel.generate is probably using some torch.compile-wrapped function
    )

    data_by_rank = {  # differently-sized causes FSDP to hang
        0: "Hello world!",
        1: "The quick brown fox jumps over the lazy dog."
    }

    batch = tokenizer(
        data_by_rank[torch.distributed.get_rank()],
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(fsdp_model):  # required to overcome to the error "'weight' must be 2-D"
        generated_text = fsdp_model.module.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=20,
            synced_gpus=True,  # True is required to use differently sized data with FSDP + generate (current default is False)
        )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
