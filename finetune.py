import os
import sys
from typing import List, Optional, Union

import fire
import torch
import transformers
from datasets import load_dataset
#
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from llama_vert import CustomLlamaForCausalLM, CustomQwenForCausalLM, CustomMistralForCausalLM, CustomGemmaForCausalLM, CustomPhiForCausalLM

import torch.distributed as dist
import wandb

os.environ["WANDB_DISABLED"] = "true"

if wandb.run is None:
    print("WandB not initialized. Disabled.")

def initialize_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        set_seed(42)
        dist.barrier()
    else:
        raise ValueError("DDP is not properly initialized. Please use torchrun to launch the script.")

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def train(
    base_model: str = "togethercomputer/Llama-2-7B-32K-Instruct",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    adapter_name: str = "lora",
    load_8bit: bool = False,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    eval_step: int = 200,
    save_step: int = 200,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    use_moslora: bool = False,
    bottleneck_size: int = 256,
    non_linearity: str = "tanh",
    adapter_dropout: float = 0.0,
    use_parallel_adapter: bool = False,
    use_adapterp: bool = False,
    target_modules: List[str] = None,
    scaling: Union[float, str] = 1.0,
    use_gradient_checkpointing: bool = False,
    num_virtual_tokens: int = 30,
    train_on_inputs: bool = True,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    mi_max: float = 0.05,
    mi_min: float = 0.0001,
    mi_warmup: float = 0.3,
    mi_decay: float = 0.99,
    ad_max: float = 1.0,
    ad_min: float = 0.01,
    ad_warmup: float = 0.0,
    ad_decay: float = 0.99,
    orth_max: float = 0.05,
    orth_min: float = 0.0001,
    orth_warmup: float = 0.3,
    orth_decay: float = 0.5,
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"use_moslora: {use_moslora}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert base_model
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model_kwargs = dict(
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        mi_max=mi_max,
        mi_min=mi_min,
        mi_warmup=mi_warmup,
        mi_decay=mi_decay,
        ad_max=ad_max,
        ad_min=ad_min,
        ad_warmup=ad_warmup,
        ad_decay=ad_decay,
        orth_max=orth_max,
        orth_min=orth_min,
        orth_warmup=orth_warmup,
        orth_decay=orth_decay,
    )

    if "llama" in base_model.lower():
        model = CustomLlamaForCausalLM.from_pretrained(base_model, **model_kwargs)
    elif "mistral" in base_model.lower():
        model = CustomMistralForCausalLM.from_pretrained(base_model, **model_kwargs)
    elif "phi" in base_model.lower():
        model = CustomPhiForCausalLM.from_pretrained(base_model, **model_kwargs)
    elif "gemma" in base_model.lower():
        model = CustomGemmaForCausalLM.from_pretrained(base_model, **model_kwargs)
    elif "qwen" in base_model.lower():
        model = CustomQwenForCausalLM.from_pretrained(base_model, **model_kwargs)
    else:
        raise ValueError(f"Unknown base model type: {base_model}")

    def reinitialize_broken_conv_layers(model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                with torch.no_grad():
                    module.weight = torch.nn.Parameter(torch.empty_like(module.weight, dtype=torch.float32))
                    torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        module.bias = torch.nn.Parameter(torch.zeros_like(module.bias, dtype=torch.float32))
                module.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def reinitialize_linear_layers(model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "shared_adapter" in name:
                with torch.no_grad():
                    module.weight = torch.nn.Parameter(torch.empty_like(module.weight, dtype=torch.float32))
                    torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        module.bias = torch.nn.Parameter(torch.zeros_like(module.bias, dtype=torch.float32))
                module.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def reinitialize_crossattn_linear_layers(model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "shared_adapter" in name:
                with torch.no_grad():
                    module.weight = torch.nn.Parameter(torch.empty_like(module.weight, dtype=torch.float32))
                    torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        module.bias = torch.nn.Parameter(torch.zeros_like(module.bias, dtype=torch.float32))
                module.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    reinitialize_linear_layers(model)
    reinitialize_crossattn_linear_layers(model)
    reinitialize_broken_conv_layers(model)

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    tokenizer = LlamaTokenizer.from_pretrained(base_model) if "llama2" in base_model else AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result if "chatglm" not in base_model else {"input_ids": result["input_ids"], "labels": result["labels"]}

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_use_mixer=use_moslora,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, config)
    for name, param in model.named_parameters():
        if "shared_adapter" in name:
            param.requires_grad = True
            param.data = param.data.float()

    def check_shared_adapter_trainable(model):
        for name, param in model.named_parameters():
            if "shared_adapter" in name:
                status = "Trainable" if param.requires_grad else "Frozen"
                print(f"{name}: {status}")

    check_shared_adapter_trainable(model)

    if adapter_name == "prefix-tuning":
        model.to('cuda')

    data = load_dataset("json", data_files=data_path) if data_path.endswith(".json") else load_dataset(data_path)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    shared_adapter_params = [param for name, param in model.named_parameters() if "shared_adapter" in name]
    other_params = [param for name, param in model.named_parameters() if "shared_adapter" not in name and param.requires_grad]

    optimizer_grouped_parameters = [
        {"params": shared_adapter_params, "lr": 3e-5, "weight_decay": 0.01},
        {"params": other_params, "lr": learning_rate, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-6)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=True if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            max_grad_norm=1.0,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        optimizers=(optimizer, None),
    )

    model.config.use_cache = False

    from math import ceil

    def estimate_total_steps(train_dataset, training_args):
        effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        total_samples = len(train_dataset)
        steps_per_epoch = ceil(total_samples / effective_batch_size)
        total_steps = int(training_args.num_train_epochs * steps_per_epoch)
        return total_steps

    total_steps = estimate_total_steps(trainer.train_dataset, trainer.args)
    model.set_total_training_steps(total_steps)

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    model.state_dict = old_state_dict
    model_sd = model.state_dict()
    custom_weights = {key: val.cpu() for key, val in model_sd.items() if "shared_adapter" in key}
    torch.save(custom_weights, os.path.join(output_dir, "custom_weight.bin"))
    print("shared_adapter weights saved to custom_weight.bin")

    print("If there's a warning about missing keys above, please disregard.")

def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    if dist.is_initialized():
        dist.barrier()

torch.use_deterministic_algorithms(True)
set_seed(42)
cleanup_distributed()

if __name__ == "__main__":
    initialize_distributed()
    set_seed(42)
    fire.Fire(train)
