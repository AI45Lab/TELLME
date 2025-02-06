from functools import partial
import logging
import os
import sys
import json
import gc
import atexit
import numpy as np
import wandb
import torch.distributed as dist
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, TaskType
import transformers
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

print(sys.path)
print(src_dir)
from src.loss import compute_retain_loss, clip_loss, nt_xent_loss, cal_sim_loss

from src.datasets_config import SplitSet
from utils import save_model_and_tokenizer
from args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    import torch.nn.functional as F
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    import torch.nn.functional as F
    p = F.softmax(logits, dim=2)

    if not gather:
        return p
    py = torch.gather(p, 2, labels.unsqueeze(2)).squeeze(-1)
    return py

def compute_loss(self, model, inputs, target_layers, tracker, control_rate=1, return_outputs=False, wo_kl=False, tokens=[-1], nt_xent=False, **kwargs):

    self.current_training_step += 1

    coeff = 10
    mask  = coeff

    # === retain ===
    retain_input_ids = inputs.get(f"input_ids")
    retain_attention_mask = inputs.get(f"attention_mask")
    # ==== value split ====     
    input_ids_view1 = inputs.get(f"input_ids_view1")
    attention_mask_view1 = inputs.get(f"attention_mask_view1")

    input_ids_view2 = inputs.get("input_ids_view2")
    attention_mask_view2 = inputs.get("attention_mask_view2")

    # ==== behavior retain ====
    attention_mask_kl_sample = inputs.get("attention_mask_kl_sample")
    input_ids_kl_sample = inputs.get("input_ids_kl_sample")

    module = 'hidden_states'
    retain_inputs = dict(input_ids=retain_input_ids, attention_mask=retain_attention_mask, output_hidden_states=True)
    view_1_inputs = dict(input_ids=input_ids_view1, attention_mask=attention_mask_view1, output_hidden_states=True)
    view_2_inputs = dict(input_ids=input_ids_view2, attention_mask=attention_mask_view2, output_hidden_states=True)
    kl_sample = dict(input_ids=input_ids_kl_sample, attention_mask=attention_mask_kl_sample)

    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
                
            orig_retain_outputs = model(**retain_inputs)[module]
            orig_retain_hidden = torch.stack(orig_retain_outputs).detach()
            layers_retain_attention_mask = retain_attention_mask.repeat(len(orig_retain_outputs), 1, 1).unsqueeze(-1)
            orig_retain_hidden *= layers_retain_attention_mask
            orig_retain_hidden = torch.stack([out for out in orig_retain_outputs], dim=1)

            del orig_retain_outputs
            gc.collect()


            outputs = model(**kl_sample)
            logits = outputs.logits
            old_prob = probs_from_logits(logits[:, :-1, :], input_ids_kl_sample[:, 1:]).detach()
            del outputs
            gc.collect()

    model.train()

    retain_outputs = model(**retain_inputs)["hidden_states"]
    retain_embedding = torch.stack(retain_outputs)  
    retain_embedding *= layers_retain_attention_mask
    retain_embedding = torch.stack([out for out in retain_embedding], dim=1)    

    value_lst = 0
    same_lst = 0
    diff_lst = 0
    
    for l in target_layers:
        cache = []
        def hook(module, input, output):
            if isinstance(output, tuple):
                cache.append(output[0])
            else:
                cache.append(output)
            return None
        
        hook_handle = model.module.model.model.layers[l].register_forward_hook(hook)
        _ = model(**view_1_inputs)
        _ = model(**view_2_inputs)
        for token in tokens:
            lora_view1_outputs = cache[0][:, token, :]
            lora_view2_outputs = cache[1][:, token, :]

            hook_handle.remove()

            if nt_xent:
                value_loss = nt_xent_loss(lora_view1_outputs, lora_view2_outputs)
                mean_same, mean_diff = cal_sim_loss(lora_view1_outputs, lora_view2_outputs, mask)
            else:
                value_loss, mean_same, mean_diff = clip_loss(lora_view1_outputs, lora_view2_outputs, mask)

            value_lst += value_loss
            same_lst += mean_same
            diff_lst += mean_diff


    scale = len(target_layers) * len(tokens)

    value_loss, mean_same, mean_diff = value_lst / scale, same_lst / scale, diff_lst / scale

    lora_outputs = model(**kl_sample)
    lora_logits = lora_outputs.logits
    logprob = logprobs_from_logits(lora_logits[:, :-1, :], input_ids_kl_sample[:, 1:])

    kl_sum_logprob = (old_prob * logprob).sum(dim=1)

    retain_embs = retain_embedding.flatten(start_dim=1)
    orig_retain_embs = orig_retain_hidden.flatten(start_dim=1)

    retain_loss = control_rate * compute_retain_loss(retain_embs, orig_retain_embs) / 100

    kl_penalty = - torch.mean(kl_sum_logprob)


    if wo_kl:
        loss = retain_loss + value_loss * 10
    else:
        loss = retain_loss + value_loss * 10 + kl_penalty

    if dist.get_rank() == 0:
        if tracker:
            tracker.log({
            "loss": loss.item(),
            "retain_loss": retain_loss.item(),
            "value_loss": value_loss.item(),
            "kl_penalty": kl_penalty,
            "mean_same": mean_same,
            "mean_diff": mean_diff
            })

    return (loss, ) if return_outputs else loss

def data_collator(batch_list):
    batch_inputs = {}
    for features in batch_list:
        for k, input in features.items():
            batch_inputs.setdefault(k , []).append(input)
    
    for k, inputs in batch_inputs.items():
        if isinstance(inputs[0], torch.Tensor):
            batch_inputs[k] = torch.cat(inputs, dim=0)
        elif isinstance(inputs[0], int):
            batch_inputs[k] = torch.tensor(inputs)
        else:
            raise ValueError(f"Return data type not implemented {type(inputs[0])}")
    return batch_inputs

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
        lorra_args,
    ) = parser.parse_args_into_dataclasses()

    print(lorra_args.to_dict())
    print(lora_args)
    print(model_args)
    print(training_args)

    model_name_or_path = model_args.model_name_or_path
    target_layers = lorra_args.target_layers
    transform_layers = lorra_args.transform_layers
    
    run_name = str(training_args.output_dir).split("/")[-1]

    if dist.get_rank() == 0:
        # os.environ["WANDB_API_KEY"] = "key"
        # os.environ["WANDB_MODE"] = "mode"
        # tracker = wandb.init(
        #     project="project",
        #     entity="entity",
        #     name=run_name,
        #     )
        os.environ["WANDB_API_KEY"] = "04be7c36e53539906bdfcaae3f1a3d8960add0cb"
        os.environ["WANDB_MODE"] = "online"
        tracker = wandb.init(
            project="open_test",
            entity="quantumcgx-sjtu",
            name=run_name,
            )
        # tracker = None
    else:
        tracker = None

    lorra_target_layers = [int(layer) for layer in target_layers.split(",")]
    if "-1" in transform_layers:
        lora_layers_to_transform = [i for i in range(max(lorra_target_layers) + 1)]
    else:
        lora_layers_to_transform = [int(layer) for layer in transform_layers.split(",")]

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=lora_layers_to_transform,
        task_type=TaskType.CAUSAL_LM,
    )

    print("lorra_transform_layers", lora_layers_to_transform)

    config = AutoConfig.from_pretrained(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    extra_save_kargs = dict(tokenizer=tokenizer)
    save_model_function = save_model_and_tokenizer
    save_model_function = partial(save_model_function,
                    output_dir=training_args.output_dir,
                    **extra_save_kargs)

    print(lora_args.lora_target_modules, lora_layers_to_transform)
    model = get_peft_model(model, lora_config)
    print("model", model)

    if "with_kl" in training_args.output_dir:
        wo_kl = False
    else:
        wo_kl = True
    
    if "-nt" in training_args.output_dir:
        nt_xent = True
    else:
        nt_xent = False

    if "split_half" in training_args.output_dir:
        split_half = True
    else:
        split_half = False
    
    if "both_token" in training_args.output_dir:
        both_token = True
    else:
        both_token = False

    if split_half:
        if "llama" in model_name_or_path.lower():
            if both_token:
                tokens = [512-1, -1]
            else:
                tokens = [512-1]
        elif "qwen" in model_name_or_path.lower():
            if both_token:
                tokens = [512-2, -1]
            else:
                tokens = [512-2]
        elif "gemma" in model_name_or_path.lower():
            if both_token:
                tokens = [512-2, -1]
            else:
                tokens = [512-2]
        elif "mistral" in model_name_or_path.lower():
            if both_token:
                tokens = [512-1, -1]
            else:
                tokens = [512-1]
    else:
        tokens = [-1]

    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    train_dataset = SplitSet(tokenizer, num_examples=training_args.num_examples, model_name_or_path=model_name_or_path, path=training_args.path)
    print("TRAIN LEN: ", len(train_dataset))

    print(training_args.output_dir, " wo_kl: ", wo_kl, " nt_xent:", nt_xent)
    print(f"USING TOKEN {tokens}")


    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.current_training_step = 0
            self.lorra_args = lorra_args
            self.training_args = training_args

        def get_training_progress(self):
            return self.current_training_step / (self.args.max_steps * self.args.gradient_accumulation_steps * 2)

        def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
            return compute_loss(
                self,
                model,
                inputs,
                target_layers=lorra_target_layers, 
                return_outputs=return_outputs,
                tracker = tracker,
                control_rate = (len(lorra_target_layers) / len(lora_layers_to_transform)),
                wo_kl=wo_kl,
                tokens=tokens,
                nt_xent=nt_xent
            )

    training_args.remove_unused_columns = False
    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset, data_collator=data_collator
    )
    model.config.use_cache = False
    atexit.register(save_model_function, model=model)
    trainer.train()

if __name__ == "__main__":
    SEED = 42
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    train()