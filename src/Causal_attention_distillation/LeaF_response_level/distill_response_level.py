import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import sys
from typing import List
import math
import torch.nn as nn
import torch.nn.functional as F
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  
from datasets import load_dataset
import warnings
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_inverse_sqrt_schedule
import numpy as np
import random
import wandb
wandb.login()
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter



def train(
    # model/data params
    base_model: str = "", 
    teacher_model: str = "", 
    data_path: str = "",
    valid_data_path: str = "", # usually no inst desp data
    output_dir: str = "output_path",
    padding: str = None,
    # training hyperparams
    seed: int = 1234,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 20,
    temperature: int = 1,
    distill_loss_type: str = 'KL', # can be chosen from [entropy, KL]
    distill_from_hidden_states: bool = False,  # whether ditill the hidden states
    hidden_beta: float = 10.0,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    teacher_resume_from_checkpoint: str = None,  
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Distilling Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"teacher_model: {teacher_model}\n"
            f"data_path: {data_path}\n"
            f"valid_data_path: {valid_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    assert (
        teacher_model
    ), "Please specify a --teacher_model, e.g. --teacher_model='alpaca-lora'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 初始化 wandb yiju-zeroyre
    wandb.init(project=wandb_project, entity='yiju-zeroyre', name=wandb_run_name,config={
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'distill_loss_type': distill_loss_type,
        'temperature': temperature,
        'hidden_beta': hidden_beta,
    })

    prompter = Prompter(prompt_template_name)
    if padding is None:
        padding = False
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    
    device = torch.cuda.current_device()
    
    # load model and tokenizer
    def load_model_and_tokenizer(model_path, teacher=False, local_resume_from_checkpoint=None):

        tokenizer = AutoTokenizer.from_pretrained(model_path)  # 构建Tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            # torch_dtype=torch.bfloat16, 
            device_map="auto",  
        )  

        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference
        
        return model, tokenizer

    def tokenize(tokenizer, prompt, add_eos_token=True, padding=False):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        if not add_eos_token:
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=padding,
                return_tensors=None,
            )
        ###########
        else:
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len - 1,
                padding=padding,
                return_tensors=None,
            )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["step_response"],
            data_point["output"],
        )
       
        tokenized_full_prompt = tokenize(tokenizer, full_prompt, add_eos_token=add_eos_token, padding=padding)
        if not train_on_inputs:
            if padding is False:
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point["input"],data_point["step_response"]
                )
                tokenized_user_prompt = tokenize(
                    tokenizer, user_prompt, add_eos_token=add_eos_token
                )
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                if add_eos_token:
                    user_prompt_len -= 1

                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]  # could be sped up, probably
            else:
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point["input"],data_point["step_response"]
                )
                tokenized_user_prompt = tokenize(
                    tokenizer, user_prompt, add_eos_token=add_eos_token
                )
                tokenized_full_prompt_without_padding = tokenize(tokenizer, full_prompt, add_eos_token=add_eos_token)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                full_prompt_len = len(tokenized_full_prompt_without_padding["input_ids"])
                if add_eos_token:
                    user_prompt_len -= 1
                output_len = full_prompt_len - user_prompt_len
                # tokenized_full_prompt["labels"] = [
                #     -100
                # ] * user_prompt_len + tokenized_full_prompt["labels"][
                #     user_prompt_len:
                # ]  
                tokenized_full_prompt["labels"] = [
                    -100
                ] * (cutoff_len - output_len) + tokenized_full_prompt["labels"][
                    (cutoff_len - output_len):
                ]  
                
        return tokenized_full_prompt
    
    def generate_and_tokenize_prompt_teacher(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["step_response"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(teacher_tokenizer, full_prompt, add_eos_token=add_eos_token, padding=padding)
        if not train_on_inputs:
            if padding is False:
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point["input"],data_point["step_response"]
                )
                tokenized_user_prompt = tokenize(
                    teacher_tokenizer, user_prompt, add_eos_token=add_eos_token
                )
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                if add_eos_token:
                    user_prompt_len -= 1

                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]  # could be sped up, probably
            else:
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point["input"],data_point["step_response"]
                )
                tokenized_user_prompt = tokenize(
                    teacher_tokenizer, user_prompt, add_eos_token=add_eos_token
                )
                tokenized_full_prompt_without_padding = tokenize(teacher_tokenizer, full_prompt, add_eos_token=add_eos_token)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                full_prompt_len = len(tokenized_full_prompt_without_padding["input_ids"])
                if add_eos_token:
                    user_prompt_len -= 1
                output_len = full_prompt_len - user_prompt_len
                # tokenized_full_prompt["labels"] = [
                #     -100
                # ] * user_prompt_len + tokenized_full_prompt["labels"][
                #     user_prompt_len:
                # ]  
                tokenized_full_prompt["labels"] = [
                    -100
                ] * (cutoff_len - output_len) + tokenized_full_prompt["labels"][
                    (cutoff_len - output_len):
                ]  
                
        return tokenized_full_prompt

    model, tokenizer = load_model_and_tokenizer(base_model, False, resume_from_checkpoint)
    teacher_model, teacher_tokenizer = load_model_and_tokenizer(teacher_model, True, teacher_resume_from_checkpoint)
    # 获取当前设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # 确保模型在设备上
    # model = model.to(device)
    # teacher_model = teacher_model.to(device)
    def load_train_and_val_data(train_data_path, valid_data_path=None, seed=1234):
        if train_data_path.endswith(".json") or train_data_path.endswith(".jsonl"):
            train_data = load_dataset("json", data_files=train_data_path)
        else:
            train_data = load_dataset(train_data_path)
        valid_data = None
        if valid_data_path is not None:
            if valid_data_path.endswith(".json") or valid_data_path.endswith(".jsonl"):
                valid_data = load_dataset("json", data_files=valid_data_path)
            else:
                valid_data = load_dataset(valid_data_path)
        
        # train_data = train_data["train"].map(generate_and_tokenize_prompt)
        #train_data = generate_and_tokenize_prompt(tokenizer, train_data["train"])
        val_data = None
        if valid_data is not None:
            train_data = train_data["train"].shuffle(seed).map(generate_and_tokenize_prompt)
            print("sampling valid data...")
            if val_set_size >= len(valid_data["train"]):
                val_data = valid_data["train"].map(generate_and_tokenize_prompt)
            else:
                train_val = valid_data["train"].train_test_split(
                    test_size=val_set_size, shuffle=True, seed=seed
                )
                val_data = (
                    train_val["test"].map(generate_and_tokenize_prompt)
                )
            # val_data = valid_data["train"].map(generate_and_tokenize_prompt)
            # val_data = generate_and_tokenize_prompt(tokenizer, valid_data["train"])
        else:
            train_data = train_data["train"].shuffle(seed).map(generate_and_tokenize_prompt_teacher)
        return train_data, val_data
    
    teacher_train_data, _ = load_train_and_val_data(data_path, seed=seed)
    student_train_data, student_val_data = load_train_and_val_data(data_path, valid_data_path, seed=seed)
    
    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
        teacher_model.is_parallelizable = True
        teacher_model.model_parallel = True
    
    teacher_model.eval()
    
    def get_optimizer_params(model: nn.Module):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'ln_f.weight', 'ln_1.weight', 'ln_2.weight', 'ln_cross_attn']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)]},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters
    
    def get_learning_rate_scheduler(lr_scheduler_name, optimizer, total_iters, warmup_steps=0):
        if lr_scheduler_name == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps)
        elif lr_scheduler_name == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_iters)
        elif lr_scheduler_name == "noam":
            lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_iters,
                power=0.5)
        elif lr_scheduler_name == "inverse_sqrt":
            lr_scheduler = get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=warmup_steps)
        else:
            raise ValueError(f"lr_scheduler of type {lr_scheduler_name} is not supported yet.")

        return lr_scheduler

    optimizer = AdamW(get_optimizer_params(model), lr=learning_rate, weight_decay=0)
    
    num_steps = math.ceil(len(teacher_train_data) / micro_batch_size)
    num_update_steps_per_epoch = math.ceil(len(teacher_train_data) / batch_size)
    num_training_steps = num_epochs * (math.ceil(len(teacher_train_data) / batch_size))
    num_eval_steps = math.ceil(len(student_val_data) / micro_batch_size)
    lr_scheduler = get_learning_rate_scheduler(lr_scheduler, optimizer, num_training_steps, warmup_steps)
    
    print("################ Start Distilling ##############")
    grad_steps = 0
    print("beta: ", hidden_beta)
    model.train()
    model.zero_grad()
    total_loss = 0
    update_loss = 0
    hidden_mse_loss = 0
    best_loss = 10000000
    for epoch in range(num_epochs):
        grad_steps = 0
        num_updates_this_epoch = 0
        for step in range(num_steps):
            grad_steps += 1
            teacher_train_batch = teacher_train_data[step * micro_batch_size: min((step + 1) * micro_batch_size, len(teacher_train_data))]
            student_train_batch = student_train_data[step * micro_batch_size: min((step + 1) * micro_batch_size, len(student_train_data))]
            
            
            outputs_model = model(input_ids=torch.LongTensor(student_train_batch["input_ids"]).to(device),
                                  attention_mask=torch.LongTensor(student_train_batch["attention_mask"]).to(device),
                                  labels=torch.LongTensor(student_train_batch["labels"]).to(device),
                                  use_cache=False,
                                  output_hidden_states=distill_from_hidden_states)
            model_logits_masks = (torch.tensor(student_train_batch["labels"]) > -100).to(device)
            model_output_logits = outputs_model["logits"]# [model_logits_masks]
            with torch.no_grad():
                teacher_model.eval()
                outputs_teacher = teacher_model(input_ids=torch.LongTensor(teacher_train_batch["input_ids"]).to(device),
                                            attention_mask=torch.LongTensor(teacher_train_batch["attention_mask"]).to(device),
                                            labels=torch.LongTensor(teacher_train_batch["labels"]).to(device),
                                            use_cache=False,
                                            output_hidden_states=distill_from_hidden_states)
                teacher_logits_masks = torch.tensor(teacher_train_batch["labels"]) > -100
                teacher_output_logits = outputs_teacher["logits"]# [teacher_logits_masks]

            # 如果教师模型和学生模型输出形状不匹配，对学生模型进行padding或对教师模型进行截断
            if model_output_logits.shape[-1] != teacher_output_logits.shape[-1]:               
                teacher_output_logits = teacher_output_logits[:, :, :model_output_logits.shape[-1]]
                
            # calculate distill loss
            # inf_mask = torch.isinf(model_output_logits)
            teacher_probs = F.softmax(teacher_output_logits / temperature, dim=-1, dtype=torch.float32)
            logprobs = F.log_softmax(model_output_logits / temperature, dim=-1, dtype=torch.float32)
            if distill_loss_type == 'Entropy':
                # prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
                prod_probs = teacher_probs * logprobs
                prod_probs = torch.sum(prod_probs, dim=-1).view(-1)
                distil_loss = - torch.sum(prod_probs * model_logits_masks.view(-1), dim=0) / torch.sum(model_logits_masks.view(-1), dim=0) #- torch.sum(prod_probs, dim=0) / len(prod_probs)
                distil_loss = distil_loss * temperature * temperature
            elif distill_loss_type == 'KL':
                prod_probs = teacher_probs * torch.log(teacher_probs) - teacher_probs * logprobs
                prod_probs = torch.sum(prod_probs, dim=-1).view(-1)
                
                prod_probs = torch.nan_to_num(prod_probs)
                
                distil_loss = torch.sum(prod_probs * model_logits_masks.view(-1), dim=0) / torch.sum(model_logits_masks.view(-1), dim=0) #- torch.sum(prod_probs, dim=0) / len(prod_probs)
                distil_loss = distil_loss * temperature * temperature
            else:
                print("Not implemented loss")
                assert 0==1
            
            # whether distill from hidden states, now set the distillation layers to be all the layers
            if distill_from_hidden_states:
                model_hidden_states = torch.stack(outputs_model['hidden_states'][: -1]).transpose(0, 1).transpose(1, 2)
                teacher_hidden_states = torch.stack(outputs_teacher['hidden_states'][: -1]).transpose(0, 1).transpose(1, 2)

                # when calculate mse loss, first normalize the hidden states
                model_hidden_states = F.normalize(model_hidden_states, p=2, dim=3)
                teacher_hidden_states = F.normalize(teacher_hidden_states, p=2, dim=3)
                # reshape for masking [bsz, seq_len, n_layers, n_dim] -> [bsz, seq_len, -1]
                model_hidden_states = model_hidden_states.reshape(model_hidden_states.shape[0], model_hidden_states.shape[1], -1)
                teacher_hidden_states = teacher_hidden_states.reshape(teacher_hidden_states.shape[0], teacher_hidden_states.shape[1], -1)
                
                # hidden_mse = F.mse_loss(teacher_hidden_states.float(), model_hidden_states.float()).half()
                hidden_mse = torch.mean((teacher_hidden_states - model_hidden_states) ** 2, dim=-1)
                hidden_mse = torch.sum(hidden_mse.view(-1) * model_logits_masks.view(-1), dim=0) / torch.sum(model_logits_masks.view(-1), dim=0)
                hidden_mse *= hidden_beta
                hidden_mse_loss += hidden_mse.item()
                distil_loss += hidden_mse
            if num_updates_this_epoch == (num_update_steps_per_epoch - 1):
                current_gradient_accumulation_steps = len(teacher_train_data) - batch_size * num_updates_this_epoch
            else:
                current_gradient_accumulation_steps = gradient_accumulation_steps
            distil_loss /= current_gradient_accumulation_steps
            distil_loss.backward()
            update_loss += distil_loss.item()
            total_loss += distil_loss.item()
   
            if (grad_steps % gradient_accumulation_steps == 0) or (grad_steps == num_steps):
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

                # Log training metrics to wandb
                wandb.log({
                    'epoch': epoch + 1,
                    'step': grad_steps + num_steps * epoch,
                    'iter_loss': update_loss,
                    'hidden_mse_loss': hidden_mse_loss / current_gradient_accumulation_steps,
                    'lr': lr_scheduler.get_last_lr()[0],
                    'total_loss': total_loss,
                })

                print("Train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | iter_loss: {:.4f} | hidden_mse_loss: {:.4f} | lr: {:.4e} |".format(
                    (epoch + 1),
                    (step + 1),
                    num_steps,
                    grad_steps + num_steps * epoch,
                    num_steps * num_epochs,
                    update_loss,
                    hidden_mse_loss / current_gradient_accumulation_steps,
                    lr_scheduler.get_last_lr()[0],
                    )
                )
                update_loss = 0
                hidden_mse_loss = 0
                num_updates_this_epoch += 1
        # eval
        current_eval_loss = 0
        model.eval()
        with torch.no_grad():
            for eval_step in range(num_eval_steps):
                no_desp_val_batch = student_val_data[eval_step * micro_batch_size: min((eval_step + 1) * micro_batch_size, len(student_val_data))]
            
                outputs_eval = model(input_ids=torch.LongTensor(no_desp_val_batch["input_ids"]).to(device),
                                    attention_mask=torch.LongTensor(no_desp_val_batch["attention_mask"]).to(device),
                                    labels=torch.LongTensor(no_desp_val_batch["labels"]).to(device),
                                    use_cache=False,
                                    output_hidden_states=False)
                current_eval_loss += (outputs_eval['loss'].item() * micro_batch_size)
            current_eval_loss /= len(student_val_data)

            # Log evaluation loss to wandb
            wandb.log({
                'eval_loss': current_eval_loss,
                'best_loss': best_loss
            })

            checkpoint_dir = f"{output_dir}/checkpoint-epoch-{epoch+1}"
            print(f"Saving checkpoint for epoch {epoch+1} to {checkpoint_dir}...")
            model.save_pretrained(checkpoint_dir)

            if current_eval_loss < best_loss:
                print("Saving best model...")
                model.save_pretrained(output_dir)
                best_loss = current_eval_loss

            print("Train | epoch {:3d} | eval_loss: {:.4f} | best_loss: {:.4f} |".format(
                    (epoch + 1), current_eval_loss, best_loss))


        model.train()
        model.zero_grad()            
        # model.train()
        # model.zero_grad()
    
    # model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
