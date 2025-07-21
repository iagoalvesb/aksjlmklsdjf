from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import list_datasets
import pandas as pd
import torch
import argparse
import os
import json
from accelerate import Accelerator
from utils import clean_index_columns
from UTILS_BENCHMARKS import BENCHMARKS_INFORMATIONS
import logging
from logger_config import init_logger
import re

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompts_path",
    type=str,
    required=True,
    help="Huggingface path to the prompt dataset [dataset must be in the format aquired by generate_prompts.sh]"
)

parser.add_argument(
    "--answers_path",
    type=str,
    required=True,
    help="Full Huggingface path to save models outputs"
)

parser.add_argument(
    "--model_path",
    type=str,
    nargs='+',
    required=True,
    help="Huggingface path to the models"
)

parser.add_argument(
    "--use_accelerate",
    action="store_true",
    help="Use Accelerate for multi-GPU inference"
)

parser.add_argument(
    "--use_flash_attention",
    action="store_true",
    help="Enable Flash Attention 2 for faster inference"
)

parser.add_argument(
    "--run_local",
    action="store_true",
    help="If set, read/save results locally as CSV instead of using HuggingFace Hub"
)

args = parser.parse_args()

init_logger()

if args.use_accelerate:
    accelerator = Accelerator()
    device = accelerator.device
    logging.info(f"Using Accelerate with device: {device}")
else:
    accelerator = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Usando single device: {device}")

def get_prompt_for_model(prompt_json_str, model_path):
    try:
        prompt_data = json.loads(prompt_json_str)
        for tokenizer_path, tokenizer_info in prompt_data.items():
            if model_path in tokenizer_info['models']:
                return tokenizer_info['prompt']
        
        raise ValueError(f"Model {model_path} not found in prompt data")
    except json.JSONDecodeError:
        raise ValueError(f"Prompt is not a valid JSON: {prompt_json_str}")

def generate_answer(prompt_text):
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=4,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=attention_mask
    )
    # vamos pegar apenas os tokens da resposta (ou seja, descartamos os tokens do input)
    num_tokens_prompt = input_ids.shape[1]
    generated_text = tokenizer.decode(output_ids[0][num_tokens_prompt:], skip_special_tokens=True)
    return generated_text

def parse_yes_no(text):
    text_lower = text.lower()
    if "sim" in text_lower:
        return "1"
    elif "não" in text_lower:
        return "0"

    text = text.strip()[0].upper()
    answer = 1 if text == 'S' else 0 if text == 'N' else None
    if answer is None:
        return None
    return str(answer)

def parse_multiple_choice(text):
    # Extract the first character of the answer
    text = text.strip()[0].upper()
    return text[0]

def parse_multiple_choice_full_word(text):
    # Extract the first word of the answer
    stripped_text = text.strip()
    words = stripped_text.split()
    first_word = words[0].capitalize() # Capitalize the first word just in case
    first_word = words[0].rstrip('.,!?;:').capitalize()
    return first_word

def parse_continue_value(text):
    # Extract the first character of the answer
    text = text.strip()
    match = re.search(r"\d+([\.,]?\d+)?", text)
    text = match.group()
    text = text.replace(',', '.')
    return text

def parse_answer(example):
    # Extract the answer in the correct format (e.g. anser "Resposta: E" to "E")
    benchmark_name = example['benchmark']
    benchmark = BENCHMARKS_INFORMATIONS[benchmark_name]
    try:
        if benchmark.answer_pattern == "yes_no":
            return parse_yes_no(example['model_answer'])
        elif benchmark.answer_pattern == "multiple_choice":
            return parse_multiple_choice(example['model_answer'])
        elif benchmark.answer_pattern == "multiple_choice_full_word":
            return parse_multiple_choice_full_word(example['model_answer'])
        elif benchmark.answer_pattern == "continue_value":
            return parse_continue_value(example['model_answer'])
        else:
            raise ValueError(f"Unknown answer pattern: {benchmark.answer_pattern}")
    except:
        return None

def map_answer(example, model_path):
    actual_prompt = get_prompt_for_model(example['prompt'], model_path)
    model_answer = generate_answer(actual_prompt)
    example['model_answer'] = model_answer
    example['parsed_model_answer'] = parse_answer(example)
    example['prompt'] = actual_prompt
    return example

def get_available_benchmarks():
    """Get all benchmarks from the prompts file"""
    if args.run_local:
        filename = args.prompts_path.replace("/", "_").replace("-", "_") + ".csv"
        filepath = os.path.join("eval_processing", filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prompts file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        return sorted(df['benchmark'].unique())
    else:
        dataset = load_dataset(args.prompts_path, split='train')
        return sorted(list(set(dataset['benchmark']))) 

def load_benchmark_data(benchmark_name):
    """Load data for a specific benchmark"""
    if args.run_local:
        filename = args.prompts_path.replace("/", "_").replace("-", "_") + ".csv"
        filepath = os.path.join("eval_processing", filename)
        
        df = pd.read_csv(filepath)
        df = clean_index_columns(df)
        benchmark_df = df[df['benchmark'] == benchmark_name]
        dataset = Dataset.from_pandas(benchmark_df)
        logging.info(f"Loaded {len(dataset)} examples for {benchmark_name}")
        return dataset
    else:
        dataset = load_dataset(args.prompts_path, split='train')
        benchmark_dataset = dataset.filter(lambda x: x['benchmark'] == benchmark_name)
        logging.info(f"Loaded {len(benchmark_dataset)} examples for {benchmark_name}")
        return benchmark_dataset

def determine_benchmarks_to_process(model_name, available_benchmarks):
    """Determine which benchmarks to process based on existing results"""
    
    if args.run_local:
        answers_filename = args.answers_path.replace("/", "_").replace("-", "_") + ".csv"
        answers_filepath = os.path.join("eval_processing", answers_filename)
        
        # No answers file exists - start from scratch
        if not os.path.exists(answers_filepath):
            logging.info("No answers file found - starting from scratch")
            return available_benchmarks, []
        
        # Load existing answers
        df = pd.read_csv(answers_filepath)
        df = clean_index_columns(df)
        
        # Get benchmarks for this model
        model_df = df[df['model_name'] == model_name]
        existing_benchmarks = set(model_df['benchmark'].unique())
        
    else:
        # Check if dataset exists
        possible_datasets = list_datasets(search=args.answers_path)
        dataset_exists = any(ds.id == args.answers_path for ds in possible_datasets)
        
        # No answers dataset exists - start from scratch
        if not dataset_exists:
            logging.info("No answers dataset found - starting from scratch")
            return available_benchmarks, []
        
        # Load existing dataset
        for attempt in range(3):
            try:
                dataset = load_dataset(args.answers_path, split='train')
                break
            except Exception as e:
                if attempt == 2:
                    raise Exception(f"Failed to load existing dataset after 3 attempts: {e}")
                logging.info(f"Attempt {attempt + 1} failed, retrying...")
        
        # Get benchmarks for this model
        model_data = dataset.filter(lambda x: x['model_name'] == model_name)
        existing_benchmarks = set(model_data['benchmark'])
    
    # Convert to sets for the subtraction operation
    missing_benchmarks = set(available_benchmarks) - existing_benchmarks
    
    if len(missing_benchmarks) == 0:
        # All benchmarks present - start from scratch
        logging.info(f"All benchmarks present for {model_name} - starting from scratch")
        return available_benchmarks, []
    elif len(missing_benchmarks) < len(available_benchmarks):
        # Some benchmarks missing - continue from missing ones (preserve order)
        missing_ordered = [b for b in available_benchmarks if b in missing_benchmarks]
        existing_ordered = [b for b in available_benchmarks if b in existing_benchmarks]
        logging.info(f"Missing benchmarks for {model_name}: {missing_ordered} - continuing from these")
        return missing_ordered, existing_ordered
    else:
        # No benchmarks present - start from scratch
        logging.info(f"No benchmarks found for {model_name} - starting from scratch")
        return available_benchmarks, []

# Get all available benchmarks from prompts
available_benchmarks = get_available_benchmarks()
logging.info(f"Available benchmarks: {sorted(available_benchmarks)}")

for model_path in args.model_path:
    logging.info(f"\n** RUNNING MODEL: {model_path}")
    model_name = model_path.split("/")[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.use_accelerate:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if args.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        model = accelerator.prepare(model)
    else:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
        if args.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )

    # Determine which benchmarks to process
    benchmarks_to_process, existing_benchmarks = determine_benchmarks_to_process(model_name, available_benchmarks)
    
    all_datasets = []
    
    # ALWAYS load existing data for OTHER models (regardless of start from scratch or continue)
    if args.run_local:
        answers_filename = args.answers_path.replace("/", "_").replace("-", "_") + ".csv"
        answers_filepath = os.path.join("eval_processing", answers_filename)
        
        if os.path.exists(answers_filepath):
            df = pd.read_csv(answers_filepath)
            df = clean_index_columns(df)
            
            if existing_benchmarks:
                # CONTINUING: Keep existing data for this model (only for benchmarks not being reprocessed)
                keep_df = df[
                    (df['model_name'] == model_name) & 
                    (df['benchmark'].isin(existing_benchmarks))
                ]
                # Keep data for other models
                other_models_df = df[df['model_name'] != model_name]
                
                # Combine
                if not keep_df.empty or not other_models_df.empty:
                    combined_df = pd.concat([other_models_df, keep_df], ignore_index=True)
                    if not combined_df.empty:
                        existing_dataset = Dataset.from_pandas(combined_df)
                        all_datasets.append(existing_dataset)
            else:
                # STARTING FROM SCRATCH: Only keep data for other models
                other_models_df = df[df['model_name'] != model_name]
                if not other_models_df.empty:
                    existing_dataset = Dataset.from_pandas(other_models_df)
                    all_datasets.append(existing_dataset)
    else:
        # Similar logic for HuggingFace Hub
        possible_datasets = list_datasets(search=args.answers_path)
        dataset_exists = any(ds.id == args.answers_path for ds in possible_datasets)
        
        if dataset_exists:
            for attempt in range(3):
                try:
                    dataset = load_dataset(args.answers_path, split='train')
                    break
                except Exception as e:
                    if attempt == 2:
                        raise Exception(f"Failed to load existing dataset after 3 attempts: {e}")
                    logging.info(f"Attempt {attempt + 1} failed, retrying...")
            
            if existing_benchmarks:
                # CONTINUING: Keep existing data for this model (only benchmarks not being reprocessed)
                keep_data = dataset.filter(
                    lambda x: x['model_name'] == model_name and x['benchmark'] in existing_benchmarks
                )
                # Keep data for other models
                other_models_data = dataset.filter(lambda x: x['model_name'] != model_name)
                
                # Add to datasets list
                if len(keep_data) > 0:
                    all_datasets.append(keep_data)
                if len(other_models_data) > 0:
                    all_datasets.append(other_models_data)
            else:
                # STARTING FROM SCRATCH: Only keep data for other models
                other_models_data = dataset.filter(lambda x: x['model_name'] != model_name)
                if len(other_models_data) > 0:
                    all_datasets.append(other_models_data)

    # Process each benchmark individually
    for benchmark_name in benchmarks_to_process:
        logging.info(f"*** Processing benchmark: {benchmark_name}")
        
        # Load only this benchmark's data
        benchmark_dataset = load_benchmark_data(benchmark_name)
        
        # Add model name and process answers
        benchmark_dataset = benchmark_dataset.map(lambda example: {"model_name": model_name})
        benchmark_dataset = benchmark_dataset.map(
            lambda example: map_answer(example, model_path), 
            desc=f"{model_name} - {benchmark_name}"
        )
        
        all_datasets.append(benchmark_dataset)
        
        # Clear some memory after each benchmark
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Concatenate and save results
    if all_datasets:
        full_dataset = concatenate_datasets(all_datasets)

        full_dataset = full_dataset.map(
            lambda example, idx: {**example, "id": idx + 1},
            with_indices=True
        )

        if args.run_local:
            os.makedirs("eval_processing", exist_ok=True)
            answers_filename = args.answers_path.replace("/", "_").replace("-", "_") + ".csv"
            answers_filepath = os.path.join("eval_processing", answers_filename)
            
            final_df = full_dataset.to_pandas()
            final_df.to_csv(answers_filepath, index=False)
            logging.info(f"**SAVED MODEL {model_name} AT: {answers_filepath}")
        else:
            full_dataset.push_to_hub(args.answers_path)
            logging.info(f"**SAVED MODEL {model_name} AT: {args.answers_path}")
    else:
        logging.info(f"**NO BENCHMARKS TO PROCESS FOR MODEL {model_name}")

    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if args.run_local:
    answers_filename = args.answers_path.replace("/", "_").replace("-", "_") + ".csv"
    answers_filepath = os.path.join("eval_processing", answers_filename)
    logging.info(f"**ALL MODELS LOCALLY SAVED AT: {answers_filepath}")
else:
    logging.info(f"**ALL MODELS SAVED AT: {args.answers_path}")