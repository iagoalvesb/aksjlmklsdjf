from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import list_datasets
import torch
import argparse
from UTILS_BENCHMARKS import BENCHMARKS_INFORMATIONS
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

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    # Extract the first character of the answer
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


def map_answer(example):
    model_answer = generate_answer(example['prompt'])
    example['model_answer'] = model_answer
    example['parsed_model_answer'] = parse_answer(example)
    return example



# check if the dataset already exists in the hub
possible_datasets = list_datasets(search=args.answers_path)
dataset_exists = any(ds.id == args.answers_path for ds in possible_datasets)


for model_path in args.model_path:
    print(f"\n** RUNNING MODEL: {model_path}")
    model_name = model_path.split("/")[-1]
    dataset = load_dataset(args.prompts_path, split='train')
    dataset = dataset.map(lambda example: {"model_name": model_name})

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            low_cpu_mem_usage=True,
                            device_map=device,
                            # attn_implementation = "flash_attention_2",
                        )

    dataset = dataset.map(map_answer, desc=f"{model_path.split('/')[1]}")

    all_datasets = []
    possible_datasets = list_datasets(search=args.answers_path)
    dataset_exists = any(ds.id == args.answers_path for ds in possible_datasets)

    if dataset_exists:
        original_dataset = load_dataset(args.answers_path, split='train')

        current_benchmarks = set(dataset['benchmark'])
        filtered_dataset = original_dataset.filter(
            lambda x: not (x['benchmark'] in current_benchmarks and x['model_name'] == model_name)
        )
        all_datasets.append(filtered_dataset)

    all_datasets.append(dataset)
    full_dataset = concatenate_datasets(all_datasets)

    full_dataset = full_dataset.map(
        lambda example, idx: {**example, "id": idx + 1},
        with_indices=True
    )

    full_dataset.push_to_hub(args.answers_path)
    print(f"\n**SAVED MODEL {model_name} AT: {args.answers_path}")

    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\n\n**ALL MODELS SAVED AT: {args.answers_path}")
