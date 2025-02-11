import os
import json
import requests
import sqlite3
import time
import sys
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Set the seaborn style and font scale
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def custom_annot(val):
    if val == 100:
        return '100'
    else:
        return f'{val:.1f}'

# Constants
API_URL = "https://usechatgpt.space/v1/chat/completions"
MODELS_TO_TEST = ["o3-mini-2025-01-31", "o1-mini-2024-09-12"]  # List of models to test
TEMPERATURE = 1.0  # For deterministic output
TOP_P = 1.0

# Directories
validation_base_dir = '/n/holyscratch01/rush_lab/Users/yuntian/cascade_jun12/data/long_mult_mixed_1_to_20_inter_mar1824_includingzero_padinput_short_alllowdigits_testonly/'  # Update this path to your test datasets
validation_base_dir = '/n/netscratch/shieber_lab/Lab/yuntian/cascade_jun12/data/long_mult_mixed_1_to_20_inter_mar1824_includingzero_padinput_short_alllowdigits_testonly/'  # Update this path to your test datasets

heatmap_dir = 'heatmaps_models'
if not os.path.exists(heatmap_dir):
    os.makedirs(heatmap_dir)

# Get API keys from environment variable
API_KEYS = os.environ.get("OPENAI_API_KEYS")
if API_KEYS is None:
    print("Please set the OPENAI_API_KEYS environment variable.")
    sys.exit(1)
API_KEYS = [key.strip() for key in API_KEYS.split(',')]
api_key_index = 0

def get_next_api_key():
    global api_key_index
    api_key = API_KEYS[api_key_index % len(API_KEYS)]
    api_key_index += 1
    return api_key

# Setup SQLite database
conn = sqlite3.connect('results_models.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT,
    m INTEGER,
    n INTEGER,
    example_id INTEGER,
    input TEXT,
    ground_truth TEXT,
    model_answer TEXT,
    is_correct BOOLEAN,
    usage TEXT,
    model_output TEXT
);
''')
conn.commit()

# Function to check if an example already exists in the database
def example_exists(model_name, m, n, example_id):
    cursor.execute('SELECT 1 FROM results WHERE model=? AND m=? AND n=? AND example_id=?', 
                   (model_name, m, n, example_id))
    return cursor.fetchone() is not None

# Function to process each example
def process_example(model_name, m, n, example_id, number1, number2, ground_truth):
    # Modify the prompt to allow chain-of-thought reasoning
    prompt = f"""Calculate the product of {number1} and {number2}. Please provide the final answer in the format:

Final Answer: [result]"""

    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n": 1,
        "stream": True,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    api_key = get_next_api_key()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    #import pdb; pdb.set_trace()
    def get_response(headers, payload):
        response = requests.post(API_URL, headers=headers, json=payload, stream=True)
        counter = 0
        partial_words = ''
        usage = {}
        for chunk in response.iter_lines():
            if counter == 0:
                counter += 1
            else:
                if chunk.decode():
                    chunk_decoded = chunk.decode()
                    if len(chunk_decoded) > 12:
                        data = json.loads(chunk_decoded[6:])
                        delta = data['choices'][0]['delta']
                        if "content" in delta:
                            id = data['id']
                            created = data['created']
                            model = data['model']
                            partial_words = partial_words + delta["content"]
                            if 'usage' in data:
                                usage = data['usage']
        return partial_words, usage

    found = False
    #import pdb; pdb.set_trace()
    if example_exists(model_name, m, n, example_id):
        cursor.execute('SELECT model_output, usage FROM results WHERE model=? AND m=? AND n=? AND example_id=?',
                       (model_name, m, n, example_id))
        row = cursor.fetchone()
        partial_words = row[0]
        usage_json = row[1]
        usage = json.loads(usage_json) if usage_json else {}
        found = True
    else:
        try:
            partial_words, usage = get_response(headers, payload)
        except Exception as e:
            print (e)
            time.sleep(60)
            partial_words, usage = get_response(headers, payload)

    model_output = partial_words
    # Extract the answer using regex
    #answer_match = re.search(r'Final Answer:\s*(\S+)', model_output)
    answer_match = re.search(r'Final Answer:\s+(\d\S*)', model_output)
    if answer_match:
        model_answer = answer_match.group(1)
    else:
        answer_match = re.search(r'Final Answer.*?(\d\S*)', model_output, re.M)
        if answer_match:
            model_answer = answer_match.group(1)
        else:
            model_answer = None

    # Remove any non-digit characters from the model answer
    if model_answer:
        model_answer = re.sub(r'\D', '', model_answer)

    is_correct = (model_answer == ground_truth)
    print (ground_truth, partial_words, model_name, is_correct)

    # Store the entire usage object
    usage_json = json.dumps(usage) if usage else None

    # Save the result to the database
    if not found:
        cursor.execute('''
            INSERT OR REPLACE INTO results (model, m, n, example_id, input, ground_truth, model_answer, is_correct, usage, model_output)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, m, n, example_id, f"{number1} * {number2}", ground_truth, model_answer, is_correct, usage_json, model_output))
        conn.commit()

    # Rate limiting: 20 requests per minute per API key
    #time.sleep(3)  # Adjust this delay as needed based on your actual rate limits
    return is_correct, usage  # Return is_correct and usage for further processing

# Function to process numbers from the dataset
def process_number(s):
    # Remove spaces and reverse the digits (since they were reversed in the dataset)
    s = s.replace(' ', '')[::-1]
    return s

# Load all datasets into memory
datasets = {}
max_examples = 0
DDD = 21
for m in range(1, DDD):
    for n in range(1, DDD):
        validation_dataset = os.path.join(validation_base_dir, f'{m}_by_{n}/test.txt')
        if not os.path.isfile(validation_dataset):
            print(f"Validation dataset not found: {validation_dataset}")
            continue
        with open(validation_dataset, 'r') as f:
            lines = f.readlines()
            datasets[(m, n)] = lines
            if len(lines) > max_examples:
                max_examples = len(lines)

print(f"Loaded datasets for all (m, n). Maximum examples per dataset: {max_examples}")

# Initialize results dictionaries
all_models_accs = {model_name: defaultdict(lambda: defaultdict(float)) for model_name in MODELS_TO_TEST}
all_models_reasoning_tokens = {model_name: defaultdict(lambda: defaultdict(lambda: None)) for model_name in MODELS_TO_TEST}

# Main loop over example indices
for example_id in range(max_examples):
    print(f"Processing example_id: {example_id}")
    for m in range(1, DDD):
        for n in range(1, DDD):
            print (f'{m} X {n}')
            # Check if dataset exists and has this example
            if (m, n) not in datasets or example_id >= len(datasets[(m, n)]):
                continue
            line = datasets[(m, n)][example_id]
            line = line.strip()
            if '||' not in line:
                continue  # Invalid format
            input_part, output_part = line.split('||')
            input_part = input_part.strip()
            output_part = output_part.strip()

            # Process input
            tokens = input_part.split('*')
            if len(tokens) != 2:
                continue  # Invalid format
            number1_s = tokens[0].strip()
            number2_s = tokens[1].strip()

            number1 = process_number(number1_s)
            number2 = process_number(number2_s)

            # Process output
            # Ground truth: Remove any '####' and process the number
            output_tokens = output_part.split('####')
            ground_truth_s = output_tokens[-1].strip()
            ground_truth = process_number(ground_truth_s)

            # Remove any non-digit characters from ground truth
            ground_truth = re.sub(r'\D', '', ground_truth)

            for model_name in MODELS_TO_TEST:
                if False and example_exists(model_name, m, n, example_id):
                    # Fetch the stored results
                    cursor.execute('SELECT is_correct, usage FROM results WHERE model=? AND m=? AND n=? AND example_id=?',
                                   (model_name, m, n, example_id))
                    row = cursor.fetchone()
                    is_correct_example = row[0]
                    usage_json = row[1]
                    usage = json.loads(usage_json) if usage_json else {}
                else:
                    # Process the example
                    result = process_example(model_name, m, n, example_id, number1, number2, ground_truth)
                    if result is None:
                        print(f"Failed to process example {example_id} for {m}x{n} on model {model_name}")
                        continue
                    is_correct_example, usage = result

                # Update accuracy counts
                # We use a simple averaging over examples processed so far
                if 'total_examples' not in all_models_accs[model_name][m]:
                    all_models_accs[model_name][m]['total_examples'] = defaultdict(int)
                    all_models_accs[model_name][m]['correct_examples'] = defaultdict(int)
                all_models_accs[model_name][m]['total_examples'][n] += 1
                if is_correct_example:
                    all_models_accs[model_name][m]['correct_examples'][n] += 1

                # Update reasoning tokens if available
                if usage:
                    if 'completion_tokens_details' in usage and 'reasoning_tokens' in usage['completion_tokens_details']:
                        reasoning_tokens = usage['completion_tokens_details']['reasoning_tokens']
                    elif 'completion_tokens' in usage:
                        reasoning_tokens = usage['completion_tokens']  # Fallback
                    else:
                        reasoning_tokens = None
                else:
                    reasoning_tokens = None

                if 'reasoning_tokens_sum' not in all_models_reasoning_tokens[model_name][m]:
                    all_models_reasoning_tokens[model_name][m]['reasoning_tokens_sum'] = defaultdict(float)
                    all_models_reasoning_tokens[model_name][m]['reasoning_tokens_count'] = defaultdict(int)
                if reasoning_tokens is not None:
                    all_models_reasoning_tokens[model_name][m]['reasoning_tokens_sum'][n] += reasoning_tokens
                    all_models_reasoning_tokens[model_name][m]['reasoning_tokens_count'][n] += 1

    # After each example, update the heatmaps
    for model_name in MODELS_TO_TEST:
        accs = {}
        reasoning_tokens_dict = {}
        for m in range(1, DDD):
            if m not in all_models_accs[model_name]:
                continue
            accs[m] = {}
            reasoning_tokens_dict[m] = {}
            for n in range(1, DDD):
                total = all_models_accs[model_name][m]['total_examples'].get(n, 0)
                correct = all_models_accs[model_name][m]['correct_examples'].get(n, 0)
                if total > 0:
                    accuracy = (correct / total) * 100
                else:
                    accuracy = None
                accs[m][n] = accuracy

                tokens_sum = all_models_reasoning_tokens[model_name][m]['reasoning_tokens_sum'].get(n, 0)
                tokens_count = all_models_reasoning_tokens[model_name][m]['reasoning_tokens_count'].get(n, 0)
                if tokens_count > 0:
                    avg_tokens = tokens_sum / tokens_count
                else:
                    avg_tokens = None
                reasoning_tokens_dict[m][n] = avg_tokens

        # Convert accs to DataFrame for heatmap
        df_acc = pd.DataFrame.from_dict(accs, orient='index').sort_index().transpose()
        
        plt.figure(figsize=(12, 7.4))
        cmap = 'RdYlGn'
        #ax = sns.heatmap(df_acc, annot=True, fmt=".1f", cbar=True, cmap=cmap, linewidths=0.5,
        #            annot_kws={"size": 8}, cbar_kws={"shrink": 0.75})
        ax = sns.heatmap(df_acc, annot=df_acc.applymap(custom_annot), fmt="", cbar=True, cmap=cmap, linewidths=0.5, annot_kws={"size": 11}, cbar_kws={"shrink": 0.75}, vmin=0)
        #ax.set_aspect('equal', 'box')
        ax.set_xlabel('Digits in Number 1') # flipped, but for consistency
        ax.set_ylabel('Digits in Number 2')
        # Move x-axis labels to the top
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', which='both', length=0)
        cbar = ax.collections[0].colorbar
        cbar.set_label('Accuracy (%)')
        plt.title(f'Accuracy of {model_name}', fontsize=18)

        #plt.tight_layout()
        heatmap_path = f'{heatmap_dir}/heatmap_accuracy_{model_name}example{example_id +1}.png'
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Accuracy heatmap saved to {heatmap_path}")

        # Reasoning Tokens Heatmap
        if any(any(rt is not None for rt in row.values()) for row in reasoning_tokens_dict.values()):
            df_rt = pd.DataFrame.from_dict(reasoning_tokens_dict, orient='index').sort_index().transpose()
            plt.figure(figsize=(12, 7.4))
            cmap = 'YlOrBr'
            #ax = sns.heatmap(df_rt, annot=True, fmt=".1f", cbar=True, cmap=cmap, linewidths=0.5,
            #            annot_kws={"size": 8}, cbar_kws={"shrink": 0.75})
            ax = sns.heatmap(df_rt, annot=True, fmt=".0f", cbar=True, cmap=cmap, linewidths=0.5, annot_kws={"size": 9}, cbar_kws={"shrink": 0.75}, vmin=0)
            #ax.set_aspect('equal', 'box')
            ax.set_xlabel('Digits in Number 1')
            ax.set_ylabel('Digits in Number 2')
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            ax.tick_params(axis='x', which='both', length=0)
            cbar = ax.collections[0].colorbar
            cbar.set_label('# Tokens')
            plt.title(f'# Private Reasoning Tokens of {model_name}', fontsize=18)
            #plt.tight_layout()
            heatmap_path = f'{heatmap_dir}/heatmap_reasoning_tokens_{model_name}example{example_id +1}.png'
            plt.savefig(heatmap_path)
            plt.close()
            print(f"Reasoning tokens heatmap saved to {heatmap_path}")

        # Save intermediate results
        with open(f'all_accuracies_{model_name}.json', 'w') as f:
            json.dump(accs, f, indent=2)
        with open(f'all_reasoning_tokens_{model_name}.json', 'w') as f:
            json.dump(reasoning_tokens_dict, f, indent=2)

print("All processing complete.")

# Optionally, you can save all models' accuracies and reasoning tokens to combined JSON files
with open('all_models_accuracies.json', 'w') as f:
    json.dump(all_models_accs, f, indent=2)

with open('all_models_reasoning_tokens.json', 'w') as f:
    json.dump(all_models_reasoning_tokens, f, indent=2)

print("All models' accuracy results saved to 'all_models_accuracies.json'")
print("All models' reasoning tokens results saved to 'all_models_reasoning_tokens.json'")
