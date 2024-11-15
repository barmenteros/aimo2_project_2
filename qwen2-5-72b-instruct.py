import os  # operating system
import time
import warnings

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl  # polars

import torch  # pytorch
import kaggle_evaluation.aimo_2_inference_server

pd.set_option('display.max_colwidth', None)  # display all columns
cutoff_time = time.time() + (4 * 60 + 30) * 60  # 4 hours 30 minutes from now

# VLLM model and sampling parameters for inference and evaluation
from vllm import LLM, SamplingParams

# ignore warnings
warnings.simplefilter('ignore')

# use all GPUs available on the Kaggle notebook (4 GPUs) for tensor parallelism with PyTorch and VLLM model inference
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# disable tokenizers parallelism to avoid deadlocks with PyTorch tensor parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm_model_pth = '/kaggle/input/qwen2.5/transformers/72b-instruct-awq/1'

# Load the VLLM model for inference
llm = LLM(
    llm_model_pth,
    # The data type for the model weights and activations. Options: "float", "half". Default is "float"
    dtype="half",
    # Maximum number of sequences per iteration. Default is 256 for "float" and 128 for "half"
    max_num_seqs=16,  # Increased from 8 to process more sequences in parallel
    # Model context length. Default is 4096
    max_model_len=4096,
    # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer. Default is False
    trust_remote_code=True,
    # The number of GPUs to use for distributed execution with tensor parallelism. Default is 4
    tensor_parallel_size=4,
    # The ratio (between 0 and 1) of GPU memory to reserve for the model. Default is 0.95
    gpu_memory_utilization=0.95,
    # Random seed for reproducibility. Default is 2024
    seed=2024,
)

# Get the tokenizer for the VLLM model
tokenizer = llm.get_tokenizer()

import re  # regular expressions
import keyword  # Python keywords


# Extract Python code from the text using regular expressions (regex) and return it as a string separated by two newlines (\n\n) between each code block
# (e.g., "code1\n\n\ncode2\n\n\ncode3")
def extract_python_code(text):
    # Regular expression pattern for Python code blocks
    pattern = r'```python\s*(.*?)\s*```'
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text, re.DOTALL)
    # Return the Python code blocks as a string separated by two newlines
    return "\n\n".join(matches)  # Join all matches with two newlines


# Process the Python code by adding import statements and printing variables that are not inside any indentation (e.g., "x = 42" will be printed as "x=42")
# and return the processed code as a string separated by newlines (\n) between each row (e.g., "row1\nrow2\nrow3")
def process_python_code(query):
    # Add import statements
    # Also print variables if they are not inside any indentation
    query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
    # Split the query into rows
    current_rows = query.strip().split("\n")
    new_rows = []
    for row in current_rows:
        # Add the current row
        new_rows.append(row)
        if not row.startswith(" ") and "=" in row:
            # Get the variable name
            variables_to_print = row.split("=")[0].strip()
            # Split multiple variables
            for variable_to_print in variables_to_print.split(","):
                # Remove leading/trailing spaces
                variable_to_print = variable_to_print.strip()
                # Check if the variable is a valid identifier and not a Python keyword
                if variable_to_print.isidentifier(
                ) and not keyword.iskeyword(variable_to_print):
                    if row.count("(") == row.count(")") and row.count(
                            "[") == row.count("]"):
                        # TODO: use some AST to parse code
                        new_rows.append(
                            f'\ntry:\n    print(f"{variable_to_print}={{str({variable_to_print})[:100]}}")\nexcept:\n    pass\n'
                        )
    return "\n".join(new_rows)


# Extract text inside "boxed" curly braces {text} using regular expressions (regex) and return it as a string or an empty string if no match is found (e.g., "{text}" will return "text")
def extract_boxed_text(text):
    # Regular expression pattern for boxed text
    pattern = r'oxed{(.*?)}'
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    # Return the first match
    return matches[0]


from collections import Counter  # Counter for counting occurrences of elements in a list
import random  # random numbers and shuffling lists (e.g., for random sampling)


# Select the most common answer from a list of answers and return it as an integer (e.g., [1, 2, 2, 3, 3, 3] will return 3)
def select_answer(answers):
    # Counter for counting occurrences of elements
    counter = Counter()
    for answer in answers:
        try:
            # Check if the answer is an integer
            if int(answer) == float(answer):
                # Add the answer to the counter with a small random noise to break ties
                counter[int(answer)] += 1 + random.random() / 1_000
        except:
            pass
    if not counter:
        # Return the default answer if no valid answers are found
        return 210
    # Select the most common answer from the counter
    _, answer = sorted([(v, k) for k, v in counter.items()], reverse=True)[0]
    # Return the answer modulo 1000 (e.g., 1000 will be returned as 0)
    return answer % 1000


import os
import tempfile  # temporary files and directories
import subprocess  # subprocesses


# Python Read-Eval-Print Loop (REPL) for executing Python code with a timeout using subprocesses and temporary files and directories for security and resource management
class PythonREPL:

    def __init__(self, timeout=5):
        # Timeout for code execution in seconds
        self.timeout = timeout

    def __call__(self, query):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporary Python file path
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            # Write the query to the temporary file
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)

            try:
                # Execute the Python file with a timeout
                result = subprocess.run(
                    ["python3", temp_file_path],
                    # Capture stdout and stderr
                    capture_output=True,
                    # Do not raise an exception on non-zero exit codes
                    check=False,
                    # Return stdout and stderr as text
                    text=True,
                    # Timeout for code execution
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return False, f"Execution timed out after {self.timeout} seconds."

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0:
                return True, stdout
            else:
                # Process the error message to remove the temporary file path
                # This makes the error message cleaner and more user-friendly
                error_lines = stderr.split("\n")
                cleaned_errors = []
                for line in error_lines:
                    if temp_file_path in line:
                        # Remove the path from the error line
                        line = line.replace(temp_file_path, "<temporary_file>")
                    cleaned_errors.append(line)
                cleaned_error_msg = "\n".join(cleaned_errors)
                # Include stdout in the error case
                combined_output = f"{stdout}\n{cleaned_error_msg}" if stdout else cleaned_error_msg
                return False, combined_output


# Sampling parameters for the VLLM model inference and evaluation (e.g., temperature, min_p, max_tokens, etc.)
sampling_params = SamplingParams(
    # The temperature of the sampling distribution. Higher values mean more randomness. Default is 1.0
    temperature=1.0,
    # The minimum token probability for nucleus sampling. Default is 0.01
    min_p=0.01,
    # Whether to skip special tokens in the output. Default is True
    skip_special_tokens=True,
    # Maximum number of tokens to generate. Default is 1024
    max_tokens=2400,
    # Stop generation at the end of the code block
    stop=["```\n"],
    # Include the stop string in the output. Default is False
    include_stop_str_in_output=True,
)


# Generate a message using the VLLM model and return the generated message as a string (e.g., "Hello, world!")
# or an empty string if no message is generated (e.g., if the input is empty)
def batch_message_generate(list_of_messages) -> list[list[dict]]:

    list_of_texts = [
        # Apply the chat template to each conversation and add the generation prompt to each message in the conversation
        # (e.g., "role: user\ncontent: Hello, world!") and return the list of texts as a list of strings
        # (e.g., ["role: user\ncontent: Hello, world!", "role: assistant\ncontent: Hi!"])
        tokenizer.apply_chat_template(conversation=messages,
                                      tokenize=False,
                                      add_generation_prompt=True)
        for messages in list_of_messages
    ]
    # Generate messages using the VLLM model with the list of texts and sampling parameters
    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )

    for messages, single_request_output in zip(list_of_messages,
                                               request_output):
        # print()
        # print(single_request_output.outputs[0].text)
        # print()
        messages.append({
            'role': 'assistant',
            'content': single_request_output.outputs[0].text
        })

    return list_of_messages


# Filter messages that contain boxed text and extract the boxed text as the answer
def batch_message_filter(
        list_of_messages) -> tuple[list[list[dict]], list[str]]:
    extracted_answers = []
    list_of_messages_to_keep = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]['content'])
        if answer:
            extracted_answers.append(answer)
        else:
            list_of_messages_to_keep.append(messages)
    return list_of_messages_to_keep, extracted_answers


# Execute Python code in messages and return the output as a string (e.g., "Hello, world!")
# or an empty string if no output is generated (e.g., if the input is empty)
# or an error occurs (e.g., syntax error) during code execution (e.g., "SyntaxError: invalid syntax")
# or if the code execution times out (e.g., "Execution timed out after 5 seconds.")
# or if the code execution fails (e.g., "Execution failed with exit code 1.")
# or if the code execution is successful but no output is generated (e.g., "Execution successful but no output.")
# or if the code execution is successful but the output is empty (e.g., "Execution successful but empty output.")
# or if the code execution is successful but the output is too long (e.g., "Execution successful but output is too long.")
# or if the code execution is successful but the output is too short (e.g., "Execution successful but output is too short.")
# or if the code execution is successful but the output is invalid (e.g., "Execution successful but invalid output.")
# or if the code execution is successful but the output is not a string (e.g., "Execution successful but output is not a string.")
# or if the code execution is successful but the output is not a valid answer
def batch_message_execute(list_of_messages) -> list[list[dict]]:
    for messages in list_of_messages:
        python_code = extract_python_code(messages[-1]['content'])
        python_code = process_python_code(python_code)
        # print('\n\n' + python_code + '\n\n')
        try:
            print('c', end='')
            is_successful, output = PythonREPL()(python_code)
            if is_successful:
                print('o', end='')
            else:
                print('e', end='')
        except Exception as e:
            print('f', end='')
            output = str(e)
        print(python_code)
        print()
        print(output)
        print("\n\n")
        messages.append({
            'role': 'user',
            'content': "```output\n" + output + "\n```"
        })
    print()
    return list_of_messages


def create_starter_messages(question, index):
    cycle_size = 2
    if False:
        pass
    elif index % cycle_size == 1:
        # https://github.com/QwenLM/Qwen2.5-Math?tab=readme-ov-file#-hugging-face-transformers
        return [{
            "role":
            "system",
            "content":
            "Please reason step by step, and put your final answer within \\boxed{}."
        }, {
            "role": "user",
            "content": question
        }]
    else:
        # https://github.com/QwenLM/Qwen2.5-Math?tab=readme-ov-file#-hugging-face-transformers
        return [{
            "role":
            "system",
            "content":
            "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."
        }, {
            "role":
            "user",
            "content":
            question + "\n\nBegin your answer by importing sympy."
        }]


"""
Dynamic Batch Size Optimization
-----------------------------
The following implementation replaces the static batch size of 32 with a dynamic batch sizing mechanism.

Rationale:
- Fixed batch sizes can be suboptimal as GPU memory availability varies during execution
- Large static batch sizes may cause OOM errors or memory fragmentation
- Small static batch sizes may underutilize available resources
- Memory requirements per conversation can vary based on problem complexity

Benefits:
- Adaptive resource utilization based on real-time GPU memory availability
- Reduced risk of OOM errors during long running sessions
- Better throughput by maximizing parallel processing when possible
- More resilient to varying problem complexities

Implementation notes:
- Uses 80% of available GPU memory to leave headroom for fluctuations
- Sets reasonable min/max bounds (8-48) to maintain stability
- Falls back to conservative batch size (16) if memory detection fails
- Considers multi-GPU setups by checking all available devices

Last modified: November 2024
"""


def get_optimal_batch_size():
    """
    Dynamically determine the optimal batch size for parallel message processing
    based on available GPU memory across all devices.
    
    The function:
    1. Checks available memory across all GPU devices
    2. Calculates safe batch size using conservative memory estimates
    3. Applies bounds to ensure stable execution
    
    Returns:
        int: Optimal batch size between 8 and 48
        
    Note:
        - Assumes ~0.5GB memory usage per conversation
        - Uses 80% of available memory as safety margin
        - Falls back to batch size 16 if memory detection fails
    """
    try:
        # Get available GPU memory
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            free_memory = total_memory - allocated_memory
            gpu_memory.append(free_memory)

        # Use the minimum free memory across all GPUs
        min_free_memory = min(gpu_memory)

        # Calculate batch size based on free memory
        # Conservative estimate: assume each conversation requires about 0.5GB
        memory_per_conversation = 0.5 * (1024**3)  # 0.5GB in bytes
        optimal_batch_size = int(
            (min_free_memory * 0.8) /
            memory_per_conversation)  # Use 80% of free memory

        # Ensure batch size is within reasonable bounds
        optimal_batch_size = max(8, min(48, optimal_batch_size))

        return optimal_batch_size
    except Exception as e:
        print(f"Error calculating batch size: {e}")
        return 16  # Default fallback batch size


def predict_for_question(question: str) -> int:
    import os
    # only run this code if it's not a competition rerun
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # only run this code if the question is not the example question from the competition page (to avoid wasting time)
        if question != "Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?":
            return 210
    # check if the time limit has been reached
    if time.time() > cutoff_time:
        return 210

    question += "\nIf the final answer is a number larger than 1 million, take modulo 1000."
    print(question)

    # Dynamic batch size instead of fixed 32 for parallel message processing (e.g., for tensor parallelism) based on available GPU memory across all devices (e.g., 8-48 conversations in parallel)
    batch_size = get_optimal_batch_size()
    list_of_messages = [
        create_starter_messages(question, index) for index in range(batch_size)
    ]

    all_extracted_answers = []
    # 4 rounds of message generation, filtering, and execution
    for _ in range(4):
        # Generate messages using the VLLM model
        list_of_messages = batch_message_generate(list_of_messages)
        # Filter messages that contain boxed text and extract the boxed text as the answer
        list_of_messages, extracted_answers = batch_message_filter(
            list_of_messages)
        # Extend the list of extracted answers
        all_extracted_answers.extend(extracted_answers)
        if not list_of_messages:
            break
        # Execute Python code in messages and return the output as a string
        list_of_messages = batch_message_execute(list_of_messages)

    print(all_extracted_answers)
    answer = select_answer(all_extracted_answers)
    print(answer)

    print("\n\n")
    return answer


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame,
            question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    # get the first element of the DataFrame
    id_ = id_.item(0)
    print("------")
    print(id_)

    # get the first element of the DataFrame
    question = question.item(0)
    answer = predict_for_question(question)
    print(question)
    print("------\n\n\n")
    return pl.DataFrame({'id': id_, 'answer': answer})


# predict_for_question("Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?")

pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index=False)

inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(
    predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway((
        #             '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
        'reference.csv', ))
"""
import gc  # garbage collector

# clean memory (RAM and GPU memory) to avoid memory leaks and out-of-memory errors (e.g., due to PyTorch tensor parallelism)
# and improve performance (e.g., by reducing memory fragmentation) and stability (e.g., by avoiding memory leaks) of the VLLM model
def clean_memory(deep=False):
    gc.collect()  # garbage collector (RAM)
    if deep:
        # memory allocator (RAM) for PyTorch tensors
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    # memory allocator (GPU) for PyTorch tensors
    torch.cuda.empty_cache()


# delete the VLLM model to free up GPU memory
del llm

# clean memory (RAM and GPU memory) to avoid memory leaks and out-of-memory errors
clean_memory(deep=True)
"""
