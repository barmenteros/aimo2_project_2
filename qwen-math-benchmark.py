"""
This script runs a benchmark of the Qwen2.5-Math-7B-Instruct model on a math problem dataset.
It uses the Chain-of-Thought and Tool-Integrated Reasoning prompts to solve math problems step-by-step.
The benchmark processes problems in batches to optimize GPU memory usage and performance.
It's intended to be run in a Kaggle notebook with the Qwen2.5-Math-7B-Instruct model installed.

Requirements:
pip install packaging>=23.1
pip install numpy>=1.24.3
pip install pandas>=2.1.1
pip install torch>=2.1.0
pip install transformers>=4.37.0
pip install tqdm>=4.65.0
pip install bitsandbytes>=0.44.1
"""

# Standard library imports
import os
import time
import logging
from typing import List, Tuple, Optional, Dict, NoReturn
from dataclasses import dataclass
from datetime import datetime
import gc
from contextlib import contextmanager
import signal
from types import FrameType
import sys
import subprocess

# Configure logging
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
logging_level = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f'qwen_math_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def debug_print(msg: str):
    """Print debug messages if DEBUG is enabled"""
    if DEBUG:
        print(f"[DEBUG] {msg}")
        logger.debug(msg)


def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package])


# Ensure that 'packaging' is installed
try:
    from packaging import version
except ImportError:
    print("Packaging not found. Installing packaging...")
    install_package("packaging>=23.1")
    from packaging import version

# Ensure the latest version of bitsandbytes is installed
try:
    import bitsandbytes as bnb

    if version.parse(bnb.__version__) < version.parse("0.44.1"):
        print("Updating bitsandbytes to version 0.44.1...")
        install_package("bitsandbytes>=0.44.1")
        import bitsandbytes as bnb
except ImportError:
    print("bitsandbytes not found. Installing bitsandbytes version 0.44.1...")
    install_package("bitsandbytes>=0.44.1")
    import bitsandbytes as bnb

# Ensure 'tqdm' is installed
try:
    from tqdm.auto import tqdm
except ImportError:
    print("tqdm not found. Installing tqdm...")
    install_package("tqdm>=4.65.0")
    from tqdm.auto import tqdm

# Protected third-party imports
try:
    import pandas as pd
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    print("Error: Required packages not installed. Please run:")
    print("pip install numpy>=1.24.3 pandas>=2.1.1 torch>=2.1.0 transformers>=4.37.0")
    sys.exit(1)

# Version validation
REQUIRED_VERSIONS = {
    "np": "1.24.3",
    "pd": "2.1.1",
    "torch": "2.1.0",
    "transformers": "4.37.0",
    "bnb": "0.44.1",
}

PACKAGE_MAP = {
    "np": "numpy",
    "pd": "pandas",
    "torch": "torch",
    "transformers": "transformers",
    "bnb": "bitsandbytes",
}

for package_name, min_version in REQUIRED_VERSIONS.items():
    pkg_name = PACKAGE_MAP[package_name]
    try:
        pkg = __import__(pkg_name)
        current_version = pkg.__version__
    except ImportError:
        print(f"{pkg_name} is not installed. Installing {pkg_name}...")
        install_package(f"{pkg_name}>={min_version}")
        pkg = __import__(pkg_name)
        current_version = pkg.__version__
    if version.parse(current_version) < version.parse(min_version):
        print(
            f"{pkg_name} version {current_version} is too old. Upgrading to {min_version} or newer..."
        )
        install_package(f"{pkg_name}>={min_version}")
        # Reload the upgraded package
        pkg = __import__(pkg_name)
        current_version = pkg.__version__
    # Update the module in sys.modules
    sys.modules[pkg_name] = pkg


def optimize_gpu_performance():
    """Configure PyTorch for optimal GPU performance"""
    if not torch.cuda.is_available():
        return

    # Enable TF32 for matmul and convolution operations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable CuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

    # Disable gradient synchronization
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()


@contextmanager
def timeout(seconds: int):
    def signal_handler(signum: int, frame: Optional[FrameType]) -> NoReturn:
        # Use parameters to avoid unused warning
        logger.debug(f"Signal {signum} received in frame {frame}")
        raise TimeoutError(f"Timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)


@dataclass
class PredictionResult:
    """Stores the result of a single prediction"""

    problem_id: str
    problem: str
    model_output: str
    parsed_answer: Optional[int]
    actual_answer: int
    is_correct: bool
    computation_time: float
    status: str
    error: Optional[str] = None


class GPUMonitor:
    """Monitor GPU memory usage"""

    @staticmethod
    def log_gpu_memory():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
                logger.info(
                    f"GPU {i} Memory: Allocated={memory_allocated:.2f}MB, Reserved={memory_reserved:.2f}MB"
                )

    @staticmethod
    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


def log_gpu_utilization():
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE)
    print(result.stdout.decode("utf-8"))


class QwenMathBenchmark:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"):
        """Initialize the benchmark environment"""
        debug_print("Starting initialization...")

        # Verify Kaggle environment
        if not os.path.exists("/kaggle/working"):
            raise RuntimeError("This script must be run in a Kaggle notebook")
        debug_print("Kaggle environment verified")

        logger.info("Initializing QwenMathBenchmark...")
        self.initialization_state: Dict[str, bool] = {
            "environment_setup": False,
            "model_loaded": False,
            "gpu_monitor_initialized": False,
        }

        try:
            logger.info("Setting up GPU monitor...")
            self.gpu_monitor = GPUMonitor()
            self.initialization_state["gpu_monitor_initialized"] = True

            logger.info("Setting up environment...")
            self.setup_environment()
            self.initialization_state["environment_setup"] = True

            logger.info(f"Loading model: {model_name}")
            # Add timeout for model loading
            with timeout(600):  # 10 minute timeout
                self.load_model(model_name)
            self.initialization_state["model_loaded"] = True

            self.results: List[PredictionResult] = []
            self.batch_size = self._determine_optimal_batch_size()
            logger.info(f"Initialization complete. Batch size: {self.batch_size}")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            logger.error("Initialization state:")
            for state, value in self.initialization_state.items():
                logger.error(f"  {state}: {value}")
            raise

    def setup_environment(self):
        """Configure the computing environment with optimizations"""
        try:
            # Set up GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            if self.device.type == "cuda":
                logger.info("Configuring CUDA settings...")
                # Enable TF32 for better performance on A100 GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # Set optimal CUDA settings
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

                # Log CUDA configuration
                logger.info("CUDA Configuration:")
                logger.info(f"  TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
                logger.info(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")

                # Clear GPU memory
                self.gpu_monitor.clear_gpu_memory()

            # Log initial GPU state
            self.gpu_monitor.log_gpu_memory()
            return True

        except Exception as e:
            logger.error(f"Environment setup failed: {str(e)}")
            return False

    def _determine_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on available GPU memory"""
        if not torch.cuda.is_available():
            return 1

        try:
            total_memory = sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(torch.cuda.device_count())
            )
            # Use approximately 80% of available memory
            usable_memory = total_memory * 0.8
            # Adjust based on the model size without quantization
            estimated_model_size = (
                14 * 1024**3
            )  # 14GB for 7B model without quantization
            return max(1, int(usable_memory / estimated_model_size))
        except Exception as e:
            logger.warning(f"Error determining batch size: {e}")
            return 1

    def load_model(self, model_name: str):
        """Load the Qwen model with optimized GPU settings"""
        try:
            debug_print(f"Attempting to load model: {model_name}")
            debug_print(
                f"Available CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 'None'}"
            )

            # Initialize GPU optimization
            optimize_gpu_performance()

            # Configure model parameters for better GPU utilization
            model_kwargs = {
                # Changed from 'balanced' to 'auto'. Recommended by the model developers.
                "device_map": "auto",
                "trust_remote_code": True,
                # Changed from 'torch.float16' to 'auto'. Recommended by the model developers.
                "torch_dtype": "auto",
                # Quantization disabled. Not recommended by the model developers.
                # "quantization_config": BitsAndBytesConfig(
                #    load_in_8bit=True, bnb_4bit_compute_dtype=torch.float16
                # ),
                # Excluded 'max_memory'. The model developers do not mention it, and it can interfere with 'device_map' functionality.
                # "max_memory": {
                #    i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.8 / 1024**3)}GiB"
                #    for i in range(torch.cuda.device_count())
                # },  # Added memory limits
                "revision": "main",
                "token": None,
            }

            # Load model without autocast (not necessary during model loading)
            start_time = time.time()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
            debug_print(f"Model load time: {time.time() - start_time:.2f}s")

            # Verify device placement
            for name, param in self.model.named_parameters():
                logger.info(f"{name} is on {param.device}")

            # Load tokenizer with optimized settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                model_max_length=512,
                padding_side="left",  # Added for better batch processing
                truncation_side="right",  # Added for better batch processing
            )

            # self.model.parallelize() disabled. The model developers do not mention it, and it can interfere with 'device_map' functionality.
            # Move model to GPU explicitly if needed
            # if hasattr(self.model, "parallelize"):
            #    self.model.parallelize()  # Use model parallelization if available

            # Log GPU utilization
            log_gpu_utilization()

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def create_cot_prompt(self, problem: str) -> str:
        """Create a Chain-of-Thought prompt"""
        return f"""Please solve this math problem step by step. Show your work clearly.
After solving, provide your final answer as a single number between 0-999.

Problem:
{problem}

Solution:
Let's solve this step by step:
1) First, let's understand what we're given...

Please proceed with the solution, and end with:
Therefore, the final answer (0-999) is: [your answer]"""

    def create_tir_prompt(self, problem: str) -> str:
        """Create a Tool-Integrated Reasoning prompt"""
        return f"""Solve this math problem using available mathematical tools and operations.
Show each step of your calculation clearly.
You can use: basic arithmetic, algebra, calculus, or mathematical functions as needed.
End with a number between 0-999.

Problem:
{problem}

Let's solve this systematically:
1) Tools/operations needed:
2) Step-by-step solution:

Final answer (0-999): [your answer]"""

    def extract_answer(self, model_output: str) -> Optional[int]:
        """Extract the final numerical answer from model output"""
        try:
            # Look for explicit answer format first
            if "final answer" in model_output.lower():
                answer_text = model_output.lower().split("final answer")[-1]
                numbers = [
                    int(n)
                    for n in answer_text.replace("[", " ").replace("]", " ").split()
                    if n.isdigit()
                ]
                if numbers:
                    return numbers[0] % 1000

            # Fall back to looking for any number in the last line
            last_line = model_output.strip().split("\n")[-1]
            numbers = [int(n) for n in last_line.split() if n.isdigit()]
            if numbers:
                return numbers[-1] % 1000

            return None
        except Exception as e:
            logger.error(f"Error extracting answer: {str(e)}")
            return None

    def predict_single(
        self, problem_id: str, problem: str, actual_answer: int
    ) -> PredictionResult:
        """Make a prediction for a single problem"""
        start_time = time.time()
        try:
            # Try CoT prompt first
            prompt = self.create_cot_prompt(problem)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.1,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

            model_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            parsed_answer = self.extract_answer(model_output)

            # If CoT fails, try TIR prompt
            if parsed_answer is None:
                prompt = self.create_tir_prompt(problem)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.1,
                            num_return_sequences=1,
                        )

                model_output = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                parsed_answer = self.extract_answer(model_output)

            computation_time = time.time() - start_time

            if parsed_answer is None:
                status = "failed_to_parse"
                is_correct = False
            else:
                status = "success"
                is_correct = parsed_answer == actual_answer

            return PredictionResult(
                problem_id=problem_id,
                problem=problem,
                model_output=model_output,
                parsed_answer=parsed_answer,
                actual_answer=actual_answer,
                is_correct=is_correct,
                computation_time=computation_time,
                status=status,
            )

        except Exception as e:
            computation_time = time.time() - start_time
            logger.error(f"Error processing problem {problem_id}: {str(e)}")
            return PredictionResult(
                problem_id=problem_id,
                problem=problem,
                model_output="",
                parsed_answer=None,
                actual_answer=actual_answer,
                is_correct=False,
                computation_time=computation_time,
                status="error",
                error=str(e),
            )

    def predict_batch(
        self, problems: List[Tuple[str, str, int]]
    ) -> List[PredictionResult]:
        """Process a batch of problems with optimized GPU usage"""
        try:
            # Prepare prompts for batch processing
            prompts = [self.create_cot_prompt(p[1]) for p in problems]

            # Print the input prompts for debugging
            for i, prompt in enumerate(prompts):
                logger.info(f"Problem {i+1} Input Prompt:\n{prompt}\n")
                print(
                    f"Problem {i+1} Input Prompt:\n{prompt}\n"
                )  # Ensure output in notebook

            # Tokenize all prompts at once
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=768,  # Increased from 512
            ).to(self.device)

            # Generate all outputs at once
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.1,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV-caching
                    num_beams=1,  # Disable beam search for speed
                    # If you want to use beam search to potentially improve the quality of the generated outputs at the cost of speed,
                    # you can set num_beams to a value greater than 1 (e.g., num_beams=3) and keep early_stopping=True.
                    # Note: This will increase computation time, so adjust according to your performance needs.
                    # For example, you can set num_beams=3 and early_stopping=False to generate 3 beams without early stopping.
                    # Removed early_stopping=True
                )

            # Process outputs
            results = []
            for (problem_id, problem, actual_answer), output in zip(problems, outputs):
                model_output = self.tokenizer.decode(output, skip_special_tokens=True)
                parsed_answer = self.extract_answer(model_output)

                # Print the model's output for debugging
                logger.info(f"Problem {problem_id} Model Output:\n{model_output}\n")
                print(
                    f"Problem {problem_id} Model Output:\n{model_output}\n"
                )  # Ensure output in notebook

                results.append(
                    PredictionResult(
                        problem_id=problem_id,
                        problem=problem,
                        model_output=model_output,
                        parsed_answer=parsed_answer,
                        actual_answer=actual_answer,
                        is_correct=(parsed_answer == actual_answer),
                        computation_time=0,  # Will be updated later
                        status=(
                            "success"
                            if parsed_answer is not None
                            else "failed_to_parse"
                        ),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return [
                PredictionResult(
                    problem_id=p[0],
                    problem=p[1],
                    model_output="",
                    parsed_answer=None,
                    actual_answer=p[2],
                    is_correct=False,
                    computation_time=0,
                    status="error",
                    error=str(e),
                )
                for p in problems
            ]

    def run_benchmark(self, dataset_path: str):
        """Run the benchmark with efficient batching and progress tracking"""
        try:
            # Load dataset
            logger.info(f"Loading dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)
            total_problems = len(df)
            logger.info(f"Loaded {total_problems} problems")

            # Process only the first 2 problems for testing
            df = df.head(2)
            logger.info("Processing only the first 2 problems for testing")

            # Convert to list of tuples for easier batch processing
            problems = list(df.itertuples(index=False, name=None))

            # Initialize progress tracking
            start_time = time.time()
            total_batches = (len(problems) + self.batch_size - 1) // self.batch_size

            logger.info(
                f"Starting benchmark with {total_batches} batches (batch_size={self.batch_size})"
            )

            with tqdm(total=len(problems), desc="Processing Problems") as pbar:
                for i in range(0, len(problems), self.batch_size):
                    batch = problems[i : i + self.batch_size]
                    batch_start = time.time()

                    logger.info(
                        f"\nProcessing batch {i//self.batch_size + 1}/{total_batches}"
                    )
                    logger.info(f"Initial GPU state for batch:")
                    self.gpu_monitor.log_gpu_memory()

                    # Process batch
                    try:
                        results = self.predict_batch(batch)
                        batch_time = time.time() - batch_start

                        # Log batch results
                        if results:
                            correct = sum(1 for r in results if r.is_correct)
                            success = sum(1 for r in results if r.status == "success")
                            logger.info(f"Batch Results:")
                            logger.info(f"  Success Rate: {success}/{len(results)}")
                            logger.info(f"  Accuracy: {correct}/{len(results)}")
                            logger.info(
                                f"  Time per problem: {batch_time/len(results):.2f}s"
                            )

                            # Assign computation time
                            for result in results:
                                result.computation_time = batch_time / len(results)

                            self.results.extend(results)
                    except Exception as e:
                        logger.error(f"Batch processing failed: {str(e)}")
                        # Create error results for the batch
                        error_results = [
                            PredictionResult(
                                problem_id=p[0],
                                problem=p[1],
                                model_output="",
                                parsed_answer=None,
                                actual_answer=p[2],
                                is_correct=False,
                                computation_time=time.time() - batch_start,
                                status="error",
                                error=str(e),
                            )
                            for p in batch
                        ]
                        self.results.extend(error_results)

                    # Update progress
                    pbar.update(len(batch))

                    # Log GPU usage and clear memory
                    logger.info("Final GPU state for batch:")
                    self.gpu_monitor.log_gpu_memory()
                    self.gpu_monitor.clear_gpu_memory()
                    log_gpu_utilization()

            # Final summary
            total_time = time.time() - start_time
            logger.info("\nBenchmark Complete!")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Average time per problem: {total_time/len(problems):.2f}s")

            if self.results:
                save_success = self.save_results()
                if not save_success:
                    logger.warning(
                        "Failed to save results, but continuing with summary"
                    )
                self.print_summary()

        except Exception as e:
            logger.error(f"Benchmark run failed: {str(e)}")
            raise

    def save_results(self) -> bool:
        """
        Save benchmark results to CSV
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            if not self.results:
                logger.warning("No results to save")
                return False

            # Create results directory in Kaggle working directory
            results_dir = os.path.join("/kaggle/working", "benchmark_results")
            os.makedirs(results_dir, exist_ok=True)

            # Generate timestamp for unique filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                results_dir, f"benchmark_results_{timestamp}.csv"
            )

            # Create DataFrame from results with error checking
            valid_results = [r for r in self.results if r is not None]
            if not valid_results:
                logger.warning("No valid results to save after filtering")
                return False

            results_df = pd.DataFrame(
                [
                    {
                        "id": r.problem_id,
                        "problem": r.problem,
                        "model_output": r.model_output,
                        "parsed_answer": r.parsed_answer,
                        "actual_answer": r.actual_answer,
                        "is_correct": r.is_correct,
                        "computation_time": r.computation_time,
                        "status": r.status,
                        "error": r.error,
                    }
                    for r in valid_results
                ]
            )

            # Save with explicit encoding and mode
            results_df.to_csv(output_file, index=False, encoding="utf-8", mode="w")
            logger.info(f"Results saved successfully to {output_file}")

            # Verify the file was created
            if os.path.exists(output_file):
                return True
            else:
                logger.error("File was not created successfully")
                return False

        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}", exc_info=True)
            return False

    def print_summary(self):
        """Print detailed performance summary"""
        total_problems = len(self.results)
        successful = sum(1 for r in self.results if r.status == "success")
        correct = sum(1 for r in self.results if r.is_correct)
        avg_time = np.mean([r.computation_time for r in self.results])

        logger.info("\nBenchmark Performance Summary:")
        logger.info(f"Total Problems: {total_problems}")
        logger.info(
            f"Successful Predictions: {successful} ({successful/total_problems*100:.1f}%)"
        )
        logger.info(f"Correct Answers: {correct} ({correct/total_problems*100:.1f}%)")
        logger.info(f"Average Time per Problem: {avg_time:.2f}s")
        logger.info(
            f"Total GPU Time Used: {sum(r.computation_time for r in self.results):.2f}s"
        )

        # Memory usage summary
        if hasattr(self, "gpu_monitor"):
            self.gpu_monitor.log_gpu_memory()


def cleanup_gpu_memory(benchmark=None):
    """Safely cleanup GPU memory"""
    try:
        if benchmark is not None:
            logger.info("Cleaning up GPU monitor memory...")
            benchmark.gpu_monitor.clear_gpu_memory()
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            gc.collect()
        logger.info("GPU memory cleanup completed")
    except Exception as e:
        logger.error(f"GPU cleanup failed: {str(e)}")


def test_benchmark(
    num_problems: int = 1,
    dataset_path: str = "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv",
):
    """Test the benchmark with specified number of problems"""
    debug_print("\n=== Starting Test Benchmark ===")

    # Use Kaggle dataset path
    dataset_path = (
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv"
    )
    debug_print(f"Using dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        alt_path = os.path.join(os.getcwd(), "reference.csv")
        debug_print(f"Dataset not found, trying alternate path: {alt_path}")
        if os.path.exists(alt_path):
            dataset_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found in Kaggle or local paths")

    benchmark = None
    try:
        logger.info("\n=== Starting Benchmark Test ===")

        # Validate dataset path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

        # Load test problems
        logger.info("Loading test problems...")
        test_df = pd.read_csv(dataset_path).head(num_problems)
        logger.info(f"Loaded {len(test_df)} problems for testing")

        # Initialize benchmark
        logger.info("Initializing benchmark...")
        benchmark = QwenMathBenchmark()
        logger.info("Benchmark initialized successfully")

        # Create temporary test file
        logger.info("Creating temporary test file...")
        test_path = os.path.join(os.getcwd(), "test_problems.csv")
        test_df.to_csv(test_path, index=False)
        logger.info(f"Test file created at {test_path}")

        # Run benchmark
        logger.info("Starting benchmark run...")
        benchmark.run_benchmark(test_path)

        # Clean up
        logger.info("Cleaning up...")
        if os.path.exists(test_path):
            os.remove(test_path)
            logger.info("Test file removed")

        logger.info("=== Benchmark Test Complete ===")
        return benchmark.results

    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        return None
    finally:
        cleanup_gpu_memory(benchmark)


if __name__ == "__main__":
    debug_print("Script started")
    debug_print(f"Python version: {sys.version}")
    debug_print(f"PyTorch version: {torch.__version__}")
    debug_print(f"CUDA available: {torch.cuda.is_available()}")

    # Set DEBUG=true in Kaggle notebook
    # Add this to your notebook:
    # import os
    # os.environ['DEBUG'] = 'true'

    DATASET_PATH = (
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv"
    )
    benchmark = None

    try:
        benchmark = QwenMathBenchmark()
        benchmark.run_benchmark(DATASET_PATH)
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}", exc_info=True)
        debug_print(f"Error details: {str(e)}")
    finally:
        cleanup_gpu_memory(benchmark)
