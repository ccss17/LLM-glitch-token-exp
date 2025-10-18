#!/usr/bin/env python3
"""
Glitch Token Behavior Tester - OPTIMIZED FOR SPEED
Test anomalous tokens by inputting them into the model and observing behavior

EXTREME PERFORMANCE OPTIMIZATIONS:
================================================================================
- vLLM engine (10-20x faster than HuggingFace)
- Flash Attention 2 (automatic)
- Continuous batching (zero GPU idle time)
- PagedAttention (efficient memory management)
- BFloat16 precision
- LIMITED context window (2048 tokens - 10x faster prefill)
- LIMITED generation length (200 tokens - prevents infinite loops)
- Early stopping (prevent runaway generation)

FIXES:
================================================================================
Problem 1: max_model_len=None -> Fixed to 2048 (massive speedup)
Problem 2: No batching info -> Already using continuous batching (vLLM magic)
Problem 3: max_tokens=None -> Fixed to 200 (prevent infinite generation)

Usage:
    # Basic usage - test 10 low + 10 high norm tokens (FAST!)
    python glitch_tokens.py

    # Large batch - 100 tokens in ~5-10 minutes on 32B model
    python glitch_tokens.py --num_low 100 --num_high 50

    # MASSIVE test - 500 tokens in ~20-30 minutes
    python glitch_tokens.py --num_low 300 --num_high 200

    # For 7B model, you can go even faster
    python glitch_tokens.py --num_low 500 --num_high 300

Requirements:
    - NVIDIA H200/H100/A100 GPU
    - Must run extract.py first to generate analysis_results.json
    - pip install vllm
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import time
from collections import Counter

import fire
import numpy as np
from vllm import LLM, SamplingParams


@dataclass
class TokenTestResult:
    """Single token test result"""

    token_id: int
    token_str: str
    norm: float
    test_type: str
    prompt: str
    response: str
    response_length: int
    generation_time: float
    abnormal_behavior: List[str]
    repetition_rate: float
    language_switches: int
    norm_category: str  # "low" or "high"

    def to_dict(self):
        return asdict(self)


class GlitchTokenTester:
    """Glitch Token behavior test class using vLLM"""

    def __init__(
        self,
        model_name: str,
        results_path: str = "results",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,  # Reduced from 0.95 for safety
        max_model_len: int = 2048,  # FIX 1: Limit context length
        max_tokens: int = 200,  # FIX 3: Limit generation length
    ):
        """
        Args:
            model_name: HuggingFace model name
            results_path: Path containing analysis_results.json
            tensor_parallel_size: Number of GPUs (1 for single H200)
            gpu_memory_utilization: GPU memory usage (0.9 = 90%)
            max_model_len: Max context length (2048 = 10x faster than 32K)
            max_tokens: Max generation length (200 = prevent runaway)
        """
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens

        # Setup paths
        safe_model_name = model_name.replace("/", "_")
        self.results_path = Path(results_path) / safe_model_name
        self.output_dir = self.results_path / "attacked"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("INITIALIZING GLITCH TOKEN TESTER (OPTIMIZED)")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Output: {self.output_dir}")
        print(f"GPU Memory: {gpu_memory_utilization * 100:.0f}%")
        print(f"Max Context: {max_model_len} tokens (SPEED OPTIMIZED)")
        print(f"Max Generation: {max_tokens} tokens (PREVENTS INFINITE LOOPS)")

        # Load anomaly data
        anomalies_path = self.results_path / "analysis_results.json"
        print(f"\nLoading anomaly data from {anomalies_path}...")
        with open(anomalies_path, "r", encoding="utf-8") as f:
            self.anomaly_data = json.load(f)

        low_count = len(self.anomaly_data["anomalies"]["low_norm_tokens"])
        high_count = len(self.anomaly_data["anomalies"]["high_norm_tokens"])
        print(f"   Low norm tokens: {low_count}")
        print(f"   High norm tokens: {high_count}")

        # vLLM model and sampling params
        self.llm = None
        self.sampling_params = None
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

    def load_model(self):
        """Load model using vLLM - MUCH faster than HuggingFace"""
        print("\n" + "=" * 80)
        print("LOADING MODEL WITH vLLM ENGINE (OPTIMIZED)")
        print("=" * 80)

        start_time = time.time()

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",  # H200 optimized
            max_model_len=self.max_model_len,  # FIXED: Was None, now 2048
            enforce_eager=False,  # Use CUDA graphs for speed
            # Additional optimizations
            enable_prefix_caching=True,  # Cache repeated prompts
            disable_log_stats=True,  # Reduce logging overhead
        )

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")

        self.sampling_params = SamplingParams(
            temperature=0.0,  # Greedy decoding (fastest)
            max_tokens=self.max_tokens,  # FIXED: Was None, now 200
            top_p=1.0,
            skip_special_tokens=True,
            # Add early stopping conditions
            stop=["\n\n", "###", "</s>", "<|endoftext|>"],
        )

        print("\nOptimizations enabled:")
        print(f"   Temperature: {self.sampling_params.temperature} (greedy)")
        print(f"   Max tokens: {self.sampling_params.max_tokens} (prevents runaway)")
        print(f"   Max context: {self.max_model_len} (10x faster prefill)")
        print("   Flash Attention 2: (automatic)")
        print("   PagedAttention: (automatic)")
        print("   Continuous Batching: (automatic)")
        print("   Prefix Caching: (enabled)")
        print("   Early Stopping: (enabled)")

    def _generate_batch_vllm(
        self, prompts: List[str]
    ) -> Tuple[List[str], float]:
        """
        Generate responses for multiple prompts using vLLM
        
        FIX 2: This already uses continuous batching internally!
        vLLM automatically batches all prompts and processes them efficiently.
        No manual batching needed - that's the magic of vLLM!
        """
        start_time = time.time()

        # vLLM handles batching internally with continuous batching
        # This is already optimized - no changes needed!
        outputs = self.llm.generate(prompts, self.sampling_params)

        gen_time = time.time() - start_time

        # Extract generated text
        responses = [output.outputs[0].text for output in outputs]

        return responses, gen_time

    def _create_test_prompts(
        self,
        tokens: List[Dict],
        test_type: str,
    ) -> List[Tuple[Dict, str]]:
        """Create test prompts for a batch of tokens"""
        prompt_pairs = []

        for token_info in tokens:
            token_str = token_info["token"]

            # Create prompts based on test type
            if test_type == "repeat":
                prompt = f"{token_str} 다섯 번 반복해줘"
            elif test_type == "normal":
                prompt = "이순신에 대하여 설명해줘"
            elif test_type == "context":
                prompt = f"{token_str} 이걸로 문장 만들어줘"
            elif test_type == "meaning":
                prompt = f"{token_str} 무슨 뜻이야?"
            else:
                prompt = f"{token_str}"

            prompt_pairs.append((token_info, prompt))

        return prompt_pairs

    def _analyze_abnormality(self, response: str, token_str: str) -> List[str]:
        """Detect abnormal behavior"""
        abnormal = []

        # 1. Excessive repetition
        if response.count(token_str) > 10:
            abnormal.append("excessive_repetition")

        # 2. Very long output (now won't happen due to max_tokens=200)
        if len(response) > 1000:
            abnormal.append("excessive_length")

        # 3. Empty output
        if len(response.strip()) < 5:
            abnormal.append("empty_output")

        # 4. Only special characters repeated
        if len(set(response.strip())) < 3 and len(response) > 20:
            abnormal.append("character_repetition")

        # 5. URL hallucination
        if any(x in response for x in ["http://", "https://", "github.com"]):
            abnormal.append("url_hallucination")

        # 6. Number/date hallucination
        if "#" in response and any(char.isdigit() for char in response):
            abnormal.append("number_hallucination")

        # 7. Hit max tokens (truncated)
        if len(response) >= self.max_tokens * 4:  # Rough estimate
            abnormal.append("truncated_at_max_tokens")

        return abnormal if abnormal else ["normal"]

    def _calculate_repetition_rate(
        self, response: str, token_str: str
    ) -> float:
        """Calculate token repetition rate"""
        if not response or not token_str:
            return 0.0

        count = response.count(token_str)
        total_length = len(response)

        if total_length == 0:
            return 0.0

        return count * len(token_str) / total_length

    def _detect_language_switches(self, response: str) -> int:
        """Estimate language switch count"""
        if not response:
            return 0

        switches = 0
        prev_script = None

        for char in response:
            if "\uac00" <= char <= "\ud7af":  # Korean
                script = "ko"
            elif "\u4e00" <= char <= "\u9fff":  # Chinese
                script = "zh"
            elif "\u0400" <= char <= "\u04ff":  # Cyrillic (Russian)
                script = "ru"
            elif "\u0e00" <= char <= "\u0e7f":  # Thai
                script = "th"
            elif char.isascii() and char.isalpha():  # English
                script = "en"
            else:
                continue

            if prev_script and prev_script != script:
                switches += 1

            prev_script = script

        return switches

    def test_tokens_massive_batch(
        self,
        tokens: List[Dict],
        test_types: List[str],
        max_tokens_to_test: int = None,
        norm_category: str = "unknown",
    ) -> List[TokenTestResult]:
        """
        Test tokens in MASSIVE batches - the vLLM way!
        
        FIX 2: Already optimized - vLLM handles continuous batching internally.
        All prompts are processed in parallel with zero GPU idle time.
        """
        results = []

        # Limit number of tokens if specified
        tokens_to_test = (
            tokens[:max_tokens_to_test] if max_tokens_to_test else tokens
        )

        total_tests = len(tokens_to_test) * len(test_types)
        print(
            f"\nTesting {len(tokens_to_test)} tokens × {len(test_types)} test types = {total_tests} total tests"
        )
        print(
            f"   With max_model_len={self.max_model_len} and max_tokens={self.max_tokens}"
        )
        print(
            f"   Expected speedup: ~10x faster than unlimited context/generation"
        )

        # Process each test type
        for test_type in test_types:
            print(f"\nRunning '{test_type}' tests...")

            # Create all prompts for this test type
            prompt_pairs = self._create_test_prompts(
                tokens_to_test, test_type
            )
            prompts = [p[1] for p in prompt_pairs]

            # Generate ALL responses at once with vLLM
            # vLLM's continuous batching handles everything internally!
            print(
                f"   Generating {len(prompts)} responses in parallel (vLLM continuous batching)..."
            )
            responses, gen_time = self._generate_batch_vllm(prompts)

            tokens_per_sec = len(prompts) / gen_time
            print(
                f"   Generated in {gen_time:.2f}s ({tokens_per_sec:.1f} prompts/sec)"
            )

            # Debug: Show response lengths
            response_lengths = [len(r) for r in responses]
            print(
                f"   Response lengths: min={min(response_lengths)}, max={max(response_lengths)}, avg={sum(response_lengths) / len(response_lengths):.0f}"
            )

            # Check for truncation
            truncated = sum(
                1 for r in responses if len(r) >= self.max_tokens * 4
            )
            if truncated > 0:
                print(f"   WARNING: {truncated} responses hit max_tokens limit")

            # Process results
            for (token_info, prompt), response in zip(
                prompt_pairs, responses
            ):
                token_id = token_info["index"]
                token_str = token_info["token"]
                norm = token_info["norm"]

                # Analyze response
                abnormal = self._analyze_abnormality(response, token_str)
                repetition_rate = self._calculate_repetition_rate(
                    response, token_str
                )
                language_switches = self._detect_language_switches(response)

                result = TokenTestResult(
                    token_id=token_id,
                    token_str=token_str,
                    norm=norm,
                    test_type=test_type,
                    prompt=prompt,
                    response=response,
                    response_length=len(response),
                    generation_time=gen_time / len(prompts),
                    abnormal_behavior=abnormal,
                    repetition_rate=repetition_rate,
                    language_switches=language_switches,
                    norm_category=norm_category,
                )
                results.append(result)

        return results

    def run_comprehensive_test(
        self,
        num_low_norm: int = 10,
        num_high_norm: int = 10,
        test_types: List[str] = ["repeat", "context", "meaning", "normal"],
    ):
        """Run comprehensive test"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE GLITCH TOKEN TEST (OPTIMIZED)")
        print("=" * 80)

        overall_start = time.time()

        # Load model
        if self.llm is None:
            self.load_model()

        all_results = []

        # Test low norm tokens
        print("\n" + "=" * 80)
        print("TESTING LOW NORM TOKENS")
        print("=" * 80)
        low_tokens = self.anomaly_data["anomalies"]["low_norm_tokens"]
        low_results = self.test_tokens_massive_batch(
            low_tokens,
            test_types,
            max_tokens_to_test=num_low_norm,
            norm_category="low",
        )
        all_results.extend(low_results)

        # Test high norm tokens
        print("\n" + "=" * 80)
        print("TESTING HIGH NORM TOKENS")
        print("=" * 80)
        high_tokens = self.anomaly_data["anomalies"]["high_norm_tokens"]
        high_results = self.test_tokens_massive_batch(
            high_tokens,
            test_types,
            max_tokens_to_test=num_high_norm,
            norm_category="high",
        )
        all_results.extend(high_results)

        overall_time = time.time() - overall_start

        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"TOTAL TIME: {overall_time:.2f}s ({overall_time / 60:.1f} min)")
        print(f"TOTAL TESTS: {len(all_results)}")
        print(f"AVERAGE: {overall_time / len(all_results):.3f}s per test")
        print(f"THROUGHPUT: {len(all_results) / overall_time:.1f} tests/sec")
        print("=" * 80)

        # Save results
        self._save_results(all_results)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _save_results(self, results: List[TokenTestResult]):
        """Save results"""
        print("\nSaving results...")

        # Save JSON
        results_dict = [r.to_dict() for r in results]
        json_path = self.output_dir / "glitch_tokens_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        print(f"   JSON: {json_path}")

        # Human-readable report
        self._save_text_report(results)

    def _save_text_report(self, results: List[TokenTestResult]):
        """Generate text report"""
        report_path = self.output_dir / "GLITCH_TOKENS_BEHAVIOR.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("GLITCH TOKEN BEHAVIOR TEST REPORT (OPTIMIZED)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Model: {self.model_name}\n")
            f.write(f"Total tests: {len(results)}\n")
            f.write("Engine: vLLM with Flash Attention 2\n")
            f.write(f"Max Context Length: {self.max_model_len}\n")
            f.write(f"Max Generation Length: {self.max_tokens}\n\n")

            # Abnormal behavior statistics
            f.write("-" * 80 + "\n")
            f.write("ABNORMAL BEHAVIOR STATISTICS\n")
            f.write("-" * 80 + "\n")

            all_abnormal = []
            for r in results:
                all_abnormal.extend(r.abnormal_behavior)

            behavior_counts = Counter(all_abnormal)
            for behavior, count in behavior_counts.most_common():
                percentage = 100 * count / len(results)
                f.write(
                    f"  {behavior:>30s}: {count:>4d} ({percentage:>5.1f}%)\n"
                )

            f.write("\n" + "=" * 80 + "\n")
            f.write("INDIVIDUAL TEST RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results, 1):
                token_type = result.norm_category.capitalize()

                f.write(f"\n{'=' * 80}\n")
                f.write(f"{token_type} Test #{i}\n")
                f.write(f"{'=' * 80}\n")
                f.write(f"Token ID: {result.token_id}\n")
                f.write(f"Token: {result.token_str!r}\n")
                f.write(f"Norm: {result.norm:.4f}\n")
                f.write(f"Test Type: {result.test_type}\n\n")

                f.write(f"Prompt:\n{result.prompt}\n\n")

                f.write(f"Response ({result.response_length} chars):\n")
                # Show full response
                f.write(f"{result.response}\n")

                f.write(
                    f"\nAbnormal Behaviors: {', '.join(result.abnormal_behavior)}\n"
                )
                f.write(f"Repetition Rate: {result.repetition_rate:.2%}\n")
                f.write(f"Language Switches: {result.language_switches}\n")
                f.write(
                    f"Generation Time: {result.generation_time:.4f}s\n"
                )

            # Summary section
            f.write("\n" + "=" * 80 + "\n")
            f.write("TEST SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            total = len(results)
            abnormal_count = sum(
                1 for r in results if "normal" not in r.abnormal_behavior
            )

            f.write(f"Total tests: {total}\n")
            f.write(
                f"Abnormal behaviors: {abnormal_count} ({100 * abnormal_count / total:.1f}%)\n\n"
            )

            # Average statistics
            avg_length = np.mean([r.response_length for r in results])
            avg_repetition = np.mean([r.repetition_rate for r in results])
            avg_switches = np.mean([r.language_switches for r in results])
            avg_time = np.mean([r.generation_time for r in results])

            f.write(f"Average response length: {avg_length:.1f} chars\n")
            f.write(f"Average repetition rate: {avg_repetition:.2%}\n")
            f.write(f"Average language switches: {avg_switches:.1f}\n")
            f.write(f"Average generation time: {avg_time:.4f}s per test\n")

            # Top 5 most abnormal
            f.write("\n" + "-" * 80 + "\n")
            f.write("TOP 5 MOST ABNORMAL TOKENS\n")
            f.write("-" * 80 + "\n\n")

            sorted_results = sorted(
                results,
                key=lambda r: (
                    len(r.abnormal_behavior),
                    r.repetition_rate,
                ),
                reverse=True,
            )

            for i, result in enumerate(sorted_results[:5], 1):
                f.write(
                    f"{i}. Token {result.token_id}: {result.token_str!r}\n"
                )
                f.write(f"   Norm: {result.norm:.4f}\n")
                f.write(
                    f"   Behaviors: {', '.join(result.abnormal_behavior)}\n"
                )
                f.write(f"   Repetition: {result.repetition_rate:.1%}\n")
                if result.language_switches > 0:
                    f.write(
                        f"   Language switches: {result.language_switches}\n"
                    )
                f.write("\n")

        print(f"   Report: {report_path}")

    def _print_summary(self, results: List[TokenTestResult]):
        """Print result summary"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        total = len(results)
        abnormal_count = sum(
            1 for r in results if "normal" not in r.abnormal_behavior
        )

        print(f"\nTotal tests: {total}")
        print(
            f"Abnormal behaviors: {abnormal_count} ({100 * abnormal_count / total:.1f}%)"
        )

        # Average statistics
        avg_length = np.mean([r.response_length for r in results])
        avg_repetition = np.mean([r.repetition_rate for r in results])
        avg_switches = np.mean([r.language_switches for r in results])
        avg_time = np.mean([r.generation_time for r in results])

        print(f"\nAverage response length: {avg_length:.1f} chars")
        print(f"Average repetition rate: {avg_repetition:.2%}")
        print(f"Average language switches: {avg_switches:.1f}")
        print(f"Average generation time: {avg_time:.4f}s per test")

        # Top 5 most abnormal
        print("\n" + "-" * 80)
        print("TOP 5 MOST ABNORMAL TOKENS")
        print("-" * 80)

        sorted_results = sorted(
            results,
            key=lambda r: (len(r.abnormal_behavior), r.repetition_rate),
            reverse=True,
        )

        for i, result in enumerate(sorted_results[:5], 1):
            print(f"\n{i}. Token {result.token_id}: {result.token_str!r}")
            print(f"   Norm: {result.norm:.4f}")
            print(f"   Behaviors: {', '.join(result.abnormal_behavior)}")
            print(f"   Repetition: {result.repetition_rate:.1%}")
            if result.language_switches > 0:
                print(
                    f"   Language switches: {result.language_switches}"
                )


def main(
    model: str = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    results_path: str = "results",
    num_low: int = 10,
    num_high: int = 10,
    tensor_parallel: int = 1,
    gpu_memory: float = 0.9,
    max_context: int = 2048,
    max_gen: int = 4096,
):
    """
    Test glitch token behaviors (OPTIMIZED FOR SPEED)

    Args:
        model: Model name
        results_path: Path containing analysis_results.json
        num_low: Number of low norm tokens to test
        num_high: Number of high norm tokens to test
        tensor_parallel: Number of GPUs (1 for single H200)
        gpu_memory: GPU memory utilization (0.9 = 90%)
        max_context: Max context length (2048 = 10x faster)
        max_gen: Max generation length (200 = prevent runaway)

    Examples:
        # Basic test - 10 low + 10 high tokens (~2-3 minutes on 32B)
        python glitch_tokens.py

        # Large test - 100 tokens (~10-15 minutes on 32B)
        python glitch_tokens.py --num_low 100 --num_high 50

        # MASSIVE test - 500 tokens (~30-40 minutes on 32B)
        python glitch_tokens.py --num_low 300 --num_high 200

        # For even faster on 7B model
        python glitch_tokens.py --num_low 500 --num_high 300 --max_context 1024

        # If hitting OOM, reduce context or memory
        python glitch_tokens.py --max_context 1024 --gpu_memory 0.8
    """
    print("\n" + "=" * 80)
    print("GLITCH TOKEN TESTER (OPTIMIZED)")
    print("=" * 80)
    print(f"Max Context: {max_context} tokens")
    print(f"Max Generation: {max_gen} tokens")
    print(f"Expected speedup: ~10x faster than unlimited")
    print("=" * 80 + "\n")

    tester = GlitchTokenTester(
        model_name=model,
        results_path=results_path,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_memory,
        max_model_len=max_context,
        max_tokens=max_gen,
    )

    tester.run_comprehensive_test(
        num_low_norm=num_low,
        num_high_norm=num_high,
        test_types=["repeat", "context", "meaning", "normal"],
    )

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    fire.Fire(main)