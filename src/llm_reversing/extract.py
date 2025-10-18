#!/usr/bin/env python3
"""
EXAONE Glitch Token Hunter - Step 1: Embedding Extraction
Extract embedding matrix and calculate L2 Norm

Principle:
- Tokens not in training data: gradient=0 → only weight decay → norm→0
- Frequently used tokens: large gradient → high norm

Usage:
    pixi run python src/llm_reversing/token_analysis/extract_embeddings.py --model "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct" --output_dir "results" --threshold 2.0
"""

import os
import json
import numpy as np
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import fire
from dotenv import load_dotenv
from huggingface_hub import login


class EmbeddingExtractor:
    """Extract and analyze embeddings from EXAONE model"""

    def __init__(self, model_name: str, output_dir: str = "results", overwrite: bool = False):
        """
        Args:
            model_name: HuggingFace model name (e.g., "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
            output_dir: Output directory for results
            overwrite: Whether to overwrite existing files
        """
        self.model_name = model_name
        self.overwrite = overwrite

        # Create model-specific subdirectory
        # Convert model name to safe directory name (replace "/" with "_")
        safe_model_name = model_name.replace("/", "_")
        self.output_dir = Path(output_dir) / safe_model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading model: {model_name}")
        print(f"Output directory: {self.output_dir}")

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        self.norms = None

    def load_model(self):
        """Load model and tokenizer"""
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,  # Use bfloat16 as in official example
            trust_remote_code=True,
            device_map="auto",
        )

        print("Model loaded successfully!")
        print(f"   Vocab size: {self.tokenizer.vocab_size:,}")

    def extract_embeddings(self):
        """Extract embedding matrix"""
        print("\nExtracting embedding matrix...")

        # Get embedding layer
        embedding_layer = self.model.get_input_embeddings()

        # GPU → CPU → Float32 → NumPy (BFloat16 not supported by numpy)
        self.embeddings = embedding_layer.weight.data.cpu().float().numpy()

        vocab_size, embed_dim = self.embeddings.shape
        print("Embeddings extracted!")
        print(f"   Shape: {self.embeddings.shape}")
        print(f"   Vocab size: {vocab_size:,}")
        print(f"   Embedding dimension: {embed_dim:,}")

        # Print memory usage
        mem_mb = self.embeddings.nbytes / (1024**2)
        print(f"   Memory: {mem_mb:.2f} MB")

    def calculate_norms(self):
        """Calculate L2 Norm"""
        print("\nCalculating L2 norms...")

        # L2 norm = √(Σ xᵢ²)
        self.norms = np.linalg.norm(self.embeddings, axis=1)

        print("Norms calculated!")
        print(f"   Min: {self.norms.min():.4f}")
        print(f"   Max: {self.norms.max():.4f}")
        print(f"   Mean: {self.norms.mean():.4f}")
        print(f"   Std: {self.norms.std():.4f}")

    def get_basic_stats(self):
        """Return basic statistics"""
        return {
            "vocab_size": len(self.norms),
            "embedding_dim": self.embeddings.shape[1],
            "norm_stats": {
                "min": float(self.norms.min()),
                "max": float(self.norms.max()),
                "mean": float(self.norms.mean()),
                "std": float(self.norms.std()),
                "median": float(np.median(self.norms)),
                "q25": float(np.percentile(self.norms, 25)),
                "q75": float(np.percentile(self.norms, 75)),
            },
        }

    def find_anomalies(
        self, low_std_threshold: float = 3.0, high_std_threshold: float = 1.0
    ):
        """Detect anomalous tokens with separate thresholds for low and high norm tokens

        Args:
            low_std_threshold: Standard deviation threshold for low norm tokens (default 3.0 = more strict)
            high_std_threshold: Standard deviation threshold for high norm tokens (default 1.0 = more lenient)

        Returns:
            dict: Low norm tokens and high norm tokens
        """
        print(
            f"\nFinding anomalies (low: {low_std_threshold}σ, high: {high_std_threshold}σ)..."
        )

        mean = self.norms.mean()
        std = self.norms.std()

        # Calculate separate thresholds
        low_threshold = mean - low_std_threshold * std
        high_threshold = mean + high_std_threshold * std

        print(
            f"   Low threshold: {low_threshold:.4f} (mean - {low_std_threshold}σ)"
        )
        print(
            f"   High threshold: {high_threshold:.4f} (mean + {high_std_threshold}σ)"
        )

        # Find anomalous indices
        low_norm_indices = np.where(self.norms < low_threshold)[0]
        high_norm_indices = np.where(self.norms > high_threshold)[0]

        print("\nResults:")
        print(
            f"   Low norm tokens: {len(low_norm_indices)} ({100 * len(low_norm_indices) / len(self.norms):.2f}%)"
        )
        print(
            f"   High norm tokens: {len(high_norm_indices)} ({100 * len(high_norm_indices) / len(self.norms):.2f}%)"
        )
        print(
            f"   Normal tokens: {len(self.norms) - len(low_norm_indices) - len(high_norm_indices)}"
        )

        # Collect token information
        low_norm_tokens = self._get_token_info(low_norm_indices)
        high_norm_tokens = self._get_token_info(high_norm_indices)

        return {
            "thresholds": {
                "low": float(low_threshold),
                "high": float(high_threshold),
                "low_std_multiplier": low_std_threshold,
                "high_std_multiplier": high_std_threshold,
            },
            "counts": {
                "low_norm": len(low_norm_indices),
                "high_norm": len(high_norm_indices),
                "normal": len(self.norms)
                - len(low_norm_indices)
                - len(high_norm_indices),
            },
            "low_norm_tokens": low_norm_tokens,
            "high_norm_tokens": high_norm_tokens,
        }

    def _get_token_info(self, indices):
        """Extract token information from indices"""
        tokens = []
        for idx in indices:
            try:
                # Decode token - convert numpy.int64 to Python int first
                token_str = self.tokenizer.decode([int(idx)])

                # Original token ID → token string (from vocab)
                if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                    raw_token = self.tokenizer.convert_ids_to_tokens(int(idx))
                else:
                    raw_token = token_str

                tokens.append(
                    {
                        "index": int(idx),
                        "token": token_str,
                        "raw_token": str(raw_token),
                        "norm": float(self.norms[idx]),
                        "unicode": self._get_unicode_info(token_str),
                    }
                )
            except Exception as e:
                tokens.append(
                    {
                        "index": int(idx),
                        "token": f"[ERROR: {str(e)}]",
                        "raw_token": "",
                        "norm": float(self.norms[idx]),
                        "unicode": "",
                    }
                )

        # Sort by norm
        tokens.sort(key=lambda x: x["norm"])
        return tokens

    def _get_unicode_info(self, text: str):
        """Extract unicode information"""
        try:
            return ", ".join(
                [f"U+{ord(c):04X}" for c in text if c.isprintable()]
            )
        except Exception:
            return ""

    def _decode_unicode_info(self, text: str):
        """Decode unicode information to readable characters"""
        try:
            return "".join([c for c in text if c.isprintable()])
        except Exception:
            return ""

    def save_results(self, anomalies: dict, stats: dict):
        """Save results"""
        print("\nSaving results...")

        # Save analysis results as JSON
        results_file = self.output_dir / "analysis_results.json"
        if results_file.exists() and not self.overwrite:
            print("   Skipped: analysis_results.json (already exists, use --overwrite to overwrite)")
        else:
            results = {
                "model_name": self.model_name,
                "stats": stats,
                "anomalies": anomalies,
            }

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print("   Saved: analysis_results.json")

        # 3. Generate human-readable text report
        self._save_text_report(anomalies, stats)

    def _save_text_report(self, anomalies: dict, stats: dict):
        """Generate text report"""
        report_path = self.output_dir / "REPORT.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("EXAONE GLITCH TOKEN ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Model: {self.model_name}\n")
            f.write(f"Vocab Size: {stats['vocab_size']:,}\n")
            f.write(f"Embedding Dim: {stats['embedding_dim']:,}\n\n")

            f.write("-" * 80 + "\n")
            f.write("NORM STATISTICS\n")
            f.write("-" * 80 + "\n")
            for key, value in stats["norm_stats"].items():
                f.write(f"{key:>10s}: {value:>10.4f}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("ANOMALY DETECTION\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Low threshold:  {anomalies['thresholds']['low']:.4f} ({anomalies['thresholds']['low_std_multiplier']}σ)\n"
            )
            f.write(
                f"High threshold: {anomalies['thresholds']['high']:.4f} ({anomalies['thresholds']['high_std_multiplier']}σ)\n\n"
            )

            f.write(
                f"Low norm tokens:  {anomalies['counts']['low_norm']:>6,} "
                f"({100 * anomalies['counts']['low_norm'] / stats['vocab_size']:.2f}%)\n"
            )
            f.write(
                f"High norm tokens: {anomalies['counts']['high_norm']:>6,} "
                f"({100 * anomalies['counts']['high_norm'] / stats['vocab_size']:.2f}%)\n"
            )
            f.write(
                f"Normal tokens:    {anomalies['counts']['normal']:>6,}\n\n"
            )

            # Top 20 low norm tokens
            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP 20 LOW NORM TOKENS (Potential Glitch Tokens)\n")
            f.write("=" * 80 + "\n\n")

            for i, token in enumerate(anomalies["low_norm_tokens"][:20], 1):
                f.write(
                    f"{i:2d}. Token {token['index']:>6d}: {token['token']!r:>20s} "
                    f"(norm: {token['norm']:>8.4f})\n"
                )
                if token["unicode"]:
                    f.write(f"    Unicode: {token['unicode']}\n")
                f.write("\n")

            # Top 20 high norm tokens
            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP 20 HIGH NORM TOKENS (Over-trained)\n")
            f.write("=" * 80 + "\n\n")

            high_sorted = sorted(
                anomalies["high_norm_tokens"], key=lambda x: -x["norm"]
            )
            for i, token in enumerate(high_sorted[:20], 1):
                f.write(
                    f"{i:2d}. Token {token['index']:>6d}: {token['token']!r:>20s} "
                    f"(norm: {token['norm']:>8.4f})\n"
                )
                if token["unicode"]:
                    f.write(f"    Unicode: {token['unicode']}\n")
                f.write("\n")

        print("   Saved: REPORT.txt")

    def run_full_analysis(
        self, low_std_threshold: float = 3.0, high_std_threshold: float = 1.0
    ):
        """Run full analysis pipeline"""
        print("Starting full analysis pipeline...\n")

        # 1. Load model
        self.load_model()

        # 2. Extract embeddings
        self.extract_embeddings()

        # 3. Calculate norms
        self.calculate_norms()

        # 4. Get statistics
        stats = self.get_basic_stats()

        # 5. Detect anomalies
        anomalies = self.find_anomalies(low_std_threshold, high_std_threshold)

        # 6. Save results
        self.save_results(anomalies, stats)

        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)
        print(f"\nResults saved to: {self.output_dir}")
        print("   - embeddings.npy: Full embedding matrix")
        print("   - norms.npy: L2 norms for all tokens")
        print("   - analysis_results.json: Detailed analysis")
        print("   - REPORT.txt: Human-readable report")

        return anomalies, stats


def main(
    model: str = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    output_dir: str = "results",
    low_threshold: float = 3.0,
    high_threshold: float = 1.0,
    overwrite: bool = False,
):
    """Extract and analyze embeddings from EXAONE model

    Args:
        model: Model name from HuggingFace
        output_dir: Output directory for results
        low_threshold: Standard deviation threshold for low norm tokens (default 3.0 = more strict)
        high_threshold: Standard deviation threshold for high norm tokens (default 1.0 = more lenient)
        overwrite: Whether to overwrite existing files
    """
    # Load environment variables
    load_dotenv()

    # Login to HuggingFace Hub if token is available
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Logging in to HuggingFace Hub...")
        login(token=hf_token)
        print("Successfully logged in to HuggingFace Hub!")
    else:
        print(
            "Warning: HF_TOKEN not found in environment variables. Some models may not be accessible."
        )

    # Run analysis
    extractor = EmbeddingExtractor(model, output_dir, overwrite)
    extractor.run_full_analysis(low_threshold, high_threshold)


if __name__ == "__main__":
    fire.Fire(main)
