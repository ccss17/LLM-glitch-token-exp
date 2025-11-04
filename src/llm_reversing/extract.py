import os
import json
import numpy as np
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login


class EmbeddingExtractor:
    def __init__(self, model_name: str, output_dir: str = "results"):
        self.model_name = model_name

        safe_model_name = model_name.replace("/", "_")
        self.output_dir = Path(output_dir) / safe_model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = None
        self.model = None
        self.embeddings = None
        self.norms = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        print("Model loaded")
        print(f"   Vocab size: {self.tokenizer.vocab_size:,}")

    def extract_embeddings(self):
        print("\nExtracting embedding matrix")

        embedding_layer = self.model.get_input_embeddings()

        # GPU → CPU → Float32 → NumPy (BFloat16 not supported by numpy)
        self.embeddings = embedding_layer.weight.data.cpu().float().numpy()

        vocab_size, embed_dim = self.embeddings.shape
        print("Embeddings extracted")
        print(f"   Shape: {self.embeddings.shape}")
        print(f"   Vocab size: {vocab_size:,}")
        print(f"   Embedding dimension: {embed_dim:,}")

    def calculate_norms(self):
        self.norms = np.linalg.norm(self.embeddings, axis=1)

        print("Norms calculated:")
        print(f"   Min: {self.norms.min():.4f}")
        print(f"   Max: {self.norms.max():.4f}")
        print(f"   Mean: {self.norms.mean():.4f}")
        print(f"   Std: {self.norms.std():.4f}")

    def get_basic_stats(self):
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

    def find_anomalies(self, low_std_th: float = 3.0, high_std_th: float = 1.0):
        """
        low_std_th: Standard deviation threshold for low norm tokens
        high_std_th: Standard deviation threshold for high norm tokens
        """
        print(f"\nFinding anomalies (low: {low_std_th}σ, high: {high_std_th}σ)")

        mean = self.norms.mean()
        std = self.norms.std()

        low_threshold = mean - low_std_th * std
        high_threshold = mean + high_std_th * std

        print(f"   Low threshold: {low_threshold:.4f} (mean - {low_std_th}σ)")
        print(f"   High threshold: {high_threshold:.4f} (mean + {high_std_th}σ)")

        low_norm_indices = np.where(self.norms < low_threshold)[0]
        high_norm_indices = np.where(self.norms > high_threshold)[0]

        print("\nResults:")
        print(f"Low norm tokens: {len(low_norm_indices)} ({100 * len(low_norm_indices) / len(self.norms):.2f}%)")
        print(f"High norm tokens: {len(high_norm_indices)} ({100 * len(high_norm_indices) / len(self.norms):.2f}%)")
        print(f"   Normal tokens: {len(self.norms) - len(low_norm_indices) - len(high_norm_indices)}")

        low_norm_tokens = self._get_token_info(low_norm_indices)
        high_norm_tokens = self._get_token_info(high_norm_indices)

        return {
            "thresholds": {
                "low": float(low_threshold),
                "high": float(high_threshold),
                "low_std_multiplier": low_std_th,
                "high_std_multiplier": high_std_th,
            },
            "counts": {
                "low_norm": len(low_norm_indices),
                "high_norm": len(high_norm_indices),
                "normal": len(self.norms) - len(low_norm_indices) - len(high_norm_indices),
            },
            "low_norm_tokens": low_norm_tokens,
            "high_norm_tokens": high_norm_tokens,
        }

    def _get_token_info(self, indices):
        tokens = []
        for idx in indices:
            # numpy.int64 --> int
            token_str = self.tokenizer.decode([int(idx)])
            # token ID --> token string (from vocab)
            raw_token = self.tokenizer.convert_ids_to_tokens(int(idx))
            token_info = {
                "index": int(idx),
                "token": token_str,
                "raw_token": str(raw_token),
                "norm": float(self.norms[idx]),
                "unicode": self._get_unicode_info(token_str),
            }
            tokens.append(token_info)

        tokens.sort(key=lambda x: x["norm"])
        return tokens

    def _get_unicode_info(self, text: str):
        try:
            return ", ".join(
                [f"U+{ord(c):04X}" for c in text if c.isprintable()]
            )
        except Exception:
            return ""

    def save_results(self, anomalies: dict, stats: dict):
        results_file = self.output_dir / "analysis_results.json"
        results = {
            "model_name": self.model_name,
            "stats": stats,
            "anomalies": anomalies,
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("Saved: analysis_results.json")

        self._save_text_report(anomalies, stats)

    def _save_text_report(self, anomalies: dict, stats: dict):
        report_path = self.output_dir / "REPORT.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 80}\nEXAONE GLITCH TOKEN ANALYSIS REPORT\n{'=' * 80}\n\n")

            f.write(f"Model: {self.model_name}\n")
            f.write(f"Vocab Size: {stats['vocab_size']:,}\n")
            f.write(f"Embedding Dim: {stats['embedding_dim']:,}\n\n")

            f.write(f"{'-' * 80}\nNORM STATISTICS\n{'-' * 80}\n")
            for key, value in stats["norm_stats"].items():
                f.write(f"{key:>10s}: {value:>10.4f}\n")

            f.write(f"\n{'-' * 80}\nANOMALY DETECTION\n{'-' * 80}\n")
            f.write(
                f"Low threshold:  {anomalies['thresholds']['low']:.4f} ({anomalies['thresholds']['low_std_multiplier']}σ)\n"
            )
            f.write(
                f"High threshold: {anomalies['thresholds']['high']:.4f} ({anomalies['thresholds']['high_std_multiplier']}σ)\n\n"
            )

            f.write(f"Low norm tokens:  {anomalies['counts']['low_norm']:>6,} ({100 * anomalies['counts']['low_norm'] / stats['vocab_size']:.2f}%)\n")
            f.write(f"High norm tokens: {anomalies['counts']['high_norm']:>6,} ({100 * anomalies['counts']['high_norm'] / stats['vocab_size']:.2f}%)\n")
            f.write(f"Normal tokens:    {anomalies['counts']['normal']:>6,}\n\n")

            # Top 20 low norm tokens
            f.write(f"\n{'=' * 80}\nTOP 20 LOW NORM TOKENS (Potential Glitch Tokens)\n{'=' * 80}\n\n")

            for i, token in enumerate(anomalies["low_norm_tokens"][:20], 1):
                f.write(f"{i:2d}. Token {token['index']:>6d}: {token['token']!r:>20s} (norm: {token['norm']:>8.4f})\n")
                if token["unicode"]:
                    f.write(f"    Unicode: {token['unicode']}\n")
                f.write("\n")

            # Top 20 high norm tokens
            f.write(f"\n{'=' * 80}\nTOP 20 HIGH NORM TOKENS (Over-trained)\n{'=' * 80}\n\n")

            high_sorted = sorted(anomalies["high_norm_tokens"], key=lambda x: -x["norm"])
            for i, token in enumerate(high_sorted[:20], 1):
                f.write(f"{i:2d}. Token {token['index']:>6d}: {token['token']!r:>20s} (norm: {token['norm']:>8.4f})\n")
                if token["unicode"]:
                    f.write(f"    Unicode: {token['unicode']}\n")
                f.write("\n")

        print("   Saved: REPORT.txt")

    def run(self, low_std_th: float = 3.0, high_std_th: float = 1.0):
        self.load_model()
        self.extract_embeddings()
        self.calculate_norms()
        stats = self.get_basic_stats()
        anomalies = self.find_anomalies(low_std_th, high_std_th)
        self.save_results(anomalies, stats)

        return anomalies, stats


def main(
    model: str,
    output_dir: str = "results",
    low_threshold: float = 3.0,
    high_threshold: float = 1.0,
):
    """
    low_threshold: Standard deviation threshold for low norm tokens
    high_threshold: Standard deviation threshold for high norm tokens
    """
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    extractor = EmbeddingExtractor(model, output_dir)
    extractor.run(low_threshold, high_threshold)


if __name__ == "__main__":
    """
    Extract embedding matrix and calculate L2 Norm
    - Tokens not in training data: gradient=0 → only weight decay → norm→0
    - Frequently used tokens: large gradient → high norm

    pixi run python src/llm_reversing/token_analysis/extract_embeddings.py --model "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct" --output_dir "results" --threshold 2.0
    """
    import fire
    fire.Fire(main)
