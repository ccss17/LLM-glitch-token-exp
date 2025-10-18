#!/usr/bin/env python3
"""
Normalize and deduplicate glitch token injection test results

Features:
1. Deduplication by token_id + attack_type
2. Response pattern detection (repetition, underscore flood, etc.)
3. Statistical analysis
4. Categorization by abnormal behavior
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import hashlib
import fire
from multiprocessing import Pool


@dataclass
class NormalizedResult:
    """Normalized test result"""
    
    token_id: int
    token_str: str
    payload: str
    response: str  # Cleaned response if repetitive
    response_length: int
    
    # Unique identifier for deduplication
    result_hash: str

    def to_dict(self):
        return asdict(self)


class ResultNormalizer:
    """Normalize and analyze injection test results"""
    
    def __init__(self, clean_responses: bool = True):
        self.results: List[Dict] = []
        self.normalized: List[NormalizedResult] = []
        self.stats = defaultdict(int)
        self.clean_responses = clean_responses
        
    def load_results(self, filepath: str):
        """Load JSON results"""
        print(f"Loading results from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        print(f"   Loaded {len(self.results)} results")
        
    def detect_repetition(self, text: str) -> tuple[bool, float, str, int]:
        """Detect text repetition pattern
        
        Returns:
            tuple: (has_repetition, repetition_ratio, repeated_phrase, repetition_count)
        """
        if not text or len(text) < 100:
            return False, 0.0, "", 0
        
        # Try multiple splitting strategies to catch different repetition patterns
        all_chunks = []
        
        # Strategy 1: Split by newlines (catches line-by-line repetitions)
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        if len(lines) >= 3:
            all_chunks.extend(lines)
        
        # Strategy 2: Split by numbered list pattern (e.g., "  1. text  2. text")
        # Split at spaces followed by numbers and periods/parens
        numbered_items = re.split(r'\s+(?=\d+[.)])', text)
        numbered_items = [item.strip() for item in numbered_items if item.strip() and len(item) > 5]
        if len(numbered_items) >= 3:
            all_chunks.extend(numbered_items)
        
        # Strategy 3: Split by commas and spaces (e.g., "text  , text  , text")
        comma_items = re.split(r'\s*,\s*', text)
        comma_items = [item.strip() for item in comma_items if item.strip() and len(item) > 5]
        if len(comma_items) >= 3:
            all_chunks.extend(comma_items)
        
        # Strategy 3b: Split by sentence delimiters
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 3:
            all_chunks.extend(sentences)
        
        # Strategy 4: Split by quoted strings (e.g., "text" "text" "text")
        # Find all quoted strings in the text
        quoted_items = re.findall(r'"([^"]+)"', text)
        if len(quoted_items) >= 3:
            all_chunks.extend(quoted_items)
        
        # Strategy 5: Split by multiple spaces (e.g., "●   ●   ●   ●")
        # Split on 2+ consecutive spaces
        space_items = re.split(r'\s{2,}', text)
        space_items = [item.strip() for item in space_items if item.strip()]
        if len(space_items) >= 10:  # Need many items since they're short
            all_chunks.extend(space_items)
        
        # Strategy 6: Detect character-level n-gram repetitions (e.g., "тнгтнгтнг..." or "000000...")
        # Check for short repeating patterns (1-10 chars)
        for n in [3, 4, 5, 2, 1, 6, 7, 8]:  # Try common lengths first, including single char
            if len(text) < n * 10:  # Need at least 10 repetitions for single chars
                continue
            pattern = text[:n]
            # Check if this pattern repeats throughout the text
            if pattern * (len(text) // n) == text[:len(text) // n * n]:
                # Found a repeating pattern
                repetitions = len(text) // n
                if repetitions >= 10:  # Require more repetitions for very short patterns
                    all_chunks.extend([pattern] * repetitions)
                    break  # Found the pattern, no need to check other lengths
        
        # Strategy 7: Split by common separators (periods, spaces, mixed)
        mixed_split = re.split(r'[.!?]\s*(?=\S)', text)  # Period followed by non-space
        mixed_split = [s.strip() for s in mixed_split if s.strip() and len(s) > 3]
        if len(mixed_split) >= 3:
            all_chunks.extend(mixed_split)
        
        if not all_chunks:
            return False, 0.0, "", 0
        
        # Normalize chunks by removing incrementing numbers (e.g., "1. text", "2. text" -> "text")
        normalized_chunks = []
        original_to_normalized = {}
        
        for chunk in all_chunks:
            # Remove leading numbers like "1. ", "123. ", "1) ", etc.
            normalized = re.sub(r'^\d+[.)]\s*', '', chunk)
            normalized_chunks.append(normalized)
            if normalized not in original_to_normalized and len(normalized) > 3:
                original_to_normalized[normalized] = chunk
        
        # Count duplicates on normalized chunks
        chunk_counts = Counter(normalized_chunks)
        
        # Find the most repeated chunk
        most_common = chunk_counts.most_common(1)
        if not most_common or most_common[0][1] < 3:
            return False, 0.0, "", 0
        
        repeated_phrase_normalized = most_common[0][0]
        repetition_count = most_common[0][1]
        
        # Get the original form (with first number occurrence)
        repeated_phrase = original_to_normalized.get(repeated_phrase_normalized, repeated_phrase_normalized)
        
        # Calculate ratio based on normalized chunks
        total_chunks = len(normalized_chunks)
        duplicates = sum(count - 1 for count in chunk_counts.values() if count > 1)
        repetition_ratio = duplicates / total_chunks if total_chunks > 0 else 0.0
        
        # Consider it repetitive if the most common phrase appears 5+ times
        # OR if overall repetition ratio is > 30%
        has_repetition = (repetition_count >= 5) or (repetition_ratio > 0.3)
        
        return has_repetition, repetition_ratio, repeated_phrase, repetition_count
    
    def clean_repetitive_response(self, text: str, repeated_phrase: str, repetition_count: int) -> str:
        """Clean up repetitive responses by showing just a sample
        
        Args:
            text: Original response text
            repeated_phrase: The phrase that's being repeated
            repetition_count: How many times it repeats
            
        Returns:
            Cleaned response with sample + metadata
        """
        if not repeated_phrase or repetition_count < 5:
            return text  # Not repetitive enough to clean
        
        # For highly repetitive content (10+ times), show very little
        # For moderately repetitive (5-9 times), show a bit more
        if repetition_count >= 10:
            sample_count = 1  # Show only 1 occurrence
            max_sample_length = 200  # Limit to 200 chars
        else:
            sample_count = 2  # Show 2 occurrences
            max_sample_length = 400
        
        # Find first few occurrences
        pattern = re.escape(repeated_phrase)
        matches = list(re.finditer(pattern, text))
        
        if len(matches) < sample_count:
            return text
        
        # Get position of the sample_count-th occurrence
        cutoff_pos = matches[sample_count - 1].end()
        
        # Limit the sample length
        sample = text[:min(cutoff_pos, max_sample_length)]
        if cutoff_pos > max_sample_length:
            sample = sample + "..."
        
        # Add metadata about repetition
        remaining = repetition_count - sample_count
        cleaned = f"{sample} [Repeated x{remaining} more: \"{repeated_phrase[:40]}{'...' if len(repeated_phrase) > 40 else ''}\"]"
        
        return cleaned
    
    
    def generate_hash(self, token_id: int, attack_type: str, response: str) -> str:
        """Generate unique hash for deduplication"""
        # Hash based on token_id, attack_type, and response preview
        content = f"{token_id}|{attack_type}|{response}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def normalize_single(self, result: Dict, clean_responses: bool = True) -> NormalizedResult:
        """Normalize single result
        
        Args:
            result: Raw result dictionary
            clean_responses: If True, clean up repetitive responses
        """
        response = result.get('response', '')
        payload = result.get('payload', '')
        original_length = result.get('response_length', len(response))
        
        # Detect repetition and clean if needed
        if clean_responses:
            has_rep, rep_ratio, repeated_phrase, rep_count = self.detect_repetition(response)
            if has_rep:
                response = self.clean_repetitive_response(response, repeated_phrase, rep_count)
        
        # Generate hash (use original response for deduplication)
        result_hash = self.generate_hash(
            result['token_id'],
            result.get('attack_type', ''),
            result.get('response', '')  # Use original
        )
        
        return NormalizedResult(
            token_id=result['token_id'],
            token_str=result['token_str'],
            payload=payload,
            response=response,
            response_length=original_length,
            result_hash=result_hash
        )
    
    def deduplicate(self) -> List[NormalizedResult]:
        """Remove duplicates by hash"""
        seen_hashes: Set[str] = set()
        deduped: List[NormalizedResult] = []
        
        for result in self.normalized:
            if result.result_hash not in seen_hashes:
                seen_hashes.add(result.result_hash)
                deduped.append(result)
            else:
                self.stats['duplicates_removed'] += 1
        
        return deduped
    
    def normalize_all(self):
        """Normalize all results"""
        print("\nNormalizing results...")
        if self.clean_responses:
            print("   Cleaning repetitive responses...")
        
        for result in self.results:
            normalized = self.normalize_single(result, clean_responses=self.clean_responses)
            self.normalized.append(normalized)
        
        print(f"   Normalized {len(self.normalized)} results")
        
        # Deduplicate
        print("\nDeduplicating...")
        self.normalized = self.deduplicate()
        print(f"   Removed {self.stats['duplicates_removed']} duplicates")
        print(f"   Final count: {len(self.normalized)} unique results")
    
    def analyze_patterns(self):
        """Simple summary of results"""
        print("\n" + "="*80)
        print("NORMALIZATION SUMMARY")
        print("="*80)
        
        total = len(self.normalized)
        print(f"\nTotal unique results: {total}")
        print(f"Duplicates removed: {self.stats['duplicates_removed']}")
    
    def save_normalized(self, output_path: str):
        """Save normalized results"""
        print(f"\nSaving normalized results to {output_path}...")
        
        # Save as simple array without metadata
        output = [r.to_dict() for r in self.normalized]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print("   Saved!")
    
    def generate_report(self, output_path: str):
        """Generate human-readable report"""
        print(f"\nGenerating report to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("GLITCH TOKEN INJECTION TEST REPORT\n")
            f.write("="*80 + "\n\n")
            
            total = len(self.normalized)
            f.write(f"Total unique results: {total}\n")
            f.write(f"Duplicates removed: {self.stats['duplicates_removed']}\n\n")
            
            # All results
            f.write("-"*80 + "\n")
            f.write("ALL RESULTS\n")
            f.write("-"*80 + "\n\n")
            
            for i, result in enumerate(self.normalized, 1):
                f.write(f"\n{i}. Token: {result.token_str} (ID: {result.token_id})\n")
                f.write(f"   Response length: {result.response_length}\n")
                f.write(f"\n   Payload: {result.payload}\n")
                f.write(f"   Response: {result.response}\n")
        
        print("   Report saved!")


def process_single_file(args: tuple[str, bool]) -> tuple[str, bool, str]:
    """Process a single injection file
    
    Args:
        args: tuple of (input_file, clean_responses)
        
    Returns:
        tuple: (input_file, success, message)
    """
    input_file, clean_responses = args
    try:
        # Process single file
        normalizer = ResultNormalizer(clean_responses=clean_responses)
        
        # Convert to Path objects
        input_path = Path(input_file)
        input_dir = input_path.parent
        input_stem = input_path.stem  # filename without extension
        
        # Generate output file names based on input file
        output_json = input_dir / f"normalized_{input_stem}.json"
        output_report = input_dir / f"normalized_{input_stem}.txt"
        
        # Process
        normalizer.load_results(str(input_path))
        normalizer.normalize_all()
        normalizer.analyze_patterns()
        normalizer.save_normalized(str(output_json))
        normalizer.generate_report(str(output_report))
        
        return (input_file, True, f"✓ Completed: {input_stem}\n  JSON: {output_json}\n  Report: {output_report}")
        
    except Exception as e:
        return (input_file, False, f"✗ Error processing {input_file}: {e}")


def main(results_dir: str = "results", clean_responses: bool = True):
    """Main function to process all injection_*.json files in results directory
    
    Args:
        results_dir: Directory containing model results (default: results)
        clean_responses: If True, clean up repetitive responses (default: True)
    """
    # Find all injection_*.json files using pathlib
    results_path = Path(results_dir)
    json_files = list(results_path.rglob("injection_*.json"))
    json_files = [str(f) for f in json_files]  # Convert to strings for compatibility
    
    if not json_files:
        print(f"No injection_*.json files found in {results_dir}/")
        return
    
    print(f"Found {len(json_files)} injection files to process:")
    for file in json_files:
        print(f"  - {file}")
    
    print("\n" + "="*80)
    print("BATCH PROCESSING STARTED (PARALLEL)")
    print(f"Clean responses: {clean_responses}")
    print("="*80)
    
    processed_count = 0
    error_count = 0
    
    # Prepare arguments for parallel processing
    file_args = [(f, clean_responses) for f in json_files]
    
    # Process files in parallel using multiprocessing.Pool
    max_workers = min(4, len(json_files))  # Don't create more workers than files
    with Pool(processes=max_workers) as pool:
        # Process files with real-time progress
        for i, result in enumerate(pool.imap(process_single_file, file_args), 1):
            input_file, success, message = result
            
            print(f"\n[{i}/{len(json_files)}] {input_file}")
            print("-" * 60)
            print(message)
            
            if success:
                processed_count += 1
            else:
                error_count += 1
    
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nProcessed: {processed_count}/{len(json_files)} files successfully")
    if error_count > 0:
        print(f"Errors: {error_count} files failed")


if __name__ == "__main__":
    fire.Fire(main)