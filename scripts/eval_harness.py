"""
Evaluation Harness for Voice Attribute Inference Service
Runs inference against Mozilla Common Voice sample datasets and computes accuracy/calibration metrics.
"""

import time
import torch
import numpy as np
from datasets import load_dataset
from app.audio_processor import AudioProcessor

def run_evaluation(num_samples: int = 50):
    print(f"--- Running Evaluation Harness on {num_samples} Mozilla Common Voice Samples ---")
    processor = AudioProcessor()
    
    # Load public validation dataset
    ds = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation", streaming=True)
    
    correct_gender = 0
    total_evaluated = 0
    total_latencies = []

    for sample in ds.take(num_samples):
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        actual_gender = sample.get("gender", "unknown")

        start = time.time()
        gender_res, _, _ = processor.predict_attributes(audio_array, sr)
        latency = (time.time() - start) * 1000
        
        total_latencies.append(latency)
        
        if actual_gender in ["male", "female"] and gender_res["prediction"] == actual_gender:
            correct_gender += 1
            
        total_evaluated += 1

    acc = (correct_gender / total_evaluated) * 100 if total_evaluated > 0 else 0
    avg_latency = np.mean(total_latencies)

    print("\n=== Evaluation Results ===")
    print(f"Total Samples Evaluated: {total_evaluated}")
    print(f"Gender Accuracy: {acc:.2f}%")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print("==========================")

if __name__ == "__main__":
    run_evaluation(num_samples=20)