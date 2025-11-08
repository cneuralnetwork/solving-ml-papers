import time
import numpy as np
import matplotlib.pyplot as plt
from spec import SpecDecoder

def benchmark_methods(decoder, prompt, max_new_tok, num_sample_tok=None):
    start_time = time.time()
    normal_output = decoder.autoregressive_gen(prompt, max_new_tok)
    normal_time = time.time() - start_time
    
    if num_sample_tok is not None:
        start_time = time.time()
        spec_output = decoder.spec_gen(prompt, max_new_tok, num_sample_tok)
        spec_time = time.time() - start_time
        return normal_output, normal_time, spec_output, spec_time
    
    return normal_output, normal_time, None, None

def run_num_sample_sweep():
    decoder = SpecDecoder('gpt2', 'gpt2-xl')
    prompt = "Artificial intelligence is"
    
    max_new_tok = 50  # fixed for all runs
    num_sample_tok_values = [1, 2, 3, 4, 5, 8]
    
    speedups = []
    spec_times = []
    for sample_tok in num_sample_tok_values:
        print(f"Testing num_sample_tok={sample_tok}")
        _, normal_time, _, _ = benchmark_methods(decoder, prompt, max_new_tok)
        _, _, _, spec_time = benchmark_methods(decoder, prompt, max_new_tok, sample_tok)
        
        speedup = normal_time / spec_time if spec_time > 0 else 0
        speedups.append(speedup)
        spec_times.append(spec_time)
        
        print(f"  Normal decoding: {normal_time:.3f}s, Speculative ({sample_tok}): {spec_time:.3f}s, Speedup: {speedup:.2f}x")

    return {'num_sample_tok': num_sample_tok_values, 'speedup': speedups, 'spec_time': spec_times}

def plot_speedup_vs_num_sample(results):
    num_sample_tok = np.array(results['num_sample_tok'])
    speedup = np.array(results['speedup'])
    spec_times = np.array(results['spec_time'])

    plt.figure(figsize=(8, 6))
    plt.plot(num_sample_tok, speedup, 'o-', color='blue')
    plt.xlabel("num_sample_tok")
    plt.ylabel("Speedup (x)")
    plt.title("Speedup vs Number of Samples Taken (fixed max_new_tok=50)")
    plt.grid(True, alpha=0.3)
    for x, y in zip(num_sample_tok, speedup):
        plt.text(x, y, f"{y:.2f}x", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig('speedup_vs_num_sample_tok.png', dpi=300, bbox_inches='tight')
    # plt.show()

def main():
    print("Speculative Decoding Speedup by Number of Samples Taken")
    print("=" * 50)
    
    results = run_num_sample_sweep()
    
    print("\nGenerating plot...")
    plot_speedup_vs_num_sample(results)
    
    max_speedup_idx = np.argmax(results['speedup'])
    print(f"\nBest number of samples: {results['num_sample_tok'][max_speedup_idx]}")
    print(f"Speedup: {results['speedup'][max_speedup_idx]:.2f}x")
    print(f"Speculative time: {results['spec_time'][max_speedup_idx]:.3f}s")
    avg_speedup = np.mean(results['speedup'])
    print(f"\nAverage speedup across num_sample_tok: {avg_speedup:.2f}x")

if __name__ == "__main__":
    main()