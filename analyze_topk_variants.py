"""
Analyse AdapMoE avec différents patterns top-k
Compare l'impact d'AdapMoE avec différentes configurations d'experts
"""

import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import math

# Poids de sensibilité par layer
LAYER_SENSITIVITY_WEIGHTS = [
    46.69189453125, 17.303466796875, 13.0157470703125, 7.640838623046875, 
    4.169464111328125, 2.2296905517578125, 1.2559890747070312, 0.8444786071777344, 
    0.6837844848632812, 0.5602836608886719, 0.5125999450683594, 0.4780292510986328, 
    0.44536590576171875, 0.4355907440185547, 0.38361549377441406, 0.30994415283203125, 
    0.23305416107177734, 0.1760721206665039, 0.13840198516845703, 0.1137852668762207, 
    0.10472536087036133, 0.09542703628540039, 0.08624792098999023, 0.07712841033935547, 
    0.06937980651855469, 0.06109476089477539, 0.0502467155456543, 0.042557716369628906, 
    0.03349781036376953, 0.025272369384765625, 0.020682811737060547, 0.02294778823852539
]


def get_adaptive_threshold(layer_idx: int, base_threshold: float = 0.005) -> float:
    """Calcule le threshold adaptatif pour une layer"""
    weight = LAYER_SENSITIVITY_WEIGHTS[layer_idx]
    return math.sqrt(base_threshold / weight)


def apply_adaptive_gating_topk(routing_weights: torch.Tensor, layer_idx: int, 
                                 base_threshold: float, top_k: int) -> list:
    """
    Applique le mécanisme d'adaptive gating avec un top-k configurable
    """
    threshold = get_adaptive_threshold(layer_idx, base_threshold)
    
    # Obtenir les top-k experts
    top_weights, top_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    # Appliquer le threshold: si le dernier expert a un poids < threshold, réduire k
    selected_experts = []
    for i in range(routing_weights.shape[0]):
        # Chercher le dernier expert qui dépasse le threshold
        n_keep = top_k
        for j in range(top_k - 1, 0, -1):
            if top_weights[i, j] < threshold:
                n_keep = j
            else:
                break
        
        selected_experts.append([top_experts[i, k].item() for k in range(n_keep)])
    
    return selected_experts


def analyze_mixtral_with_topk(trace_path: str, top_k_values: list, n_samples: int = 50, base_threshold: float = 0.005):
    """
    Analyse Mixtral avec différentes valeurs de top-k
    """
    traces = torch.load(trace_path, weights_only=True)
    
    results = {}
    
    for top_k in top_k_values:
        print(f"\n{'='*80}")
        print(f"MIXTRAL - Top-{top_k} configuration")
        print(f"{'='*80}\n")
        
        results[top_k] = {
            'original': defaultdict(lambda: {'activations': 0}),
            'adapmoe': defaultdict(lambda: {'activations': 0})
        }
        
        for prompt_idx in range(min(n_samples, len(traces))):
            if prompt_idx % 10 == 0:
                print(f"  Processing prompt {prompt_idx}/{n_samples}...")
            
            trace = traces[prompt_idx]
            n_layers, n_tokens, n_experts = trace.shape
            
            for layer_idx in range(n_layers):
                layer_logits = trace[layer_idx]
                routing_weights = torch.softmax(layer_logits, dim=-1)
                
                # Original: top-k experts
                _, topk_experts = torch.topk(routing_weights, top_k, dim=-1)
                orig_count = n_tokens * top_k
                results[top_k]['original'][layer_idx]['activations'] += orig_count
                
                # AdapMoE: gating adaptatif
                selected_experts_list = apply_adaptive_gating_topk(routing_weights, layer_idx, base_threshold, top_k)
                adap_count = sum(len(experts) for experts in selected_experts_list)
                results[top_k]['adapmoe'][layer_idx]['activations'] += adap_count
        
    
    return results


def analyze_llama_with_topk(trace_path: str, top_k_values: list, n_prompts: int = 10, base_threshold: float = 0.005):
    """
    Analyse Llama-MoE avec différentes valeurs de top-k
    """
    traces = torch.load(trace_path, map_location=torch.device("cpu"), weights_only=True)
    
    results = {}
    
    for top_k in top_k_values:
        print(f"\n{'='*80}")
        print(f"LLAMA-MOE - Top-{top_k} configuration")
        print(f"{'='*80}\n")
        
        results[top_k] = {
            'original': defaultdict(lambda: {'activations': 0}),
            'adapmoe': defaultdict(lambda: {'activations': 0})
        }
        
        for prompt_idx in range(min(n_prompts, 10)):
            print(f"  Processing prompt {prompt_idx}/{n_prompts}...")
            
            n_layers = len(traces)
            
            for layer_idx in range(n_layers):
                layer_data = traces[layer_idx][(prompt_idx, 0)]
                n_tokens = len(layer_data)
                
                # Original: top-k experts
                orig_count = n_tokens * top_k
                results[top_k]['original'][layer_idx]['activations'] += orig_count
                
                # AdapMoE: réduction adaptative (heuristique basée sur threshold)
                threshold = get_adaptive_threshold(layer_idx, base_threshold)
                
                # Heuristique pour déterminer combien d'experts garder
                if threshold > 0.3:
                    n_keep = max(1, top_k // 4)
                elif threshold > 0.2:
                    n_keep = max(1, top_k // 3)
                elif threshold > 0.1:
                    n_keep = max(1, top_k // 2)
                else:
                    n_keep = max(1, int(top_k * 0.75))
                
                adap_count = n_tokens * n_keep
                results[top_k]['adapmoe'][layer_idx]['activations'] += adap_count
        
    
    return results


def generate_comparison_table(results: dict, model_name: str):
    """
    Génère un tableau comparatif des résultats
    """
    print(f"\n{'='*80}")
    print(f"COMPARATIVE RESULTS - {model_name}")
    print(f"{'='*80}\n")
    
    print(f"{'Top-k':<10} {'Original':<15} {'AdapMoE':<15} {'Reduction':<12} {'Avg Exp/Token'}")
    print("-" * 75)
    
    summary = {}
    
    for top_k in sorted(results.keys()):
        orig_total = sum(results[top_k]['original'][l]['activations'] for l in results[top_k]['original'])
        adap_total = sum(results[top_k]['adapmoe'][l]['activations'] for l in results[top_k]['adapmoe'])
        reduction = (1 - adap_total / orig_total) * 100 if orig_total > 0 else 0
        
        # Calculer le nombre moyen d'experts par token
        n_layers = len(results[top_k]['original'])
        n_tokens_total = orig_total / (n_layers * top_k)
        avg_experts_orig = top_k
        avg_experts_adap = adap_total / (n_tokens_total * n_layers)
        
        print(f"{top_k:<10} {orig_total:<15,} {adap_total:<15,} {reduction:>6.2f}%     {avg_experts_orig:.1f} → {avg_experts_adap:.1f}")
        
        summary[top_k] = {
            'original': orig_total,
            'adapmoe': adap_total,
            'reduction': reduction,
            'avg_orig': avg_experts_orig,
            'avg_adap': avg_experts_adap
        }
    
    return summary


def plot_topk_comparison(mixtral_results: dict, llama_results: dict):
    """
    Crée des visualisations comparatives pour différents top-k
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('AdapMoE Analysis: Impact of Top-K Configuration', fontsize=16, fontweight='bold')
    
    # 1. Réduction globale par top-k (Mixtral)
    ax = axes[0, 0]
    mixtral_topks = sorted(mixtral_results.keys())
    mixtral_reductions = []
    
    for top_k in mixtral_topks:
        orig = sum(mixtral_results[top_k]['original'][l]['activations'] for l in mixtral_results[top_k]['original'])
        adap = sum(mixtral_results[top_k]['adapmoe'][l]['activations'] for l in mixtral_results[top_k]['adapmoe'])
        reduction = (1 - adap / orig) * 100
        mixtral_reductions.append(reduction)
    
    ax.bar(range(len(mixtral_topks)), mixtral_reductions, color='#3498db', alpha=0.8)
    ax.set_xlabel('Top-K Configuration', fontsize=12)
    ax.set_ylabel('Reduction (%)', fontsize=12)
    ax.set_title('Mixtral 8x7B: Reduction by Top-K', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(mixtral_topks)))
    ax.set_xticklabels([f'Top-{k}' for k in mixtral_topks])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Réduction globale par top-k (Llama)
    ax = axes[0, 1]
    llama_topks = sorted(llama_results.keys())
    llama_reductions = []
    
    for top_k in llama_topks:
        orig = sum(llama_results[top_k]['original'][l]['activations'] for l in llama_results[top_k]['original'])
        adap = sum(llama_results[top_k]['adapmoe'][l]['activations'] for l in llama_results[top_k]['adapmoe'])
        reduction = (1 - adap / orig) * 100
        llama_reductions.append(reduction)
    
    ax.bar(range(len(llama_topks)), llama_reductions, color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Top-K Configuration', fontsize=12)
    ax.set_ylabel('Reduction (%)', fontsize=12)
    ax.set_title('Llama-MoE: Reduction by Top-K', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(llama_topks)))
    ax.set_xticklabels([f'Top-{k}' for k in llama_topks])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Activations totales (Mixtral)
    ax = axes[1, 0]
    width = 0.35
    x = np.arange(len(mixtral_topks))
    
    mixtral_orig = [sum(mixtral_results[k]['original'][l]['activations'] for l in mixtral_results[k]['original']) 
                    for k in mixtral_topks]
    mixtral_adap = [sum(mixtral_results[k]['adapmoe'][l]['activations'] for l in mixtral_results[k]['adapmoe']) 
                    for k in mixtral_topks]
    
    ax.bar(x - width/2, mixtral_orig, width, label='Original', alpha=0.8, color='#3498db')
    ax.bar(x + width/2, mixtral_adap, width, label='AdapMoE', alpha=0.8, color='#27ae60')
    ax.set_xlabel('Top-K Configuration', fontsize=12)
    ax.set_ylabel('Total Activations', fontsize=12)
    ax.set_title('Mixtral 8x7B: Total Activations', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{k}' for k in mixtral_topks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Activations totales (Llama)
    ax = axes[1, 1]
    x = np.arange(len(llama_topks))
    
    llama_orig = [sum(llama_results[k]['original'][l]['activations'] for l in llama_results[k]['original']) 
                  for k in llama_topks]
    llama_adap = [sum(llama_results[k]['adapmoe'][l]['activations'] for l in llama_results[k]['adapmoe']) 
                  for k in llama_topks]
    
    ax.bar(x - width/2, llama_orig, width, label='Original', alpha=0.8, color='#e74c3c')
    ax.bar(x + width/2, llama_adap, width, label='AdapMoE', alpha=0.8, color='#27ae60')
    ax.set_xlabel('Top-K Configuration', fontsize=12)
    ax.set_ylabel('Total Activations', fontsize=12)
    ax.set_title('Llama-MoE: Total Activations', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{k}' for k in llama_topks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = '/root/ENS/M2/GPU/PROJET/AdapMoE/example-traces/topk_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')


def main():
    """
    Fonction principale
    """
    print("ADAPMOE ANALYSIS WITH VARIABLE TOP-K CONFIGURATIONS")
    
    # Analyser Mixtral avec top-2, top-3, top-4
    mixtral_topk_values = [2, 3, 4]
    mixtral_results = analyze_mixtral_with_topk(
        "/root/ENS/M2/GPU/PROJET/AdapMoE/example-traces/trace-moe-mixtral.torch",
        top_k_values=mixtral_topk_values,
        n_samples=50
    )
    
    mixtral_summary = generate_comparison_table(mixtral_results, "Mixtral 8x7B")
    
    # Analyser Llama-MoE avec top-4, top-8, top-12
    llama_topk_values = [4, 8, 12]
    llama_results = analyze_llama_with_topk(
        "/root/ENS/M2/GPU/PROJET/AdapMoE/example-traces/trace-moe-llama-helpful_2.torch",
        top_k_values=llama_topk_values,
        n_prompts=10
    )
    
    llama_summary = generate_comparison_table(llama_results, "Llama-MoE")
    
    # Créer les visualisations
    plot_topk_comparison(mixtral_results, llama_results)
    

    
    return mixtral_summary, llama_summary


if __name__ == "__main__":
    mixtral_summary, llama_summary = main()
