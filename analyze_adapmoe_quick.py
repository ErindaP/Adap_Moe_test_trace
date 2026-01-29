"""
Analyse rapide des traces MoE avec et sans AdapMoE (échantillon de 50 prompts)
"""

import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import math

# Poids de sensibilité par layer (issus de run.py)
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


def apply_adaptive_gating(routing_weights: torch.Tensor, layer_idx: int, 
                          base_threshold: float = 0.005, top_k: int = 2) -> torch.Tensor:
    """Applique le mécanisme d'adaptive gating"""
    threshold = get_adaptive_threshold(layer_idx, base_threshold)
    
    # Obtenir les top-k experts
    top_weights, top_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    # Si le 2e expert a un poids < threshold, on ne garde que le 1er
    mask = top_weights[:, 1] < threshold
    
    selected_experts = []
    for i in range(routing_weights.shape[0]):
        if mask[i]:
            selected_experts.append([top_experts[i, 0].item()])
        else:
            selected_experts.append([top_experts[i, 0].item(), top_experts[i, 1].item()])
    
    return selected_experts


def analyze_mixtral_trace_fast(trace_path: str, base_threshold: float = 0.005, n_samples: int = 50):
    """Analyse rapide des traces Mixtral"""
    traces = torch.load(trace_path, weights_only=True)
    
    results = {
        'original': defaultdict(lambda: {'expert_counts': defaultdict(int), 'unique_experts': set()}),
        'adapmoe': defaultdict(lambda: {'expert_counts': defaultdict(int), 'unique_experts': set()})
    }
    
    print(f"\n{'='*80}")
    print(f"ANALYSE MIXTRAL - {n_samples} prompts (sur {len(traces)} disponibles)")
    print(f"{'='*80}\n")
    
    for prompt_idx in range(min(n_samples, len(traces))):
        if prompt_idx % 10 == 0:
            print(f"Traitement prompt {prompt_idx}/{n_samples}...")
        
        trace = traces[prompt_idx]
        n_layers, n_tokens, n_experts = trace.shape
        
        for layer_idx in range(n_layers):
            layer_logits = trace[layer_idx]
            routing_weights = torch.softmax(layer_logits, dim=-1)
            
            # Version originale: top-2 experts
            _, top2_experts = torch.topk(routing_weights, 2, dim=-1)
            
            for token_idx in range(n_tokens):
                for expert_idx in top2_experts[token_idx]:
                    expert_id = expert_idx.item()
                    results['original'][layer_idx]['expert_counts'][expert_id] += 1
                    results['original'][layer_idx]['unique_experts'].add(expert_id)
            
            # Version avec AdapMoE
            selected_experts_list = apply_adaptive_gating(routing_weights, layer_idx, base_threshold)
            
            for token_idx, selected_experts in enumerate(selected_experts_list):
                for expert_id in selected_experts:
                    results['adapmoe'][layer_idx]['expert_counts'][expert_id] += 1
                    results['adapmoe'][layer_idx]['unique_experts'].add(expert_id)
    
    return results


def analyze_llama_trace_fast(trace_path: str, base_threshold: float = 0.005, n_prompts: int = 10):
    """Analyse rapide des traces Llama-MoE"""
    traces = torch.load(trace_path, map_location=torch.device("cpu"), weights_only=True)
    
    results = {
        'original': defaultdict(lambda: {'expert_counts': defaultdict(int), 'unique_experts': set()}),
        'adapmoe': defaultdict(lambda: {'expert_counts': defaultdict(int), 'unique_experts': set()})
    }
    
    print(f"\n{'='*80}")
    print(f"ANALYSE LLAMA-MOE - {n_prompts} prompts")
    print(f"{'='*80}\n")
    
    for prompt_idx in range(min(n_prompts, 10)):
        print(f"Traitement prompt {prompt_idx}/{n_prompts}...")
        
        n_layers = len(traces)
        
        for layer_idx in range(n_layers):
            layer_data = traces[layer_idx][(prompt_idx, 0)]
            n_tokens = len(layer_data)
            
            for token_idx in range(n_tokens):
                token_experts = layer_data[token_idx]
                
                # Version originale: top-4 experts
                for expert_idx in range(4):
                    expert_id = token_experts[expert_idx].item()
                    results['original'][layer_idx]['expert_counts'][expert_id] += 1
                    results['original'][layer_idx]['unique_experts'].add(expert_id)
                
                # Version AdapMoE: réduction adaptative
                threshold = get_adaptive_threshold(layer_idx, base_threshold)
                
                if threshold > 0.2:
                    n_keep = 1
                elif threshold > 0.1:
                    n_keep = 2
                else:
                    n_keep = 3
                
                for expert_idx in range(n_keep):
                    expert_id = token_experts[expert_idx].item()
                    results['adapmoe'][layer_idx]['expert_counts'][expert_id] += 1
                    results['adapmoe'][layer_idx]['unique_experts'].add(expert_id)
    
    return results


def print_comparison_stats(results: dict, model_name: str, n_experts: int):
    """Affiche les statistiques de comparaison"""
    print(f"\n{'='*80}")
    print(f"STATISTIQUES DE COMPARAISON - {model_name}")
    print(f"{'='*80}\n")
    
    n_layers = len(results['original'])
    
    total_activations_original = 0
    total_activations_adapmoe = 0
    
    print(f"{'Layer':<8} {'Orig. Experts':<15} {'AdapMoE Experts':<15} {'Orig. Activ.':<15} {'AdapMoE Activ.':<15} {'Réduction':<12}")
    print("-" * 95)
    
    layer_stats = []
    
    for layer_idx in range(n_layers):
        orig_unique = len(results['original'][layer_idx]['unique_experts'])
        adap_unique = len(results['adapmoe'][layer_idx]['unique_experts'])
        
        orig_activations = sum(results['original'][layer_idx]['expert_counts'].values())
        adap_activations = sum(results['adapmoe'][layer_idx]['expert_counts'].values())
        
        total_activations_original += orig_activations
        total_activations_adapmoe += adap_activations
        
        reduction = ((orig_activations - adap_activations) / orig_activations * 100) if orig_activations > 0 else 0
        
        layer_stats.append({
            'layer': layer_idx,
            'orig_unique': orig_unique,
            'adap_unique': adap_unique,
            'orig_act': orig_activations,
            'adap_act': adap_activations,
            'reduction': reduction
        })
        
        print(f"{layer_idx:<8} {orig_unique:<15} {adap_unique:<15} {orig_activations:<15} {adap_activations:<15} {reduction:>6.1f}%")
    
    print("-" * 95)
    total_reduction = ((total_activations_original - total_activations_adapmoe) / total_activations_original * 100)
    print(f"{'TOTAL':<8} {'':<15} {'':<15} {total_activations_original:<15} {total_activations_adapmoe:<15} {total_reduction:>6.1f}%")
    
    print(f"\n{'='*80}")
    print("RÉSUMÉ")
    print(f"{'='*80}")
    print(f"Réduction globale des activations d'experts: {total_reduction:.2f}%")
    print(f"Économie: {total_activations_original - total_activations_adapmoe:,} activations sur {total_activations_original:,}")
    print(f"Moyenne d'experts activés par token (original): {total_activations_original / sum(s['orig_act'] for s in layer_stats) * 2:.2f}")
    print(f"Moyenne d'experts activés par token (AdapMoE): {total_activations_adapmoe / sum(s['orig_act'] for s in layer_stats) * 2:.2f}")
    
    return layer_stats


def plot_expert_usage(results: dict, model_name: str, n_experts: int, layer_stats: list):
    """Crée des visualisations de l'utilisation des experts"""
    n_layers = len(results['original'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'AdapMoE Analysis - {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Nombre d'experts uniques par layer
    ax = axes[0, 0]
    layers = list(range(n_layers))
    orig_unique = [s['orig_unique'] for s in layer_stats]
    adap_unique = [s['adap_unique'] for s in layer_stats]
    
    x = np.arange(len(layers))
    width = 0.35
    ax.bar(x - width/2, orig_unique, width, label='Original', alpha=0.8, color='#3498db')
    ax.bar(x + width/2, adap_unique, width, label='AdapMoE', alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Number of unique experts', fontsize=12)
    ax.set_title('Unique experts used per layer', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(0, n_layers, 4))
    
    # 2. Nombre total d'activations par layer
    ax = axes[0, 1]
    orig_activations = [s['orig_act'] for s in layer_stats]
    adap_activations = [s['adap_act'] for s in layer_stats]
    
    ax.bar(x - width/2, orig_activations, width, label='Original', alpha=0.8, color='#3498db')
    ax.bar(x + width/2, adap_activations, width, label='AdapMoE', alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Number of activations', fontsize=12)
    ax.set_title('Total activations per layer', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(0, n_layers, 4))
    
    # 3. Pourcentage de réduction par layer
    ax = axes[1, 0]
    reductions = [s['reduction'] for s in layer_stats]
    
    colors = ['#27ae60' if r >= 0 else '#e74c3c' for r in reductions]
    bars = ax.bar(layers, reductions, color=colors, alpha=0.8)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Reduction (%)', fontsize=12)
    ax.set_title('Activation reduction per layer', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(0, n_layers, 4))
    
    # Ajouter une ligne moyenne
    avg_reduction = np.mean(reductions)
    ax.axhline(y=avg_reduction, color='orange', linestyle='--', linewidth=2, label=f'Average: {avg_reduction:.1f}%')
    ax.legend(fontsize=11)
    
    # 4. Distribution de l'utilisation des experts
    ax = axes[1, 1]
    
    # Compter l'utilisation de chaque expert sur toutes les layers
    expert_usage_orig = defaultdict(int)
    expert_usage_adap = defaultdict(int)
    
    for layer_idx in range(n_layers):
        for expert_id, count in results['original'][layer_idx]['expert_counts'].items():
            expert_usage_orig[expert_id] += count
        for expert_id, count in results['adapmoe'][layer_idx]['expert_counts'].items():
            expert_usage_adap[expert_id] += count
    
    expert_ids = sorted(set(list(expert_usage_orig.keys()) + list(expert_usage_adap.keys())))
    orig_counts = [expert_usage_orig[eid] for eid in expert_ids]
    adap_counts = [expert_usage_adap[eid] for eid in expert_ids]
    
    x_experts = np.arange(len(expert_ids))
    ax.bar(x_experts - width/2, orig_counts, width, label='Original', alpha=0.8, color='#3498db')
    ax.bar(x_experts + width/2, adap_counts, width, label='AdapMoE', alpha=0.8, color='#e74c3c')
    ax.set_xlabel('Expert ID', fontsize=12)
    ax.set_ylabel('Total number of activations', fontsize=12)
    ax.set_title('Distribution of expert usage', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_xticks(expert_ids)
    
    plt.tight_layout()
    output_path = f'/root/ENS/M2/GPU/PROJET/AdapMoE/example-traces/analysis_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Graphique sauvegardé: {output_path}")


def main():
    """Fonction principale"""
    # Analyser Mixtral
    mixtral_results = analyze_mixtral_trace_fast(
        "/root/ENS/M2/GPU/PROJET/AdapMoE/example-traces/trace-moe-mixtral.torch",
        base_threshold=0.005,
        n_samples=50
    )
    
    layer_stats_mixtral = print_comparison_stats(mixtral_results, "Mixtral 8x7B", n_experts=8)
    plot_expert_usage(mixtral_results, "Mixtral_8x7B", n_experts=8, layer_stats=layer_stats_mixtral)
    
    # Analyser Llama-MoE
    llama_results = analyze_llama_trace_fast(
        "/root/ENS/M2/GPU/PROJET/AdapMoE/example-traces/trace-moe-llama-helpful_2.torch",
        base_threshold=0.005,
        n_prompts=10
    )
    
    layer_stats_llama = print_comparison_stats(llama_results, "Llama-MoE", n_experts=16)
    plot_expert_usage(llama_results, "Llama-MoE", n_experts=16, layer_stats=layer_stats_llama)
    



if __name__ == "__main__":
    main()
