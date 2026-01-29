"""
Heatmap de l'utilisation des experts par layer pour Mixtral
Avant et après AdapMoE
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def create_expert_usage_heatmaps(trace_path: str, n_samples: int = 50):
    """Crée les heatmaps d'utilisation des experts avant et après AdapMoE"""
    
    traces = torch.load(trace_path, weights_only=True)
    n_layers = 32
    n_experts = 8
    
    # Matrices pour compter l'utilisation de chaque expert dans chaque layer
    usage_original = np.zeros((n_experts, n_layers))
    usage_adapmoe = np.zeros((n_experts, n_layers))
    
    print(f"\nProcessing {n_samples} prompts for Mixtral 8x7B...")
    
    for prompt_idx in range(min(n_samples, len(traces))):
        if prompt_idx % 10 == 0:
            print(f"  Prompt {prompt_idx}/{n_samples}...")
        
        trace = traces[prompt_idx]
        n_layers_trace, n_tokens, n_experts_trace = trace.shape
        
        for layer_idx in range(n_layers):
            layer_logits = trace[layer_idx]
            routing_weights = torch.softmax(layer_logits, dim=-1)
            threshold = get_adaptive_threshold(layer_idx)
            
            # Original : top-2 experts pour chaque token
            _, top2_experts = torch.topk(routing_weights, 2, dim=-1)
            
            for token_idx in range(n_tokens):
                # Original
                for expert_idx in top2_experts[token_idx]:
                    expert_id = expert_idx.item()
                    usage_original[expert_id, layer_idx] += 1
                
                # AdapMoE : vérifier le threshold
                top_weights, top_experts = torch.topk(routing_weights[token_idx], 2, dim=-1)
                
                if top_weights[1] < threshold:
                    # Seulement le 1er expert
                    usage_adapmoe[top_experts[0].item(), layer_idx] += 1
                else:
                    # Les 2 experts
                    usage_adapmoe[top_experts[0].item(), layer_idx] += 1
                    usage_adapmoe[top_experts[1].item(), layer_idx] += 1
    
    
    # Créer la figure avec 2 heatmaps côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Utiliser la même échelle pour les deux graphiques
    vmin = 0
    vmax = max(usage_original.max(), usage_adapmoe.max())
    
    # Heatmap Original
    sns.heatmap(usage_original, ax=ax1, cmap='YlOrRd', vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Number of activations'}, linewidths=0.5, linecolor='white')
    ax1.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Expert ID', fontsize=14, fontweight='bold')
    ax1.set_title('Original (Top-2 experts per token)', fontsize=15, fontweight='bold', pad=20)
    ax1.set_xticks(np.arange(0, n_layers, 4) + 0.5)
    ax1.set_xticklabels(np.arange(0, n_layers, 4))
    ax1.set_yticks(np.arange(n_experts) + 0.5)
    ax1.set_yticklabels(range(n_experts))
    
    # Heatmap AdapMoE
    sns.heatmap(usage_adapmoe, ax=ax2, cmap='YlOrRd', vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Number of activations'}, linewidths=0.5, linecolor='white')
    ax2.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Expert ID', fontsize=14, fontweight='bold')
    ax2.set_title('AdapMoE (Adaptive gating)', fontsize=15, fontweight='bold', pad=20)
    ax2.set_xticks(np.arange(0, n_layers, 4) + 0.5)
    ax2.set_xticklabels(np.arange(0, n_layers, 4))
    ax2.set_yticks(np.arange(n_experts) + 0.5)
    ax2.set_yticklabels(range(n_experts))
    
    plt.suptitle(f'Expert Usage Heatmap - Mixtral 8x7B ({n_samples} prompts)', 
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = '/root/ENS/M2/GPU/PROJET/AdapMoE/example-traces/heatmap_mixtral_experts.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # Afficher les statistiques
    print("STATISTICS")
    print(f"\nTotal activations:")
    print(f"  Original : {usage_original.sum():,.0f}")
    print(f"  AdapMoE  : {usage_adapmoe.sum():,.0f}")
    print(f"  Reduction: {(1 - usage_adapmoe.sum()/usage_original.sum())*100:.2f}%")
    
    print(f"\nMost solicited experts (Original):")
    for expert_id in np.argsort(-usage_original.sum(axis=1))[:3]:
        print(f"  Expert {expert_id}: {usage_original[expert_id].sum():,.0f} activations")
    
    print(f"\nMost solicited experts (AdapMoE):")
    for expert_id in np.argsort(-usage_adapmoe.sum(axis=1))[:3]:
        print(f"  Expert {expert_id}: {usage_adapmoe[expert_id].sum():,.0f} activations")
    
    print(f"\nBiggest reduction by layer:")
    reductions = (usage_original.sum(axis=0) - usage_adapmoe.sum(axis=0)) / usage_original.sum(axis=0) * 100
    for layer_idx in np.argsort(-reductions)[:5]:
        print(f"  Layer {layer_idx:2d}: {reductions[layer_idx]:.1f}% reduction")
    


def main():
    create_expert_usage_heatmaps(
        "/root/ENS/M2/GPU/PROJET/AdapMoE/example-traces/trace-moe-mixtral.torch",
        n_samples=50
    )


if __name__ == "__main__":
    main()
