
import torch
import numpy as np
from collections import defaultdict
import math

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
    weight = LAYER_SENSITIVITY_WEIGHTS[layer_idx]
    return math.sqrt(base_threshold / weight)

def analyze_token_level_reduction():
    """Analyse détaillée montrant la réduction au niveau token"""
    
    traces = torch.load("/root/ENS/M2/GPU/PROJET/AdapMoE/example-traces/trace-moe-mixtral.torch", 
                       weights_only=True)
    
    print("\n" + "="*80)
    print("VÉRIFICATION : Réduction au niveau TOKEN vs LAYER")
    print("="*80 + "\n")
    
    # Analyser quelques layers spécifiques
    layers_to_check = [0, 15, 30]  # Première, milieu, dernière
    prompt_idx = 0
    trace = traces[prompt_idx]
    
    for layer_idx in layers_to_check:
        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx} (threshold = {get_adaptive_threshold(layer_idx):.4f})")
        print(f"{'='*80}\n")
        
        layer_logits = trace[layer_idx]  # (n_tokens, n_experts)
        routing_weights = torch.softmax(layer_logits, dim=-1)
        n_tokens = routing_weights.shape[0]
        
        # Statistiques
        experts_per_token_original = []
        experts_per_token_adapmoe = []
        all_experts_used_original = set()
        all_experts_used_adapmoe = set()
        
        threshold = get_adaptive_threshold(layer_idx)
        
        # Analyser les 10 premiers tokens
        print(f"Analyse des {min(10, n_tokens)} premiers tokens:\n")
        print(f"{'Token':<8} {'Original':<25} {'AdapMoE':<25} {'Réduction'}")
        print("-" * 80)
        
        for token_idx in range(min(10, n_tokens)):
            # Original : top-2
            top_weights, top_experts = torch.topk(routing_weights[token_idx], 2, dim=-1)
            orig_experts = [top_experts[i].item() for i in range(2)]
            experts_per_token_original.append(2)
            all_experts_used_original.update(orig_experts)
            
            # AdapMoE : vérifier le threshold
            if top_weights[1] < threshold:
                adap_experts = [top_experts[0].item()]
                n_experts = 1
            else:
                adap_experts = [top_experts[i].item() for i in range(2)]
                n_experts = 2
            
            experts_per_token_adapmoe.append(n_experts)
            all_experts_used_adapmoe.update(adap_experts)
            
            reduction = "↓ 1 expert" if n_experts == 1 else "= 2 experts"
            
            print(f"#{token_idx:<7} {str(orig_experts):<25} {str(adap_experts):<25} {reduction}")
        
        # Statistiques globales pour toute la layer
        print(f"\n{'─'*80}")
        print(f"STATISTIQUES SUR TOUS LES {n_tokens} TOKENS:\n")
        
        # Compter pour tous les tokens
        tokens_with_1_expert = 0
        tokens_with_2_experts = 0
        all_experts_original = set()
        all_experts_adapmoe = set()
        
        for token_idx in range(n_tokens):
            top_weights, top_experts = torch.topk(routing_weights[token_idx], 2, dim=-1)
            all_experts_original.update([top_experts[i].item() for i in range(2)])
            
            if top_weights[1] < threshold:
                tokens_with_1_expert += 1
                all_experts_adapmoe.add(top_experts[0].item())
            else:
                tokens_with_2_experts += 1
                all_experts_adapmoe.update([top_experts[i].item() for i in range(2)])
        
        print(f"Répartition AdapMoE:")
        print(f"  - Tokens avec 1 expert : {tokens_with_1_expert:>4} ({tokens_with_1_expert/n_tokens*100:.1f}%)")
        print(f"  - Tokens avec 2 experts: {tokens_with_2_experts:>4} ({tokens_with_2_experts/n_tokens*100:.1f}%)")
        print(f"\nExperts uniques utilisés:")
        print(f"  - Original : {len(all_experts_original)} experts → {sorted(all_experts_original)}")
        print(f"  - AdapMoE  : {len(all_experts_adapmoe)} experts → {sorted(all_experts_adapmoe)}")
        print(f"\nActivations totales:")
        print(f"  - Original : {n_tokens * 2:>5} (tous les tokens × 2 experts)")
        print(f"  - AdapMoE  : {tokens_with_1_expert + tokens_with_2_experts * 2:>5} (réduction de {(1 - (tokens_with_1_expert + tokens_with_2_experts * 2)/(n_tokens * 2)) * 100:.1f}%)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    analyze_token_level_reduction()
