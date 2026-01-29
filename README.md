# AdapMoE Example Traces - Experiment Scripts

This directory contains the execution traces and analysis scripts for the experiments presented in the AdapMoE report.

## Traces

- `trace-moe-mixtral.torch` - Mixtral 8x7B execution traces 
- `trace-moe-llama-helpful_2.torch` - Llama-MoE execution traces 

## Analysis Scripts

### Main Experiments

**analyze_adapmoe_quick.py**
- Generates: `analysis_mixtral_8x7b.png`, `analysis_llama-moe.png`
- Analyzes AdapMoE impact on expert activations
- Creates 4 subplots: unique experts, total activations, reduction %, expert distribution
- Used for: Figure 1 and Appendix Figure in report

**analyze_topk_variants.py**
- Generates: `topk_comparison.png`
- Compares different top-k configurations (2/3/4 for Mixtral, 4/8/12 for Llama-MoE)
- Creates 2x2 comparison charts
- Used for: Figure 3 in report (Section on Pattern Influence)

**heatmap_expert_usage.py**
- Generates: `heatmap_mixtral_experts.png`
- Visualizes expert utilization intensity across layers
- Creates side-by-side heatmaps (Original vs AdapMoE)
- Used for: Figure 2 in report

**verify_token_level_reduction.py**
- No graphical output
- Generates text analysis showing token-level reduction
- Used for: Table in Appendix (Token-level Analysis)

## Generated Figures

- `analysis_mixtral_8x7b.png` - Main analysis for Mixtral (Figure 1)
- `analysis_llama-moe.png` - Main analysis for Llama-MoE (Appendix)
- `heatmap_mixtral_experts.png` - Expert utilization heatmap (Figure 2)
- `topk_comparison.png` - Top-K configuration comparison (Figure 3)

## Usage

```bash
# Generate all main figures
python analyze_adapmoe_quick.py

# Generate top-k comparison
python analyze_topk_variants.py

# Generate heatmap
python heatmap_expert_usage.py

# Run token-level verification
python verify_token_level_reduction.py
```
