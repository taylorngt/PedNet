# Pedigree-Network-Analysis
A Python toolkit & Jupyter notebook pipeline for viewing human pedigrees as graphs, computing network-science metrics, and prioritising disease-causal variants by pedigree-aware segregation.

| Feature | Description |
|---------|-------------|
| **PED ➜ NetworkX** | Converts 6-column PED files to directed graphs with phenotype metadata. |
| **Graph visualisation** | Clean pedigree layouts via Graphviz (`dot`) or spring layouts. |
| **Network metrics** | Degree, betweenness, closeness, clustering, diameter, small-world σ, power-law fit, etc. |
| **Inheritance classifier** | Rule-based AD vs AR detection from graph features. |
| **Variant simulation** | Tiny multi-sample VCF generator consistent with AD or AR models. |
| **Network-aware segregation scan** | Scores every variant by edge-consistency, generation continuity, and carrier betweenness to pinpoint the causal allele. |
| **Notebook pipeline** | `pedigree_pipeline_enriched.ipynb` runs end-to-end in <1 min. |

---

## Quick start

```bash
git clone https://github.com/<user>/pedigree-network-analysis.git
cd pedigree-network-analysis
pip install -r requirements.txt        # installs networkx, pandas, powerlaw, pygraphviz, etc.
jupyter lab                             # or `jupyter notebook`
# open `pedigree_pipeline_enriched.ipynb` and Run-All
