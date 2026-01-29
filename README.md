# Capacity without Access: Reinterpreting the Mid-Depth Spectral Plateau in LLMs
(Jan 12, 2026) First Version

The implementation of paper

This is the PyTorch code of the REFORM. The code has been tested on PyTorch 2.1.2.
Our DDI experiments on Commonsense Reasoning and Arithmetic Reasoning.

## Abstract 

Prior probing-based analyses show that individual layers specialize in distinct linguistic and semantic functions. A complementary line of work observes that deeper layers enter an alignment-dominated regime, suggesting that residual updates become largely collinear with the hidden state vectors. These findings appear to conflict: layer-wise functional specialization versus apparent representational stagnation in deeper layers. This raises a key question: {Is the observed representational stagnation in deeper layers attributable to suboptimal acquisition or encoding of novel features, or are these features learned appropriately but exhibit minimal marginal contribution to the model’s predictive output?} We address this by separating model behavior into (i) {representational capacity}: the richness and spectral diversity of the encoded features, and (ii) {accessibility}: the extent to which these features are aligned with, and exploited by, the output-relevant subspace. Analyzing hidden-state covariance across depth shows that intermediate layers maintain a broad representational span and rich spectral diversity, indicating that their representational capacity remains largely intact. Yet they project only weakly onto output-relevant subspace, indicating that their accessibility within task-relevant representational dimensions is tightly constrained. To probe their functional relevance, we add a minimal diagnostic pathway that routes intermediate-layer embeddings to the terminal readout. Analysis of the induced behavioral changes indicates that the apparent stagnation is better accounted for by geometric limitations on accessibility, rather than by a deficit in representational capacity.
