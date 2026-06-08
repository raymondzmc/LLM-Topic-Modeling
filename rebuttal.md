We thank the reviewer for the thoughtful and constructive review. We appreciate the recognition of our framework and our extensive evaluation. Below, we address each concern in detail and outline concrete revisions that we believe strengthen the paper substantially.

---

## W1
**Theoretical grounding**: Appendix I provides a formal justification of our loss function. Specifically, we show that minimizing our objective (Equation 3) corresponds to a regularized KL projection of the LM-induced semantic distribution onto the constrained hypothesis space of distributions representable by ProdLDA. The reconstruction term finds the closest approximation to the LM's semantic distribution (Equation 8), while the prior regularization term preserves the probabilistic topic structure and prevents latent-space collapse (Equation 9). We will move a condensed version into the main text given the extra space in the final version.

**Connection to broader principles**: We will additionally discuss how our framework relates to the information bottleneck principle ([Tishby et al., 2000](https://arxiv.org/abs/physics/0004057)): BoW reconstruction targets force the topic model to compress documents through a bottleneck that preserves lexical statistics, whereas our soft label targets redirect this compression toward preserving semantic structure. This reframing provides a principled explanation for why our topics better align with human-annotated labels.

---

## W2

We thank the reviewer for raising this important point. We provide a rigorous analysis below and will incorporate it into the revision.

On 20NewsGroup, our best model achieves I-RBO = 0.993 compared to 0.999 for ECRTM (best baseline in terms of IRBO). This .006 absolute difference means that, out of the top-15 words per topic, the overlap between any two topics increases by `fewer than one word on average`. In real corpora, words could be relevant to multiple themes (e.g., "science" is relevant to atheism-debates, physics, and biology). We argue that it is more important for the  topics to be semantically meaningful than whether they share zero overlapping words

It's also worth noting that our approach is based on the ProdLDA architecture, the `I-RBO score is essentially identical to the ProdLDA family` (within ±0.001 on every dataset), where "degradation" is really only relative to ECRTM (0.999–1.000) and FASTopic (0.999), both of which explicitly optimize for diversity.

| Method | 20News | Tweet | Stack |
|---|---|---|---|
| ProdLDA | .992 | .994 | .991 |
| CombinedTM | .993 | .988 | .986 |
| ZeroshotTM | .994 | .994 | .993 |
| **Ours (Llama-3.2-1B)** | .993 | .993 | .993 |

Last but not least, we find that our method can also be adapted to FASTopic to maintain perfect diversity. Table below shows the results on TweetTopic using `ERNIE-0.3B`.
| Model | CV $\uparrow$ | LLM $\uparrow$ | IRBO $\uparrow$ | Purity $\uparrow$ |
|:---|:---:|:---:|:---:|:---:|
| ProdLDA | 0.355 | 2.11 | 0.994 | 0.533 |
| ECRTM | 0.356 | 1.85 | 1.00 | 0.399 |
| FASTopic | 0.276 | 1.96 | 0.626 | 0.557 |
| ProdLDA (Ours) | 0.392 | 2.90 | 0.989 | 0.781 |
| FASTopic (Ours) | 0.361 | 2.10 | 1.00 | 0.697 |

However, we find that while it `significantly improves FASTopic across all metrics`, using the ProdLDA backbone achieves the best overall performance trade-off. We find similar trends for the other two datasets, where the complete results will be incorporated into the final draft.

---

## W3

Although we agree with the reviewer, we would like to point out that `CEMTM is already included in Section 2.1`. We will include an extensive list of graph and contextual-based topic model in the final draft given the extra space. We also welcome any suggestions from the reviewer.


---

### Question 1:



### Question 2:
We sincerely thank the reviewer for pointing this out, which is an oversight on our part. This was due to inconsistency of the graphing tools used. In the final version, we will: (a) replace Figure 1 with a high-resolution vector graphic (PDF/SVG format) to ensure crisp rendering at any zoom level; (b) increase the information density by adding numeric probability values for the most prominent words; (c) fix the erroneous bibliography hyperlink; and (d) conduct a thorough pass for formatting consistency throughout the manuscript. We appreciate the reviewer's attention to presentation quality.


