# S2 DBpedia50 db50_s500 Server Run

  Environment: AutoDL RTX 4090, torch 2.6.0+cu124

  Scale:
  - train: 1430
  - valid: 169
  - test: 169

  Training:
  - unconditional: epoch 20
  - pattern: epoch 40

  Full baseline:
  - count: 169
  - Smatch: 0.0633
  - Jaccard: 0.0000
  - Dice: 0.0000
  - Overlap: 0.0000
  - Validity: 0.0414
  - Entity-number: 0.1302
  - Relation-number: 0.0651

  Full rerank_k=4:
  - count: 169
  - candidate rows: 676
  - Smatch: 0.0978
  - Jaccard: 0.0001
  - Dice: 0.0001
  - Overlap: 0.0090
  - Validity: 0.0888
  - Entity-number: 0.1183
  - Relation-number: 0.1243

  Conclusion:
  - S2 full rerank improves Smatch, Validity, Relation-number, and introduces non-zero answer-set metrics.
  - Entity-number is slightly lower than baseline.
