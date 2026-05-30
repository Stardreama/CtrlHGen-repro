 # S3 DBpedia50 db50_s100 Server Run

  Environment: AutoDL RTX 4090, torch 2.6.0+cu124

  Scale:
  - train: 7150
  - valid: 884
  - test: 884

  Training:
  - unconditional: epoch 20
  - pattern: epoch 40

  Full baseline:
  - count: 884
  - F1: 0.0110
  - Smatch: 0.0816
  - Jaccard: 0.0105
  - Dice: 0.0110
  - Overlap: 0.0131
  - Validity: 0.0973
  - Entity-number: 0.1505
  - Relation-number: 0.0995

  Rerank_k=4, test_proportion=0.1:
  - count: 88
  - F1: 0.0329
  - Smatch: 0.1304
  - Jaccard: 0.0298
  - Dice: 0.0329
  - Overlap: 0.0353
  - Validity: 0.1818
  - Entity-number: 0.2045
  - Relation-number: 0.1818

  Rerank_k=4, test_proportion=0.25:
  - count: 221
  - F1: 0.0202
  - Smatch: 0.0823
  - Jaccard: 0.0182
  - Dice: 0.0202
  - Overlap: 0.0267
  - Validity: 0.1267
  - Entity-number: 0.1357
  - Relation-number: 0.1267

  Full rerank_k=4:
  - count: 884
  - candidate rows: 3536
  - F1: 0.0229
  - Smatch: 0.0954
  - Jaccard: 0.0219
  - Dice: 0.0229
  - Overlap: 0.0277
  - Validity: 0.1357
  - Entity-number: 0.1538
  - Relation-number: 0.1369

  Conclusion:
  - S3 full rerank improves all main metrics over full baseline.
  - Answer-set metrics become clearly non-zero at S3 scale.
  - This stage is suitable as a reportable supervised reproduction result.
