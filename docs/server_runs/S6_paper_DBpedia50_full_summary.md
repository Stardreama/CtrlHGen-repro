 # S6 Paper-Supervised DBpedia50 full Server Run

  ## Scale

  - Dataset: DBpedia50
  - Scale: full
  - Arity: 32
  - Train rows: 715104
  - Valid rows: 89388
  - Test rows: 89388

  ## Training

  - Unconditional checkpoint: epoch 400
  - Pattern checkpoint: epoch 450
  - Batch size: 256
  - Learning rate: 1e-5
  - Unconditional warm-up: 50 epochs
  - Pattern warm-up: 5 epochs
  - Pattern stage resets optimizer/scheduler from unconditional checkpoint

  ## Baseline Full Test

  - Count: 89388
  - F1: 0.1132672241
  - Smatch: 0.1095997059
  - Jaccard: 0.1109273697
  - Dice: 0.1132672241
  - Overlap: 0.1188436832
  - Validity: 0.1241777420
  - Entity-number: 0.1272094688
  - Relation-number: 0.1251622142

  ## Rerank Full Test

  - Rerank k: 4
  - Alpha: 0.5
  - Count: 89388
  - Candidate rows: 357552
  - F1: 0.1250566549
  - Smatch: 0.1176015176
  - Jaccard: 0.1229843423
  - Dice: 0.1250566549
  - Overlap: 0.1301464186
  - Validity: 0.1309124267
  - Entity-number: 0.1326688146
  - Relation-number: 0.1318521502

  ## Conclusion

  The paper-supervised DBpedia50 full run uses a training setup closer to the paper than S6: 400 unconditional
  epochs, 50 pattern-conditional epochs, batch size 256, learning rate 1e-5, and separate optimizer/scheduler
  warm-up for the pattern stage. Full rerank improves all tracked main metrics over the paper-supervised
  baseline.
