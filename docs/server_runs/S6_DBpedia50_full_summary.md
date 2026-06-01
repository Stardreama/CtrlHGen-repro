# S6 DBpedia50 full Server Run

  ## Scale

  - Dataset: DBpedia50
  - Scale: full
  - Arity: 32
  - Train rows: 715104
  - Valid rows: 89388
  - Test rows: 89388

  ## Training

  - Unconditional checkpoint: epoch 20
  - Pattern checkpoint: epoch 40

  ## Baseline Full Test

  - Count: 89388
  - F1: 0.1089453899
  - Smatch: 0.1185973330
  - Jaccard: 0.1062025318
  - Dice: 0.1089453899
  - Overlap: 0.1155527102
  - Validity: 0.1381952835
  - Entity-number: 0.1431288316
  - Relation-number: 0.1381952835

  ## Rerank Tests

  ### Rerank 0.1

  - Count: 8938
  - Candidate rows: 35752
  - F1: 0.1263858358
  - Smatch: 0.1252822865
  - Jaccard: 0.1240001931
  - Dice: 0.1263858358
  - Overlap: 0.1316288482
  - Validity: 0.1430968897
  - Entity-number: 0.1448869993
  - Relation-number: 0.1430968897

  ### Rerank 0.25

  - Count: 22347
  - Candidate rows: 89388
  - F1: 0.1270976005
  - Smatch: 0.1265364489
  - Jaccard: 0.1244377921
  - Dice: 0.1270976005
  - Overlap: 0.1331426546
  - Validity: 0.1439566832
  - Entity-number: 0.1466863561
  - Relation-number: 0.1439566832

  ### Rerank Full

  - Count: 89388
  - Candidate rows: 357552
  - F1: 0.1277623125
  - Smatch: 0.1274095347
  - Jaccard: 0.1251070049
  - Dice: 0.1277623125
  - Overlap: 0.1338833316
  - Validity: 0.1455452633
  - Entity-number: 0.1481742516
  - Relation-number: 0.1455452633

  ## Conclusion

  S6 full rerank improves the full baseline on all tracked main metrics.
