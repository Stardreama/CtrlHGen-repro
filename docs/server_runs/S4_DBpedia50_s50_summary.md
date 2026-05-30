# S4 DBpedia50 db50_s50 Server Run

## Scale

- Dataset: DBpedia50
- Scale: db50_s50
- Arity: 32
- Train rows: 14300
- Valid rows: 1781
- Test rows: 1781

## Training

- Unconditional checkpoint: epoch 20
- Pattern checkpoint: epoch 40

## Baseline Full Test

- Count: 1781
- F1: 0.0403916561
- Smatch: 0.0957235621
- Jaccard: 0.0386863281
- Dice: 0.0403916561
- Overlap: 0.0442313699
- Validity: 0.1122964627
- Entity-number: 0.1521617069
- Relation-number: 0.1134194273

## Rerank Full Test

- Rerank k: 4
- Alpha: 0.5
- Count: 1781
- Candidate rows: 7124
- F1: 0.0779688482
- Smatch: 0.1081285443
- Jaccard: 0.0754610446
- Dice: 0.0779688482
- Overlap: 0.0846903268
- Validity: 0.1285794497
- Entity-number: 0.1538461538
- Relation-number: 0.1285794497

## Conclusion

S4 full rerank improves the main metrics over the full baseline, especially F1, Smatch, Jaccard/Dice/Overlap, Validity, and Relation-number.
