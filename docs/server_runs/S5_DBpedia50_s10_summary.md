# S5 DBpedia50 db50_s10 Server Run

## Scale

- Dataset: DBpedia50
- Scale: db50_s10
- Arity: 32
- Train rows: 71500
- Valid rows: 8931
- Test rows: 8931

## Training

- Unconditional checkpoint: epoch 20
- Pattern checkpoint: epoch 40

## Baseline Full Test

- Count: 8931
- F1: 0.1320641211
- Smatch: 0.1319606923
- Jaccard: 0.1281254468
- Dice: 0.1320641211
- Overlap: 0.1451628937
- Validity: 0.1343634531
- Entity-number: 0.1458963162
- Relation-number: 0.1455604076

## Rerank Tests

### Rerank 0.1

- Count: 893
- Candidate rows: 3572
- F1: 0.1842549285
- Smatch: 0.1561141670
- Jaccard: 0.1794280847
- Dice: 0.1842549285
- Overlap: 0.1969641366
- Validity: 0.1590145577
- Entity-number: 0.1634938410
- Relation-number: 0.1735722284

### Rerank 0.25

- Count: 2232
- Candidate rows: 8928
- F1: 0.1784536985
- Smatch: 0.1523186193
- Jaccard: 0.1734682645
- Dice: 0.1784536985
- Overlap: 0.1960368834
- Validity: 0.1482974910
- Entity-number: 0.1572580645
- Relation-number: 0.1639784946

### Rerank Full

- Count: 8931
- Candidate rows: 35724
- F1: 0.1671785276
- Smatch: 0.1447834819
- Jaccard: 0.1624344125
- Dice: 0.1671785276
- Overlap: 0.1818890143
- Validity: 0.1426491994
- Entity-number: 0.1498152503
- Relation-number: 0.1574291793

## Conclusion

S5 full rerank improves the full baseline on all tracked main metrics.
