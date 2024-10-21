# Tabulated Results

## Base Model Results over Varying Dataset Sizes

Our initial set of experiments was run on the base version of all our models on the AG News dataset over varying percentages of the dataset size.

### Runtime (Seconds)

|                          | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
| ---                      | --:  | --:  | --:  | --:  | --:  | --:  | --:  | --:  | --:  | --:  | --:  | --:  | --:  | --:  |
| Tf-Idf SVM               | 2    | 6    | 18   | 20   | 22   | 27   | 30   | 32   | 40   | 45   | 52   | 58   | 61   | 70   |
| NCD (LZ4)                | 49   | 190  | 407  | 735  | 1133 | 1612 | 2254 | 2898 |      |      |      |      |      |      |
| AMDL (LZ4)               | 54   | 207  | 455  | 806  | 1264 | 1823 | 2478 | 3240 |      |      |      |      |      |      |
| AMDL Dict (LZ4)          | 2    | 8    | 17   | 31   | 48   | 71   | 97   | 128  | 201  | 285  | 390  | 509  | 651  | 797  |
| Zest                     | 8    | 18   | 23   | 32   | 44   | 50   | 59   | 69   | 88   | 114  | 137  | 171  | 183  | 216  |
| Concentrated Dict (zstd) | 19   | 17   | 27   | 36   | 44   | 52   | 60   | 70   | 90   | 109  | 124  | 142  | 157  | 174  |

### Accuracy

|                          | 0.05   | 0.20   | 0.40   | 0.60   | 0.80   | 1.00   |
| ---                      | ---    | ---    | ---    | ---    | ---    | ---    |
| Tf-Idf SVM               | 0.8865 | 0.9095 | 0.9157 | 0.9232 | 0.9268 | 0.9273 |
| NCD (LZ4)                | 0.8338 | 0.8664 | 0.8838 | 0.8953 | 0.8980 | 0.9007 |
| AMDL (LZ4)               | 0.8628 | 0.8864 | 0.8960 | 0.9008 | 0.9034 | 0.9034 |
| AMDL Dict (LZ4)          | 0.8654 | 0.8874 | 0.8963 | 0.9013 | 0.9029 | 0.9038 |
| Zest                     | 0.8127 | 0.8374 | 0.8591 | 0.8593 | 0.8622 | 0.8598 |
| Concentrated Dict (zstd) | 0.8615 | 0.8664 | 0.8799 | 0.8802 | 0.8835 | 0.8893 |

## Accuracies over all Datasets

Accuracies of all models (all compressors), over the full four datasets. NCD (literature) was taken from *[Bad numbers in the "gzip beats BERT" paper?](https://kenschutte.com/gzip-knn-paper2/)*

|                  | AG News | Twenty News | GISAID | Malware |
| ---              | --:     | --:         | --:    | --:     |
| NCD (literature) | 0.8760  | 0.6070      |        |         |
| Tf-Idf SVM       | 0.9273  | 0.8633      | 1.0000 | 0.9831  |
| NCD (zlib)       | 0.8890  | 0.6139      | 0.9862 | 0.4068  |
| NCD (lz4)        | 0.9007  | 0.6134      | 0.9862 | 0.5170  |
| NCD (zstd)       | 0.8795  | 0.6013      | 0.9585 | 0.9068  |
| AMDL (zlib)      | 0.8959  | 0.7628      | 1.0000 | 0.8644  |
| AMDL (lz4)       | 0.9034  | 0.7678      | 1.0000 | 0.9322  |
| AMDL (zstd)      | 0.8986  | 0.7884      | 1.0000 | 0.9746  |
| Zest             | 0.8891  | 0.7553      | 0.2388 | 0.5678  |
