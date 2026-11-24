# Data

The dataset is **not** stored in this repository. It is published on IEEE DataPort:

> M. T. Rahman and M. F. Hossain, "Network Performance Monitoring Dataset under Benign and DDoS Attack Traffic," IEEE DataPort, 2026. doi: [10.21227/pg53-f517](https://dx.doi.org/10.21227/pg53-f517)

Download the dataset package and copy **`merged_dataset.csv`** into this folder:

```
data/merged_dataset.csv
```

## Contents of `merged_dataset.csv`

| Item | Value |
|---|---|
| Sessions | 31 (16 benign, 15 attack) |
| Rows | 1,860 one-minute PRTG readings |
| Attack traffic | LOIC UDP flood (all 15 attack sessions) |
| Benign traffic | Legitimate TCP (9), UDP (3) and HTTP (4) sessions |
| Session id column | `source_file` |
| Label column | `session_label` (`benign` / `attack`) |

The 15 model input features are defined in `src/ddos_downtime/data.py` (`FEATURES`).
