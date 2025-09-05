
## LINE

### Excluyendo la ciudad de Madrid

| Parámetro        | Valor      |
| ---------------- | ---------- |
| `add_idea_emb`   | False      |
| `n_loops`        | 5          |
| `datat_agg_type` | LINE-R-Sum |
| `emb_dim`        | 256        |
| `batch_size`     | 10000      |
| `neg`            | 10         |
| `lr`             | 0.005      |

### Incluyendo la ciudad de Madrid

| Parámetro        | Valor        |
| ---------------- | ------------ |
| `add_idea_emb`   | False        |
| `n_loops`        | 5            |
| `datat_agg_type` | LINE-MMS-Sum |
| `emb_dim`        | 256          |
| `batch_size`     | 10000        |
| `neg`            | 5            |
| `lr`             | 0.005        |

---

## GTMAE

### Excluyendo la ciudad de Madrid

| Parámetro          | Valor               |
| ------------------ | ------------------- |
| `add_idea_emb`     | True                |
| `hid`              | 128                 |
| `out`              | 256                 |
| `heads`            | 2                   |
| `dropout`          | 0.2                 |
| `lr`               | 0.001               |
| `weight_decay`     | 0.0001              |
| `edge_drop_prob`   | 0.0                 |
| `edge_loss_type`   | huber               |
| `edge_huber_delta` | 1.0                 |
| `node_loss_type`   | huber               |
| `node_huber_delta` | 1.0                 |
| `lambda_node`      | 1.0                 |
| `node_mask_rate`   | 0.0                 |
| `add_ranking`      | False               |
| `lambda_rank`      | 0.3                 |
| `margin`           | 0.1                 |
| `monitor`          | val_edge_rmse       |
| `patience`         | 30                  |
| `min_delta`        | 0.0                 |
| `val_ratio`        | 0.2                 |
| `test_ratio`       | 0.2                 |
| `seed`             | 33                  |
| `pair_mode`        | cosine_l2_absdiff   |
| `use_pair_feats`   | True                |

### Incluyendo la ciudad de Madrid

| Parámetro          | Valor               |
| ------------------ | ------------------- |
| `add_idea_emb`     | True                |
| `hid`              | 128                 |
| `out`              | 256                 |
| `heads`            | 2                   |
| `dropout`          | 0.2                 |
| `lr`               | 0.001               |
| `weight_decay`     | 0.0001              |
| `edge_drop_prob`   | 0.2                 |
| `edge_loss_type`   | huber               |
| `edge_huber_delta` | 1.0                 |
| `node_loss_type`   | huber               |
| `node_huber_delta` | 1.0                 |
| `lambda_node`      | 1.0                 |
| `node_mask_rate`   | 0.2                 |
| `add_ranking`      | False               |
| `lambda_rank`      | 0.3                 |
| `margin`           | 0.1                 |
| `monitor`          | val_edge_rmse       |
| `patience`         | 30                  |
| `min_delta`        | 0.0                 |
| `val_ratio`        | 0.2                 |
| `test_ratio`       | 0.2                 |
| `seed`             | 33                  |
| `pair_mode`        | cosine_l2_absdiff   |
| `use_pair_feats`   | True                |

---

## GTMVAE

### Excluyendo la ciudad de Madrid

| Parámetro          | Valor               |
| ------------------ | ------------------- |
| `add_idea_emb`     | True                |
| `hid`              | 128                 |
| `out`              | 256                 |
| `heads`            | 2                   |
| `dropout`          | 0.2                 |
| `lr`               | 0.001               |
| `weight_decay`     | 0.00001             |
| `edge_drop_prob`   | 0.0                 |
| `edge_loss_type`   | huber               |
| `edge_huber_delta` | 1.0                 |
| `node_loss_type`   | huber               |
| `node_huber_delta` | 1.0                 |
| `lambda_node`      | 1.0                 |
| `node_mask_rate`   | 0.0                 |
| `add_ranking`      | False               |
| `lambda_rank`      | 0.3                 |
| `margin`           | 0.1                 |
| `monitor`          | val_edge_rmse       |
| `patience`         | 30                  |
| `min_delta`        | 0.0                 |
| `val_ratio`        | 0.2                 |
| `test_ratio`       | 0.2                 |
| `seed`             | 33                  |
| `pair_mode`        | cosine_l2_absdiff   |
| `use_pair_feats`   | True                |
| `beta_kl`          | 0.0005              |
| `kl_warmup`        | 25                  |

### Incluyendo la ciudad de Madrid

| Parámetro          | Valor               |
| ------------------ | ------------------- |
| `add_idea_emb`     | True                |
| `hid`              | 128                 |
| `out`              | 256                 |
| `heads`            | 2                   |
| `dropout`          | 0.2                 |
| `lr`               | 0.001               |
| `weight_decay`     | 0.00001             |
| `edge_drop_prob`   | 0.2                 |
| `edge_loss_type`   | huber               |
| `edge_huber_delta` | 1.0                 |
| `node_loss_type`   | huber               |
| `node_huber_delta` | 1.0                 |
| `lambda_node`      | 1.0                 |
| `node_mask_rate`   | 0.2                 |
| `add_ranking`      | False               |
| `lambda_rank`      | 0.3                 |
| `margin`           | 0.1                 |
| `monitor`          | val_edge_rmse       |
| `patience`         | 30                  |
| `min_delta`        | 0.0                 |
| `val_ratio`        | 0.2                 |
| `test_ratio`       | 0.2                 |
| `seed`             | 33                  |
| `pair_mode`        | cosine_l2_absdiff   |
| `use_pair_feats`   | True                |
| `beta_kl`          | 0.001               |
| `kl_warmup`        | 10                  |

---

## E2A-SAGE-MAE

### Excluyendo la ciudad de Madrid

| Parámetro          | Valor               |
| ------------------ | ------------------- |
| `add_idea_emb`     | False               |
| `hid`              | 256                 |
| `out`              | 256                 |
| `use_batchnorm`    | True                |
| `l2_norm_layers`   | False               |
| `dropout`          | 0.2                 |
| `lr`               | 0.0013              |
| `weight_decay`     | 0.0001              |
| `edge_drop_prob`   | 0.2                 |
| `edge_loss_type`   | huber               |
| `edge_huber_delta` | 1.0                 |
| `node_loss_type`   | huber               |
| `node_huber_delta` | 1.0                 |
| `lambda_node`      | 1.0                 |
| `node_mask_rate`   | 0.0                 |
| `add_ranking`      | False               |
| `lambda_rank`      | 0.3                 |
| `margin`           | 0.1                 |
| `monitor`          | val_edge_rmse       |
| `patience`         | 30                  |
| `min_delta`        | 0.0                 |
| `val_ratio`        | 0.2                 |
| `test_ratio`       | 0.2                 |
| `seed`             | 33                  |
| `pair_mode`        | cosine_l2_absdiff   |
| `use_pair_feats`   | True                |

### Incluyendo la ciudad de Madrid

| Parámetro          | Valor               |
| ------------------ | ------------------- |
| `add_idea_emb`     | False               |
| `hid`              | 256                 |
| `out`              | 256                 |
| `use_batchnorm`    | True                |
| `l2_norm_layers`   | False               |
| `dropout`          | 0.2                 |
| `lr`               | 0.0013              |
| `weight_decay`     | 0.0001              |
| `edge_drop_prob`   | 0.0                 |
| `edge_loss_type`   | huber               |
| `edge_huber_delta` | 1.0                 |
| `node_loss_type`   | huber               |
| `node_huber_delta` | 1.0                 |
| `lambda_node`      | 1.0                 |
| `node_mask_rate`   | 0.0                 |
| `add_ranking`      | False               |
| `lambda_rank`      | 0.3                 |
| `margin`           | 0.1                 |
| `monitor`          | val_edge_rmse       |
| `patience`         | 30                  |
| `min_delta`        | 0.0                 |
| `val_ratio`        | 0.2                 |
| `test_ratio`       | 0.2                 |
| `seed`             | 33                  |
| `pair_mode`        | cosine_l2_absdiff   |
| `use_pair_feats`   | True                |

---

## TGT

### Excluyendo la ciudad de Madrid

| Parámetro          | Valor  |
| ------------------ | ------ |
| `add_idea_emb`     | False  |
| `hidden`           | 256    |
| `heads`            | 2      |
| `tf_layers`        | 1      |
| `tf_ff`            | 256    |
| `dropout`          | 0.1    |
| `time_enc_dim`     | 32     |
| `decay`            | 0.5    |


### Incluyendo la ciudad de Madrid

| Parámetro          | Valor  |
| ------------------ | ------ |
| `add_idea_emb`     | False  |
| `hidden`           | 256    |
| `heads`            | 4      |
| `tf_layers`        | 2      |
| `tf_ff`            | 256    |
| `dropout`          | 0.1    |
| `time_enc_dim`     | 16     |
| `decay`            | 0.3    |

---

## HGT-TE

### Excluyendo la ciudad de Madrid

| Parámetro          | Valor               |
| ------------------ | ------------------- |
| `add_idea_emb`     | False               |
| `spatial_hidden`   | 128                 |
| `spatial_out`      | 256                 |
| `heads`            | 4                   |
| `dropout`          | 0.3                 |
| `temporal_layers`  | 1                   |
| `temporal_heads`   | 4                   |
| `temporal_ff`      | 256                 |
| `temporal_pe_dim`  | None                |
| `lambda_focus`     | 0.25                |

### Incluyendo la ciudad de Madrid

| Parámetro          | Valor               |
| ------------------ | ------------------- |
| `add_idea_emb`     | False               |
| `spatial_hidden`   | 128                 |
| `spatial_out`      | 256                 |
| `heads`            | 2                   |
| `dropout`          | 0.3                 |
| `temporal_layers`  | 1                   |
| `temporal_heads`   | 4                   |
| `temporal_ff`      | 256                 |
| `temporal_pe_dim`  | None                |
| `lambda_focus`     | 0.1                 |

---