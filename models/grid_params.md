
## LINE

```py
{
    'n_loops':       [5],
    'data_agg_type': ['LINE-MMS-Mean', 'LINE-MMS-Sum', 'LINE-R-Mean', 'LINE-R-Sum'],
    'emb_dim':       [128, 256],
    'n_epochs':      [100],
    'batch_size':    [10000],
    'neg':           [5, 10],
    'lr':            [0.005, 0.01, 0.025]
}
```

## GTMAE

```py
{
    "hid":              [96, 128],
    "out":              [64, 128],
    "heads":            [2],
    "dropout":          [0.1, 0.2],
    "lr":               [1e-3, 5e-4],
    "weight_decay":     [1e-5, 1e-4],
    "edge_drop_prob":   [0.0, 0.2],
    "edge_loss_type":   ["huber"],
    "edge_huber_delta": [1.0],
    "node_loss_type":   ["huber"],
    "node_huber_delta": [1.0],
    "lambda_node":      [0.3, 0.5, 1.0],
    "node_mask_rate":   [0.0, 0.2],
    "add_ranking":      [False, True],
    "lambda_rank":      [0.3],
    "margin":           [0.1],
    "monitor":          ["val_edge_rmse"],
    "patience":         [30],
    "min_delta":        [0.0],
    "val_ratio":        [0.2],
    "test_ratio":       [0.2],
    "seed":             [33],
    "pair_mode":        ["cosine_l2_absdiff"],
    "use_pair_feats":   [True],
    "epochs":           [250]
}
```

## GTMVAE

```py
{
    "hid":              [96, 128],
    "out":              [64, 128],
    "heads":            [2],
    "dropout":          [0.1, 0.2],
    "lr":               [1e-3, 5e-4],
    "weight_decay":     [1e-5, 1e-4],
    "edge_drop_prob":   [0.0, 0.2],
    "edge_loss_type":   ["huber"],
    "edge_huber_delta": [1.0],
    "node_loss_type":   ["huber"],
    "node_huber_delta": [1.0],
    "lambda_node":      [0.3, 0.5, 1.0],
    "node_mask_rate":   [0.0, 0.2],
    "add_ranking":      [False, True],
    "lambda_rank":      [0.3],
    "margin":           [0.1],
    "monitor":          ["val_edge_rmse"],
    "patience":         [30],
    "min_delta":        [0.0],
    "val_ratio":        [0.2],
    "test_ratio":       [0.2],
    "seed":             [33],
    "pair_mode":        ["cosine_l2_absdiff"],
    "use_pair_feats":   [True],
    "epochs":           [250],
    "beta_kl":          [1e-3, 5e-4],
    "kl_warmup":        [10, 25]
}
```

## E2A-SAGE-MAE

```py
{
    "hid":              [128, 256],
    "out":              [128, 256],
    "dropout":          [0.1, 0.2],
    "use_batchnorm":    [True],
    "l2_norm_layers":   [True, False],
    "lr":               [1e-3, 1.3e-3, 6.5e-4],
    "weight_decay":     [1e-5, 1e-4],
    "edge_drop_prob":   [0.0, 0.2],
    "edge_loss_type":   ["huber"],
    "edge_huber_delta": [1.0],
    "node_loss_type":   ["huber"],
    "node_huber_delta": [1.0],
    "lambda_node":      [0.3, 0.5, 1.0],
    "node_mask_rate":   [0.0, 0.2],
    "add_ranking":      [False, True],
    "lambda_rank":      [0.3],
    "margin":           [0.1],
    "monitor":          ["val_edge_rmse"],
    "patience":         [30],
    "min_delta":        [0.0],
    "val_ratio":        [0.2],
    "test_ratio":       [0.2],
    "seed":             [33],
    "pair_mode":        ["cosine_l2_absdiff"],
    "use_pair_feats":   [True],
    "epochs":           [250]
}
```

## TGT

```py
{

}
```

## HGT-TE

```py
{

}
```

---

> Además de los hiperparámetros indicados en el presente documento, en cada una de las combinaciones se probará a incluir (`add_idea_emb=True`) y excluir (`add_idea_emb=False`) la variable de _embedding_ de texto de los anuncios de viviendas del municipio. Esto se deba a su gran dimensionalidad y como ésta puede abarcar gran parte del modelo en algunas casos.

> La parametría utilizada en los métodos de validación local (para cada modelo) y global (en la comparativa de _embeddinhgs_ finales) se puede consultar en cada uno de los ficheros de entrenamiento y validación dentro de `src/models/` y `src/validation`, respectivamente.
