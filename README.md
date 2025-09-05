
</div>
  <img src="./img/Logo_VIU.png" alt="Logo VIU" style="max-width:100%; height:auto;">
</div>

# Representaciones inmobiliarias basadas en _graph-embeddings_ para el análisis de similitud entre municipios
#### _Ignacio Esplugues Conca_

## Descripción

Este repositorio contiene el código desarrollado para el Trabajo de Fin de Máster sobre representación de municipios de la Comunidad de Madrid mediante **_graph embeddings_**. El objetivo del proyecto es modelar las relaciones socioeconómicas, demográficas, geográficas e inmobiliarias entre municipios y generar un **espacio vectorial latente** que permita analizar patrones urbanos como la presión inmobiliaria, la movilidad interna o la gentrificación.

El código implementa un pipeline completo que incluye:

* **Extracción, depuración y transformación de datos** a partir de fuentes oficiales (INE, IECM, CNIG, CRTM e Idealista).
* **Construcción de grafos urbanos** con atributos nodales e internodales, estáticos y dinámicos.
* **Generación de embeddings** mediante distintas arquitecturas: métodos clásicos (LINE) y modelos basados en *Graph Neural Networks* y *Transformers* (GTMAE, GTMVAE, E2A-SAGE, TGT y HGT-TE).
* **Validación de embeddings** mediante tareas de predicción supervisada (regresión y clasificación) con regresores lineales y no lineales, utilizando métricas como RMSE, MAE, ROC-AUC y F1-score.

El repositorio posee un apartado de *notebooks* organizados para facilitar la **reproducibilidad** del TFM, incluyendo los *steps* de preprocesado, EDA, entrenamiento y validación de resultados. Su estructura permite escalar la metodología a otras unidades territoriales, desde barrios hasta países, con aplicaciones en **planificación urbana, segmentación de mercados y detección de desigualdades territoriales**.


## Estructura

```sh
.
├── img                           # Images used in Markdown files
├── data                          # Data description adn sources
│   └── sources.md
├── models                        # Grid and best model params
│   ├── grid_params.md
│   └── best_model.md
├── notebooks                     # Notebooks executing train and validation code
│   ├── models.ipynb
│   └── validation.ipynb
├── src
│   ├── prepro                    # Preprocessing main functions
│   │   └── main_functions.py
│   ├── eda                       # EDA main functions
│   │   └── main_functions.py
│   ├── models                    # Models train pipelines
│   │   ├── utils.py
│   │   ├── LINE.py
│   │   ├── GTMAE.py
│   │   ├── GTMVAE.py
│   │   ├── E2A_SAGE_MAE.py
│   │   ├── TGT.py
│   │   └── HGT_TE.py
│   └── validation
│       ├── local_validation      # Models local validation pipelines
│       │   ├── LINE.py
│       │   ├── GTMAE.py
│       │   ├── GTMVAE.py
│       │   ├── E2A_SAGE_MAE.py
│       │   ├── TGT.py
│       │   └── HGT_TE.py
│       └── global_validation.py  # Models global validation pipeline
├── README.md                     # Description and structure of the repository
└── requirements.txt              # Libreries and versions used in the project
```
