## Domino Dataset (HRM)

Este documento describe el dataset de dominó diseñado para entrenar y evaluar el modelo HRM en predicción de jugadas. Incluye el formato de datos, cómo se generan, validaciones y cómo utilizarlos con las utilidades del repositorio.

### Objetivo del dataset

Predecir una jugada óptima (ficha y posición) dada una situación de mesa (cadena de fichas), las fichas del jugador y los extremos disponibles.

### Resumen de la codificación

- **seq_len**: 72
- **vocab_size**: 8
  - 0 = PAD / vacío
  - 1..7 = valores reales 0..6 (codificados como valor+1)
- **ignore_label_id**: 0 (el dataloader lo remapea a -100 internamente para el loss)
- **num_puzzle_identifiers**: 1

### Estructura de `inputs` (longitud 72)

Layout y significado por posiciones (valores almacenados como `valor_real + 1`, 0=vacío):

- [0..55] Mesa: hasta 28 fichas, 2 posiciones por ficha (A,B)
  - Para la ficha j en mesa: `inputs[2*j] = A+1`, `inputs[2*j+1] = B+1`, j=0..27
  - Si el hueco está vacío: `0, 0`
- [56..69] Mano del jugador: hasta 7 fichas, 2 posiciones por ficha (A,B)
  - Para la ficha h en mano: `inputs[56+2*h] = A+1`, `inputs[56+2*h+1] = B+1`, h=0..6
  - Si el hueco está vacío: `0, 0`
- [70] Extremo izquierdo: `left_end + 1` (0 si no hay)
- [71] Extremo derecho: `right_end + 1` (0 si no hay)

Notas:
- Toda ficha se codifica como un par contiguo (A,B).
- La mesa se representa en el orden actual (izquierda→derecha). Los extremos en [70..71] son la “verdad” del estado.

### Estructura de `labels` (longitud 72)

Solo se usan las 3 primeras posiciones; el resto se ignoran por `ignore_label_id=0`:

- [0] Ficha A seleccionada: `A+1` (0 si no hay jugada)
- [1] Ficha B seleccionada: `B+1` (0 si no hay jugada)
- [2] Posición de la jugada:
  - 1 = left
  - 2 = right
  - 3 = first (mesa vacía)
  - 0 = sin jugada
- [3..71] = 0 (padding ignorado por el loss)

### Archivos y estructura en disco

Ejemplo: `data/domino-opt/`

```
data/domino-opt/
├── train/
│   ├── dataset.json            # PuzzleDatasetMetadata
│   ├── all__inputs.npy         # [N, 72] int32
│   ├── all__labels.npy         # [N, 72] int32
│   ├── all__puzzle_identifiers.npy  # [num_puzzles] int32
│   ├── all__puzzle_indices.npy      # [num_puzzles+1] int32
│   └── all__group_indices.npy       # [num_groups+1] int32
├── test/
│   └── ... (igual formato)
└── identifiers.json            # Mapeo de identificadores (visualización)
```

`dataset.json` incluye (entre otros):

- `seq_len=72`
- `vocab_size=8`
- `pad_id=0`
- `ignore_label_id=0` (para que los 0 de [3..71] en `labels` se ignoren)

### Generación (builder)

Script: `dataset/build_domino_dataset.py`

Config (CLI):

- `--output-dir`: destino del dataset
- `--num-games`: número de escenarios a generar por split (train/test)
- `--subsample-size`: (opcional, solo train) submuestreo aleatorio
- `--num-aug`: (opcional, solo train) número de augmentations por ejemplo

Características:

- Genera exactamente `num_games` escenarios válidos por split.
- Augmentación opcional: suma modular (1..6) a valores de fichas (inputs y `labels[0..1]`) sin alterar el código de posición `labels[2]`.
- Semilla fijada en `preprocess_data()` para reproducibilidad.

### Validación y visualización

- Inspección estructural: `scripts/inspect_domino_dataset.py`
  - Muestra decodificación legible (mesa, mano, extremos, objetivo) y arrays crudos
  - Valida longitud, rango de vocabulario, coherencia de jugada (en mano y acorde a extremos)

```
python3 scripts/inspect_domino_dataset.py --data-path data/domino-opt --split train --num 5
```

- Visualización en consola (ASCII): `scripts/visualize_domino_play.py`
  - Dibuja la mesa (con extremos), la mano (marcando jugables `*` y objetivo `>>`)

```
python3 scripts/visualize_domino_play.py --data-path data/domino-opt --split train --index 0
```

### Uso en entrenamiento y evaluación

Entrenamiento (ejemplo):

```
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py \
  data_path=data/domino-opt \
  epochs=20000 eval_interval=2000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

Evaluación/volcado de predicciones (con checkpoint):

```
python3 evaluate.py checkpoint=checkpoints/.../step_XXXX
```

### Consideraciones y mejoras futuras

- Heurística de selección de jugada: actualmente simple (prioriza fichas “pesadas”). Puede mejorarse con reglas de bloqueo, dobles, control de extremos, etc.
- Ambigüedad: cuando hay varias jugadas igual de buenas, hoy se elige una; podría contemplarse múltiples etiquetas válidas.
- Oponente/desconocido: el estado no contempla información probabilística de fichas del oponente.
- Augmentación: se puede añadir inversión de orden de la cadena (cambio consistente de extremos) si se adapta la codificación/decodificación.


