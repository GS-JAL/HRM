# GridReasoningOps Lite: Dataset y Entrenamiento

Este documento describe cómo generar datasets de GridReasoningOps Lite (multitarea) con currículum y tamaños de grilla variables, y cómo entrenar/evaluar el modelo HRM con ellos.

## Tareas incluidas

- puzzle_identifier 1 — REACHABLE_MASK
  - Dado `S` y obstáculos `#`, marcar celdas alcanzables (4-vecinos) con una máscara 0/1 (FALSE=20, TRUE=21).
- puzzle_identifier 2 — SHORTEST_PATH_MASK
  - Dado `S` y `T`, marcar el camino mínimo S→T (4-vecinos) como máscara 0/1.
- puzzle_identifier 3 — FLOODFILL_SIZE
  - Dado `S`, devolver el tamaño de la componente conexa alcanzable desde `S` (codificado en 3 dígitos en los primeros tokens del payload; dígitos=30..39).

## Formato de ejemplo

- Vocabulario inputs: `PAD=0`, `SEP=1`, `'.'=10`, `'#'=11`, `'S'=12`, `'T'=13`
- Vocabulario labels: `IGNORE=0`, `FALSE=20`, `TRUE=21`, `DIGIT_0..9=30..39`
- Layout por ejemplo:
  - Header `[SEP, OP_CODE, H, W, 0, 0, 0, SEP]` (8 tokens)
  - Payload: grilla `H×W` aplanada (row-major)
  - Padding: cada ejemplo se paddea a `seq_len` fijo por split

## Currículum y tamaños variables

- El split `train/` contiene 3 sets: `easy`, `medium`, `hard`.
- En cada set, por ejemplo se muestrea tamaño y densidad de obstáculos:
  - easy: `H∈[easy_H_min, easy_H_max]`, `W∈[easy_W_min, easy_W_max]`, `p_wall∈[easy_obstacle_prob_min, easy_obstacle_prob_max]`
  - medium: rango propio
  - hard: rango propio
- El split `test/` usa el set único `all` con rangos por defecto de `medium` (configurable si se desea).

`seq_len` por split:
- train: `HEADER_LEN + max(hard_H_max×hard_W_max, medium..., easy...)`
- test: `HEADER_LEN + (medium_H_max×medium_W_max)` por defecto

## Generación de dataset

Pequeño demo (rápido):
```bash
python3 dataset/build_grid_ops_lite.py build \
  --output-dir data/grid-ops-lite \
  --train-easy-size 4000 --train-medium-size 4000 --train-hard-size 4000 \
  --test-size 1000 --aug true \
  --easy-H-min 4 --easy-H-max 8 --easy-W-min 4 --easy-W-max 8 \
  --medium-H-min 8 --medium-H-max 16 --medium-W-min 8 --medium-W-max 16 \
  --hard-H-min 16 --hard-H-max 30 --hard-W-min 16 --hard-W-max 30
```

Salida del builder:
- `data/grid-ops-lite/train/{easy,medium,hard}__{inputs,labels,puzzle_identifiers,puzzle_indices,group_indices}.npy`
- `data/grid-ops-lite/train/dataset.json` (metadata del split)
- `data/grid-ops-lite/test/all__{...}.npy` y `data/grid-ops-lite/test/dataset.json`
- `data/grid-ops-lite/identifiers.json`: `["<blank>", "REACHABLE_MASK", "SHORTEST_PATH_MASK", "FLOODFILL_SIZE"]`

## Visualización rápida

```bash
python3 scripts/visualize_grid_ops_lite.py --dataset-dir data/grid-ops-lite --split train --set easy --num 5
```

Renderiza:
- G: grilla con `.` `#` `S` `T`
- M: máscara de salida (si aplica); para floodfill muestra `floodfill_size=<n>`

## Entrenamiento

Ejemplo:
```bash
# Config base
OMP_NUM_THREADS=1 torchrun --nproc-per-node 1 pretrain.py --config-name cfg_grid_ops_lite_base

# Config large batch
OMP_NUM_THREADS=1 torchrun --nproc-per-node 1 pretrain.py --config-name cfg_grid_ops_lite_large_batch

# Config deeper (más pasos de razonamiento)
OMP_NUM_THREADS=1 torchrun --nproc-per-node 1 pretrain.py --config-name cfg_grid_ops_lite_deeper
```
Recomendaciones:
- `arch.halt_max_steps`: entre 8 y 16 suele bastar; súbelo si ves que `q_halt_logits` para demasiado pronto.
- `arch.puzzle_emb_ndim`: por defecto = `hidden_size`; puedes poner 0 para desactivar condicionamiento explícito (no recomendado en multitarea).
- Batch grande para mezclar bien las 3 tareas.

Currículo implícito:
- El dataloader recorre sets en orden `easy → medium → hard` (según `metadata.sets`). Con batches grandes se aprende en paralelo todas las tareas.

## Evaluación

Con un checkpoint entrenado:
```bash
OMP_NUM_THREADS=1 torchrun --nproc-per-node 1 evaluate.py checkpoint=<RUTA/AL/CHECKPOINT>
```
Métricas:
- `eval/exact_accuracy` en W&B (para máscaras y escalares discretizados)
- Para inspeccionar salidas: usa `scripts/visualize_grid_ops_lite.py` sobre `test/all`

## Notas técnicas

- Augmentación dihedral (8 simetrías) sólo en payloads y máscaras; los escalares (floodfill) no se transforman.
- Los ejemplos quedan paddeados a `seq_len` del split; `PuzzleDataset` convierte `ignore_label_id=0` a `-100` para el loss.
- Vocabulario mínimo garantizado por: `max(SEP, TOK_T, TOK_TRUE, DIGIT_9) + 1`.

## Troubleshooting

- ValueError al apilar arrays: asegúrate de que el builder corre tras los cambios (el script ya paddea a `seq_len` por split).
- Sin camino S→T: el builder reintenta y, si falla, genera un corredor trivial para garantizar etiquetas válidas.
- Secuencia demasiado larga por rangos amplios: reduce `hard_H_max×hard_W_max` o separa splits con `output_dir` distintos.
