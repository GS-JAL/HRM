## Guía Completa de Entrenamiento e Inferencia (HRM)

Esta guía resume, paso a paso, cómo preparar datasets, entrenar e inferir con el Hierarchical Reasoning Model (HRM) para diferentes tareas (ARC, Sudoku, Maze, Dominó). Se basa en `README.md`, `pretrain.py`, `evaluate.py`, `config/` y el paper adjunto.

### 1) Preparación del entorno

1. CUDA y PyTorch con soporte GPU (ver README para versiones recomendadas).  
2. Dependencias Python:

```bash
pip install -r requirements.txt
```

3. (Opcional) FlashAttention según tu GPU (Hopper: FA3; Ampere o anterior: FA2).

4. Inicia sesión en W&B si quieres tracking:

```bash
wandb login
```

### 2) Construcción de datasets

Builders incluidos:
- ARC: `dataset/build_arc_dataset.py`
- Sudoku: `dataset/build_sudoku_dataset.py`
- Maze: `dataset/build_maze_dataset.py`
- Dominó: `dataset/build_domino_dataset.py`

Ejemplos:
```bash
# ARC-1 (oficial + ConceptARC)
python dataset/build_arc_dataset.py --output-dir data/arc-aug-1000

# Sudoku (1k, con augmentación)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Maze (1k)
python dataset/build_maze_dataset.py --output-dir data/maze-30x30-hard-1k

# Dominó (p. ej., 2000 escenarios)
python dataset/build_domino_dataset.py --output-dir data/domino-opt --num-games 2000 --num-aug 0
```

Requisitos de datasets (resumen):
- `inputs` y `labels` deben tener misma `seq_len`.
- `ignore_label_id` en `dataset.json` indica el token en labels que se ignora (suele ser 0).
- Vocabulario entero `[0..vocab_size-1]`.
- Índices `puzzle_indices`/`group_indices` consistentes (ver `dataset.md`).

Visualización / inspección:
```bash
# Visualizador HTML genérico de puzzles
open puzzle_visualizer.html  # y subir la carpeta del dataset

# Inspectores específicos
python scripts/inspect_sudoku_dataset.py  --data-path data/sudoku-extreme-1k-aug-1000 --split train --num 3
python scripts/inspect_domino_dataset.py  --data-path data/domino-opt --split train --num 5
python scripts/visualize_domino_play.py   --data-path data/domino-opt --split train --index 0
```

### 3) Configuración de entrenamiento

HRM usa Hydra para configuración. Plantilla base: `config/cfg_pretrain.yaml` y arquitectura `config/arch/hrm_v1.yaml`.

Parámetros relevantes (sobrescribibles vía CLI):
- `data_path`: ruta a tu dataset.
- `global_batch_size`: total por paso (se divide entre GPUs automáticamente).
- `epochs`: número lógico de “épocas” (ver cálculo de pasos más abajo).
- `eval_interval`: cada cuántas “épocas” evaluar.
- `checkpoint_every_eval`: guardar checkpoint tras cada evaluación.
- LR y WD: `lr`, `puzzle_emb_lr`, `weight_decay`, `puzzle_emb_weight_decay`.
- Scheduler: `lr_min_ratio`, `lr_warmup_steps`.
- Arquitectura (en `arch/`):
  - `hidden_size`, `num_heads`, `expansion`.
  - `H_cycles`, `L_cycles` (profundidad de razonamiento por forward).
  - `halt_max_steps`, `halt_exploration_prob` (ACT: pasos máximos y exploración de paro).
  - `pos_encodings` (rope|learned).
  - `puzzle_emb_ndim` (típicamente = `hidden_size`, puede ponerse 0 para desactivar embeddings de puzzle).

Cómo se calculan los pasos totales de entrenamiento (`pretrain.py`):
```text
total_steps = epochs * total_groups * mean_puzzle_examples / global_batch_size
```
Donde `total_groups` y `mean_puzzle_examples` vienen del `dataset.json`.

### 4) Lanzar entrenamiento

Single GPU (o CPU con CUDA desactivado, no recomendado):
```bash
OMP_NUM_THREADS=8 python pretrain.py \
  data_path=data/arc-aug-1000 \
  epochs=100000 eval_interval=10000 \
  global_batch_size=384 lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=0.1 puzzle_emb_weight_decay=0.1
```

Multi-GPU (recomendado para grandes lotes):
```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  epochs=20000 eval_interval=2000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

Sugerencias por tarea (punto de partida):
- ARC (1k): usa configuración por defecto (`cfg_pretrain.yaml`) y 8 GPUs si puedes. Tiempo ~24h.
- Sudoku (1k): valores del README funcionan bien; overfitting tardío común → early stopping cuando te acerques a 100%.
- Maze (1k): similar a Sudoku.
- Dominó (2k–10k): empieza con `epochs=20000`, `eval_interval=2000`, `lr=1e-4`, `puzzle_emb_lr=1e-4`, WD=1.0, `global_batch_size=384–768`. Ajusta `halt_max_steps` (p.ej. 8–16) si la tarea requiere más pasos.

Monitoreo:
- Métricas clave en W&B: `train/accuracy`, `train/exact_accuracy`, `train/lm_loss`, `train/steps`, `train/q_*`.
- En evaluación se reporta por set (típicamente `all`).

Checkpoints y config:
- Se guardan en `checkpoints/<project>/<run>/` con `all_config.yaml` y código fuente de los módulos de modelo/pérdida usados.

### 5) Evaluación e inferencia

Evaluación por lotes con checkpoint (usa el mismo `data_path` del entrenamiento):
```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=checkpoints/.../step_XXXX
```
Esto:
- Carga `all_config.yaml` desde la carpeta del checkpoint.
- Recrea el dataloader y el modelo con los mismos hiperparámetros.
- Recorre el split `test` en modo ACT (máximo de pasos, sin exploración) y guarda, si está configurado, tensores de salida (p. ej. `logits`).

Inferencia ad-hoc (por ejemplo, Sudoku o Dominó):
- Si no quieres montar todo `evaluate.py`, puedes replicar la lógica mínima:
  1. Carga `all_config.yaml` del checkpoint para recrear el modelo con `create_model`.
  2. Construye un batch con `inputs` y `puzzle_identifiers` (y `labels` si vas a medir loss). Ten en cuenta el `seq_len` de tu dataset.
  3. Llama a `model.initial_carry(batch)` y luego itera `model(carry, batch)` hasta que `outputs['halted']` sea verdadero.
  4. Toma `outputs['logits']` y decodifica (`argmax` por posición relevante de `labels`).

Pseudocódigo minimalista:
```python
from pretrain import PretrainConfig, create_dataloader, init_train_state
import yaml, torch, os

ckpt_dir = "checkpoints/..."
with open(os.path.join(ckpt_dir, "all_config.yaml"), "r") as f:
    cfg = PretrainConfig(**yaml.safe_load(f))

train_loader, train_meta = create_dataloader(cfg, "train", rank=0, world_size=1, test_set_mode=False, epochs_per_iter=1, global_batch_size=cfg.global_batch_size)
eval_loader,  eval_meta  = create_dataloader(cfg, "test",  rank=0, world_size=1, test_set_mode=True,  epochs_per_iter=1, global_batch_size=cfg.global_batch_size)

state = init_train_state(cfg, train_meta, world_size=1)
state.model.load_state_dict(torch.load(os.path.join(ckpt_dir, "step_XXXX"), map_location="cuda"), assign=True)
state.model.eval()

for set_name, batch, _ in eval_loader:
    batch = {k: v.cuda() for k, v in batch.items()}
    carry = state.model.initial_carry(batch)
    while True:
        carry, outputs = state.model(carry, batch)
        if outputs["halted"].all():
            break
    logits = outputs["logits"]  # [B, seq_len, vocab]
    preds = torch.argmax(logits, dim=-1)
    # Decodificar según tu tarea (p.ej., Sudoku: 81 celdas; Dominó: usar [0..2] para jugada)
    break
```

Decodificación por tarea:
- Sudoku: `preds[b]` es una secuencia de 81; mapea tokens (1..10) a dígitos reales (0..9) restando 1.
- ARC: salida es una grilla 30×30 linealizada (PAD/EOS y colores).
- Dominó: relevante `preds[b][:3]` → (A+1, B+1, pos). El resto se ignora.

### 6) Consejos de ajuste fino

- `global_batch_size`: saturar GPU(s) sin OOM; aumentar si el gradiente es ruidoso.
- `H_cycles`, `L_cycles`: más ciclos → más capacidad de razonamiento en un forward. Útil para ARC difícil y tareas con transformaciones complejas.
- `halt_max_steps`: incrementarlo si notas que la política de paro se queda corta (el modelo “necesita más pasos”). En evaluación no hay exploración.
- LR y WD: si la pérdida no baja o oscila, reduce `lr` y/o sube `weight_decay`.
- `puzzle_emb_ndim`: 0 si no quieres embeddings específicos; por defecto = `hidden_size`.
- `pos_encodings`: `rope` suele ser estable; `learned` puede rendir mejor en algunos dominios (ejemplo del README con Sudoku-Hard).

### 7) Depuración y verificación

- Usa los inspectores/visualizadores para verificar inputs/labels y que `seq_len`/`vocab` coincidan con el dataset.
- Si ves `seq_len_mismatch` o tokens fuera de rango, revisa el builder.
- Chequea W&B para detectar overfitting: si `train/accuracy` → 100% pero `eval` no, reduce `epochs` o aplica early stopping.

### 8) Referencias

- README del repo para recetas rápidas y checkpoinst públicos.
- `config/` para valores por defecto (Hydra permite sobreescribir por CLI).
- `pretrain.py` y `evaluate.py` como referencia de la tubería oficial.
- Paper HRM (PDF en el repo) para fundamentos de arquitectura y diseño de entrenamiento con ACT.


