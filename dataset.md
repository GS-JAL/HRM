## Requisitos de Datasets y Builders en HRM

Este documento define los requisitos para crear datasets compatibles con HRM y guías para implementar nuevos builders.

### Contrato de datos (PuzzleDataset)

Cada dataset debe cumplir con el siguiente contrato por `split` (`train/`, `test/`) y por `set` (normalmente `all`):

Archivos requeridos:

- `dataset.json` (JSON) con la estructura `PuzzleDatasetMetadata`:
  - `pad_id: int` — token de padding (típicamente 0)
  - `ignore_label_id: Optional[int]` — token en `labels` a ignorar en el loss (p.ej. 0)
  - `blank_identifier_id: int` — id del identificador “en blanco” (normalmente 0)
  - `vocab_size: int` — tamaño del vocabulario entero (incluye PAD)
  - `seq_len: int` — longitud fija de secuencia de inputs y labels
  - `num_puzzle_identifiers: int` — número de identificadores de puzzle (≥1)
  - `total_groups: int` — número de grupos (para sampling por grupos)
  - `mean_puzzle_examples: float` — ejemplos por puzzle en promedio (para schedule de pasos)
  - `sets: List[str]` — lista de sets presentes (típicamente `["all"]`)

- Tensores `.npy` (dtype int32) para cada `set`:
  - `all__inputs.npy`: `[N, seq_len]`
  - `all__labels.npy`: `[N, seq_len]`
  - `all__puzzle_identifiers.npy`: `[num_puzzles]`
  - `all__puzzle_indices.npy`: `[num_puzzles+1]` — offsets de ejemplos por puzzle
  - `all__group_indices.npy`: `[num_groups+1]` — offsets de puzzles por grupo

Reglas clave:

- `inputs` y `labels` deben tener la MISMA `seq_len`.
- Para posiciones de `labels` sin supervisión se debe usar `ignore_label_id` (p. ej. 0). El `PuzzleDataset` remapea internamente a `-100` para el loss.
- Valores de tokens deben estar en `[0, vocab_size-1]`.

### Batching y sampling (lo hace el dataloader)

- El `PuzzleDataset` realiza sampling por grupos para mantener coherencia de puzzles; requiere `puzzle_indices` y `group_indices` correctos.
- Padding de batch: si un batch local no se llena, el dataloader paddea con `pad_id` en `inputs` y `ignore_label_id` en `labels`, y usa `blank_identifier_id` en `puzzle_identifiers`.

### Pautas para nuevos builders

1) Definir codificación entera clara
   - Reservar `pad_id` (habitualmente 0). Mapear dominios al rango `[1..]` cuando sea posible (como Sudoku y Dominó).
   - Establecer `vocab_size` mínimo que cubra todos los símbolos (incluye `pad_id`).

2) Alinear `seq_len`
   - Determinar `seq_len` fijo para inputs y labels.
   - Si solo se supervisan algunas posiciones de `labels`, poner 0 en el resto y fijar `ignore_label_id=0` en `dataset.json`.

3) Estructurar índices para batching
   - Construir `puzzle_indices` y `group_indices` correctamente:
     - `puzzle_indices`: offsets de ejemplos por puzzle, comienza en 0 y termina en `num_examples`.
     - `group_indices`: offsets de puzzles por grupo.

4) Metadata fiel
   - `dataset.json` debe reflejar el dataset final: `seq_len`, `vocab_size`, `ignore_label_id`, `num_puzzle_identifiers`, `total_groups`, `mean_puzzle_examples`, `sets`.

5) Augmentación (opcional)
   - Asegurar que la augmentación conserve la semántica (no corromper campos de control como códigos de posición, separadores, etc.).
   - Evitar duplicados si es relevante (hashes o comprobaciones simples).

6) Reproducibilidad
   - Fijar semillas en la generación y guardar parámetros usados para poder replicar.

7) Validación e inspección
   - Proveer scripts de inspección que:
     - Decodifiquen legiblemente `inputs`/`labels`.
     - Verifiquen rango de tokens, longitudes, y reglas del dominio (ej. Sudoku: duplicados en filas/columnas; Dominó: ficha objetivo en mano y válida).
   - (Opcional) Visualización amigable (ASCII/HTML) para depuración.

### Ejemplos de referencia

- ARC: `dataset/build_arc_dataset.py`
  - `seq_len=900` (30×30), `vocab_size=12` (PAD+EOS+colores 0..9)
  - `ignore_label_id=0`; augmentación por simetrías dihedrales y traslación.

- Sudoku: `dataset/build_sudoku_dataset.py`
  - `seq_len=81`, `vocab_size=11` (PAD + "0".."9" mapeados a 1..10)
  - `ignore_label_id=0`; barajado de grilla/dígitos como augmentación.

- Domino: `dataset/build_domino_dataset.py`
  - `seq_len=72`, `vocab_size=8` (PAD + valores 0..6 como 1..7)
  - `ignore_label_id=0`; labels con jugada en [0..2] y resto ignorado.

### Checklist para validar un nuevo dataset

- [ ] `inputs.shape[1] == labels.shape[1] == seq_len`
- [ ] Tokens en `[0, vocab_size-1]`
- [ ] `dataset.json` consistente (seq_len, vocab_size, ignore_label_id, sets)
- [ ] `puzzle_indices` y `group_indices` consistentes (últimos = totales)
- [ ] Script de inspección pasa validaciones básicas del dominio
- [ ] (Opcional) Visualización comprensible para humanos


