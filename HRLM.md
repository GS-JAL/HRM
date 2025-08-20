# Hierarchical Reasoning Language Model (HRLM)

## Documentación para Adaptación de HRM a Procesamiento de Lenguaje Natural

---

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Análisis de la Arquitectura HRM Original](#análisis-de-la-arquitectura-hrm-original)
3. [Adaptaciones Necesarias para Texto](#adaptaciones-necesarias-para-texto)
4. [Estrategias de Tokenización](#estrategias-de-tokenización)
5. [Modificaciones Arquitecturales](#modificaciones-arquitecturales)
6. [Formato de Datos y Dataset](#formato-de-datos-y-dataset)
7. [Sistema de Entrenamiento](#sistema-de-entrenamiento)
8. [Decodificación y Generación](#decodificación-y-generación)
9. [Implementación Práctica](#implementación-práctica)
10. [Casos de Uso y Aplicaciones](#casos-de-uso-y-aplicaciones)
11. [Desafíos y Limitaciones](#desafíos-y-limitaciones)
12. [Guía de Implementación](#guía-de-implementación)
13. [Configuración Recomendada](#configuración-recomendada)
14. [Roadmap de Desarrollo](#roadmap-de-desarrollo)

---

## 🎯 Introducción

El **Hierarchical Reasoning Language Model (HRLM)** es una adaptación del modelo HRM (Hierarchical Reasoning Model) para el procesamiento de lenguaje natural. Esta adaptación mantiene las fortalezas únicas de HRM mientras extiende su aplicabilidad a tareas de texto que requieren razonamiento estructurado.

### Motivación

HRM ha demostrado capacidades excepcionales en:
- ✅ **Eficiencia de datos**: Aprendizaje con solo 1000 ejemplos
- ✅ **Razonamiento jerárquico**: Procesamiento en múltiples niveles de abstracción
- ✅ **Computación adaptativa**: Tiempo de procesamiento variable según complejidad
- ✅ **Estabilidad de entrenamiento**: Arquitectura recurrente estable

HRLM busca trasladar estas ventajas al dominio del procesamiento de lenguaje natural.

---

## 🏗️ Análisis de la Arquitectura HRM Original

### Componentes Principales

#### 1. **Arquitectura Jerárquica Dual**
```
┌─────────────────┐    ┌─────────────────┐
│   H-Level       │◄──►│   L-Level       │
│ (Planificación  │    │ (Ejecución      │
│  Abstracta)     │    │  Detallada)     │
└─────────────────┘    └─────────────────┘
```

- **H-Level**: Módulo de alto nivel para planificación lenta y abstracta
- **L-Level**: Módulo de bajo nivel para computaciones rápidas y detalladas

#### 2. **Procesamiento Recurrente**
```python
# Ciclos de procesamiento
for h_cycle in range(H_cycles):
    for l_cycle in range(L_cycles):
        if not (last_h_cycle and last_l_cycle):
            z_L = L_level(z_L, z_H + input_embeddings)
    if not last_h_cycle:
        z_H = H_level(z_H, z_L)

# Solo el último paso tiene gradientes
z_L = L_level(z_L, z_H + input_embeddings)  # Con gradientes
z_H = H_level(z_H, z_L)                     # Con gradientes
```

#### 3. **Mecanismo ACT (Adaptive Computation Time)**
- **Q-heads**: Predicen cuándo parar el procesamiento
- **Exploración**: Probabilidad configurable durante entrenamiento
- **Pasos máximos**: Límite superior de iteraciones

#### 4. **Configuración Actual**
```yaml
# Configuración típica para Sudoku
vocab_size: 11        # PAD + dígitos 0-9
seq_len: 81          # Grilla 9x9 aplanada
hidden_size: 512     # Dimensión de embeddings
H_cycles: 2          # Ciclos de alto nivel
L_cycles: 2          # Ciclos de bajo nivel
halt_max_steps: 16   # Pasos máximos ACT
```

---

## 🔄 Adaptaciones Necesarias para Texto

### Diferencias Fundamentales

| Aspecto | HRM Original | HRLM (Texto) |
|---------|--------------|--------------|
| **Vocabulario** | 11-12 tokens | 1000-50000 tokens |
| **Secuencias** | 81 tokens (fijo) | 512-2048 tokens (variable) |
| **Dominio** | Problemas determinísticos | Lenguaje natural |
| **Salidas** | Solución única | Múltiples respuestas válidas |
| **Evaluación** | Exactitud objetiva | Métricas de calidad textual |

### Desafíos Principales

1. **Escalabilidad de Vocabulario**: Embeddings y heads de salida mucho más grandes
2. **Longitud Variable**: Secuencias de texto de longitud variable vs. grillas fijas
3. **Complejidad Semántica**: Significado contextual vs. valores numéricos
4. **Generación Coherente**: Mantener coherencia en secuencias largas

---

## 🔤 Estrategias de Tokenización

### Opción 1: Tokenización a Nivel de Carácter (Recomendada)

```python
class CharTokenizer:
    def __init__(self):
        # Conjunto básico de caracteres
        self.charset = "abcdefghijklmnopqrstuvwxyz" + \
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + \
                      "0123456789 .,!?;:-()[]{}'\"\n"
        
        # Tokens especiales
        self.PAD_TOKEN = 0
        self.EOS_TOKEN = 1
        self.UNK_TOKEN = 2
        self.SEP_TOKEN = 3  # Separador entrada/salida
        
        # Crear mapeos
        self.char_to_id = {char: idx + 4 for idx, char in enumerate(self.charset)}
        self.vocab_size = len(self.charset) + 4
```

**Ventajas:**
- ✅ Vocabulario pequeño (~100 tokens)
- ✅ Compatible con arquitectura HRM actual
- ✅ No requiere tokenizadores externos
- ✅ Maneja cualquier texto sin tokens OOV

**Desventajas:**
- ❌ Secuencias más largas
- ❌ Menos eficiente para palabras comunes

### Opción 2: Tokenización Híbrida

```python
class HybridTokenizer:
    def __init__(self):
        # Palabras más comunes (top 500-1000)
        self.common_words = ["the", "and", "is", "to", "in", ...]
        
        # Caracteres para palabras no comunes
        self.charset = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?"
        
        # Tokens especiales
        self.special_tokens = ["<PAD>", "<EOS>", "<UNK>", "<SEP>"]
        
        self.vocab_size = len(self.common_words) + len(self.charset) + len(self.special_tokens)
```

**Ventajas:**
- ✅ Eficiencia para palabras comunes
- ✅ Vocabulario manejable (~1000 tokens)
- ✅ Flexibilidad para palabras raras

### Opción 3: BPE/WordPiece (Avanzada)

```python
# Usar tokenizadores estándar (GPT, BERT)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size  # ~50K tokens
```

**Ventajas:**
- ✅ Más eficiente para texto largo
- ✅ Mejor representación semántica

**Desventajas:**
- ❌ Vocabulario muy grande
- ❌ Requiere modificaciones significativas en HRM

---

## 🏛️ Modificaciones Arquitecturales

### Configuración Adaptada para Texto

```python
class TextHRMConfig:
    # Configuración de texto
    vocab_size: int = 1000        # Vocabulario expandido
    seq_len: int = 1024          # Secuencias más largas
    max_input_length: int = 512   # Longitud máxima de entrada
    max_output_length: int = 512  # Longitud máxima de salida
    
    # HRM expandido para texto
    hidden_size: int = 768        # vs 512 original
    H_cycles: int = 4             # vs 2 original (más razonamiento)
    L_cycles: int = 4             # vs 2 original
    H_layers: int = 6             # vs 4 original (más capacidad)
    L_layers: int = 6             # vs 4 original
    num_heads: int = 12           # vs 8 original
    halt_max_steps: int = 64      # vs 16 original (más pasos)
    
    # Nuevos parámetros
    use_input_output_separation: bool = True
    input_output_separator_token: int = 3
```

### Modificaciones en Embeddings

```python
# Embeddings escalados para vocabulario mayor
self.embed_tokens = CastedEmbedding(
    num_embeddings=config.vocab_size,    # 1000+ vs 11
    embedding_dim=config.hidden_size,    # 768 vs 512
    init_std=1.0 / math.sqrt(config.hidden_size),
    cast_to=config.forward_dtype
)

# Head de salida escalado
self.lm_head = CastedLinear(
    in_features=config.hidden_size,      # 768 vs 512
    out_features=config.vocab_size,      # 1000+ vs 11
    bias=False
)
```

### Procesamiento de Secuencias Largas

```python
# Ajustes para secuencias más largas
if config.pos_encodings == "rope":
    self.rotary_emb = RotaryEmbedding(
        dim=config.hidden_size // config.num_heads,
        max_position_embeddings=config.seq_len,  # 1024 vs 81
        base=config.rope_theta
    )
```

---

## 📊 Formato de Datos y Dataset

### Estructura de Datos para Texto

```python
# Formato de secuencia: [INPUT] [SEP] [OUTPUT] [EOS]
# Ejemplo: "Translate: Hello" [SEP] "Hola" [EOS]

def build_text_dataset(input_texts, output_texts, tokenizer, max_seq_len=1024):
    results = {
        "inputs": [],      # Secuencia completa
        "labels": [],      # Solo salida (entrada enmascarada)
        "puzzle_identifiers": [],
        "puzzle_indices": [],
        "group_indices": []
    }
    
    for inp_text, out_text in zip(input_texts, output_texts):
        # Tokenizar
        inp_tokens = tokenizer.encode(inp_text)
        out_tokens = tokenizer.encode(out_text)
        
        # Crear secuencia completa
        full_sequence = inp_tokens + [SEP_TOKEN] + out_tokens + [EOS_TOKEN]
        
        # Pad a longitud fija
        padded_sequence = full_sequence + [PAD_TOKEN] * (max_seq_len - len(full_sequence))
        
        # Input: secuencia completa
        inputs = np.array(padded_sequence[:max_seq_len])
        
        # Labels: -100 para entrada, tokens reales para salida
        labels = np.full(max_seq_len, -100)  # -100 = ignore
        sep_pos = np.where(inputs == SEP_TOKEN)[0]
        if len(sep_pos) > 0:
            start_output = sep_pos[0] + 1
            output_length = min(len(out_tokens), max_seq_len - start_output)
            labels[start_output:start_output + output_length] = out_tokens[:output_length]
        
        results["inputs"].append(inputs)
        results["labels"].append(labels)
        results["puzzle_identifiers"].append(0)
```

### Metadata para Texto

```python
metadata = PuzzleDatasetMetadata(
    seq_len=max_seq_len,
    vocab_size=tokenizer.vocab_size,
    pad_id=PAD_TOKEN,
    ignore_label_id=-100,
    blank_identifier_id=0,
    num_puzzle_identifiers=1,
    total_groups=len(input_texts),
    mean_puzzle_examples=1,
    sets=["train", "test"]
)
```

---

## 🎯 Sistema de Entrenamiento

### Función de Pérdida Adaptada

```python
def text_aware_loss(logits, labels, ignore_index=-100):
    """
    Calcula pérdida solo en tokens de salida
    """
    # Flatten para cross entropy
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    
    # Cross entropy con ignore_index
    loss = F.cross_entropy(
        flat_logits, 
        flat_labels, 
        ignore_index=ignore_index, 
        reduction='none'
    )
    
    # Promediar por secuencia válida
    loss = loss.view(labels.shape)
    valid_mask = (labels != ignore_index)
    seq_losses = loss.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
    
    return seq_losses.mean()
```

### Configuración de Entrenamiento

```yaml
# Configuración adaptada para texto
data_path: "data/text-dataset"

# Hiperparámetros ajustados
global_batch_size: 32         # Menor por memoria (vs 768)
epochs: 100000               # Más épocas para convergencia
eval_interval: 5000          # Evaluación más frecuente

lr: 1e-5                     # Learning rate más conservador
lr_min_ratio: 0.1
lr_warmup_steps: 10000       # Más warmup para estabilidad

weight_decay: 0.01           # Menos regularización
beta1: 0.9
beta2: 0.95

# Sin puzzle embeddings para texto básico
puzzle_emb_lr: 0
puzzle_emb_weight_decay: 0
```

### Métricas de Evaluación

```python
def evaluate_text_quality(predictions, references):
    """
    Métricas específicas para texto
    """
    metrics = {}
    
    # Exactitud a nivel de secuencia
    exact_matches = sum(p == r for p, r in zip(predictions, references))
    metrics["exact_accuracy"] = exact_matches / len(predictions)
    
    # BLEU score para calidad de generación
    from nltk.translate.bleu_score import sentence_bleu
    bleu_scores = [
        sentence_bleu([ref.split()], pred.split()) 
        for pred, ref in zip(predictions, references)
    ]
    metrics["bleu_score"] = np.mean(bleu_scores)
    
    # Longitud promedio generada
    metrics["avg_length"] = np.mean([len(p.split()) for p in predictions])
    
    return metrics
```

---

## 🔍 Decodificación y Generación

### Decodificación No-Autorregresiva (Recomendada)

```python
def decode_text_output(model_output, tokenizer, input_sequence, separator_token=3):
    """
    Decodifica salida del modelo manteniendo arquitectura HRM
    """
    # model_output: [batch, seq_len, vocab_size]
    predicted_tokens = torch.argmax(model_output, dim=-1)
    
    decoded_texts = []
    for batch_idx in range(predicted_tokens.shape[0]):
        pred_seq = predicted_tokens[batch_idx]
        inp_seq = input_sequence[batch_idx]
        
        # Encontrar separador
        sep_positions = torch.where(inp_seq == separator_token)[0]
        
        if len(sep_positions) > 0:
            # Extraer solo parte de salida
            start_output = sep_positions[0] + 1
            output_tokens = pred_seq[start_output:]
            
            # Encontrar EOS
            eos_positions = torch.where(output_tokens == tokenizer.EOS_TOKEN)[0]
            if len(eos_positions) > 0:
                output_tokens = output_tokens[:eos_positions[0]]
            
            decoded_text = tokenizer.decode(output_tokens.tolist())
        else:
            decoded_text = tokenizer.decode(pred_seq.tolist())
        
        decoded_texts.append(decoded_text)
    
    return decoded_texts
```

### Pipeline de Inferencia

```python
class HRLMInference:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def predict(self, input_text):
        """
        Predice texto de salida para entrada dada
        """
        # Preparar entrada
        input_tokens = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_tokens + [self.tokenizer.SEP_TOKEN]], 
                                   dtype=torch.int32)
        
        # Pad a longitud del modelo
        padded_input = F.pad(input_tensor, 
                           (0, self.config.seq_len - input_tensor.size(1)), 
                           value=self.tokenizer.PAD_TOKEN)
        
        batch = {
            "inputs": padded_input,
            "puzzle_identifiers": torch.zeros(1, dtype=torch.int32)
        }
        
        # Inferencia con ACT
        carry = self.model.initial_carry(batch)
        
        while True:
            carry, outputs = self.model(carry, batch)
            if outputs["halted"].all():
                break
        
        # Decodificar salida
        decoded = decode_text_output(
            outputs["logits"], 
            self.tokenizer, 
            padded_input,
            self.tokenizer.SEP_TOKEN
        )
        
        return decoded[0]
```

---

## 💻 Implementación Práctica

### Estructura de Archivos

```
HRLM/
├── models/
│   ├── hrm/
│   │   ├── hrm_text_v1.py          # Modelo adaptado para texto
│   │   └── __init__.py
│   ├── tokenizers/
│   │   ├── char_tokenizer.py       # Tokenizador de caracteres
│   │   ├── hybrid_tokenizer.py     # Tokenizador híbrido
│   │   └── __init__.py
│   ├── layers.py                   # Capas reutilizadas de HRM
│   ├── losses.py                   # Pérdidas adaptadas
│   └── common.py
├── dataset/
│   ├── build_text_dataset.py       # Constructor de datasets de texto
│   ├── text_dataset.py             # Dataset loader para texto
│   └── common.py
├── config/
│   ├── arch/
│   │   └── hrlm_v1.yaml           # Configuración HRLM
│   └── cfg_text_pretrain.yaml     # Configuración de entrenamiento
├── scripts/
│   ├── train_text.py              # Script de entrenamiento
│   ├── evaluate_text.py           # Script de evaluación
│   └── inference_text.py          # Script de inferencia
├── examples/
│   ├── translation/               # Ejemplo de traducción
│   ├── qa/                        # Ejemplo de Q&A
│   └── summarization/             # Ejemplo de resumen
└── HRLM.md                        # Esta documentación
```

### Tokenizador Base

```python
# models/tokenizers/char_tokenizer.py
class CharTokenizer:
    def __init__(self, charset=None):
        if charset is None:
            self.charset = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789 .,!?;:-()[]{}'\"\n\t"
            )
        else:
            self.charset = charset
        
        # Tokens especiales
        self.special_tokens = {
            "<PAD>": 0,
            "<EOS>": 1, 
            "<UNK>": 2,
            "<SEP>": 3
        }
        
        # Crear mapeos
        self.char_to_id = self.special_tokens.copy()
        for i, char in enumerate(self.charset):
            self.char_to_id[char] = i + len(self.special_tokens)
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
    
    @property
    def PAD_TOKEN(self): return 0
    @property  
    def EOS_TOKEN(self): return 1
    @property
    def UNK_TOKEN(self): return 2
    @property
    def SEP_TOKEN(self): return 3
    
    def encode(self, text, add_eos=True):
        tokens = []
        for char in text:
            tokens.append(self.char_to_id.get(char, self.UNK_TOKEN))
        if add_eos:
            tokens.append(self.EOS_TOKEN)
        return tokens
    
    def decode(self, tokens):
        chars = []
        for token in tokens:
            if token == self.EOS_TOKEN:
                break
            elif token == self.PAD_TOKEN:
                continue
            elif token in self.id_to_char:
                chars.append(self.id_to_char[token])
        return ''.join(chars)
```

### Constructor de Dataset

```python
# dataset/build_text_dataset.py
def build_text_dataset(
    input_texts: List[str],
    output_texts: List[str], 
    tokenizer,
    output_dir: str,
    max_seq_len: int = 1024,
    train_split: float = 0.8
):
    """
    Construye dataset de texto compatible con HRM
    """
    assert len(input_texts) == len(output_texts)
    
    # Dividir en train/test
    n_train = int(len(input_texts) * train_split)
    
    splits = {
        "train": (input_texts[:n_train], output_texts[:n_train]),
        "test": (input_texts[n_train:], output_texts[n_train:])
    }
    
    for split_name, (inputs, outputs) in splits.items():
        results = process_text_split(inputs, outputs, tokenizer, max_seq_len)
        
        # Guardar split
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for key, data in results.items():
            np.save(os.path.join(split_dir, f"{key}.npy"), data)
        
        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=max_seq_len,
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.PAD_TOKEN,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=len(inputs),
            mean_puzzle_examples=1,
            sets=[split_name]
        )
        
        with open(os.path.join(split_dir, "metadata.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
    
    print(f"Dataset guardado en {output_dir}")
    return metadata
```

---

## 🎮 Casos de Uso y Aplicaciones

### Nivel 1: Tareas Estructuradas (Más Adecuadas)

#### 1.1 Traducción Simple
```python
# Entrada: "Translate to Spanish: Hello world"
# Salida: "Hola mundo"

input_texts = [
    "Translate to Spanish: Hello world",
    "Translate to Spanish: How are you?",
    "Translate to Spanish: Good morning"
]
output_texts = [
    "Hola mundo",
    "¿Cómo estás?", 
    "Buenos días"
]
```

#### 1.2 Transformación de Formato
```python
# JSON a texto natural
input_texts = [
    'Convert to sentence: {"name": "John", "age": 25}',
    'Convert to sentence: {"city": "Madrid", "country": "Spain"}'
]
output_texts = [
    "John is 25 years old",
    "Madrid is in Spain"
]
```

#### 1.3 Respuesta a Preguntas Factuales
```python
input_texts = [
    "Question: What is the capital of France?",
    "Question: What is 2 + 2?",
    "Question: What color is the sky?"
]
output_texts = [
    "Paris",
    "4", 
    "Blue"
]
```

### Nivel 2: Tareas de Razonamiento (Moderadamente Adecuadas)

#### 2.1 Corrección Gramatical
```python
input_texts = [
    "Fix grammar: I are going to school",
    "Fix grammar: She don't like apples"
]
output_texts = [
    "I am going to school",
    "She doesn't like apples"
]
```

#### 2.2 Extracción de Información
```python
input_texts = [
    "Extract name: My name is John Smith and I live in Madrid",
    "Extract date: The meeting is on March 15, 2024"
]
output_texts = [
    "John Smith",
    "March 15, 2024"
]
```

### Nivel 3: Tareas Creativas (Menos Adecuadas)

#### 3.1 Resumen Extractivo
```python
input_texts = [
    "Summarize: The weather today is very nice and sunny. Perfect for outdoor activities."
]
output_texts = [
    "Nice sunny weather, good for outdoors"
]
```

---

## ⚠️ Desafíos y Limitaciones

### Desafíos Técnicos

#### 1. **Escalabilidad de Memoria**
```python
# Problema: Vocabulario grande aumenta memoria exponencialmente
# Sudoku: vocab_size=11, hidden_size=512
# Embedding: 11 * 512 = 5.6K parámetros
# LM Head: 512 * 11 = 5.6K parámetros

# Texto: vocab_size=10000, hidden_size=768  
# Embedding: 10000 * 768 = 7.68M parámetros
# LM Head: 768 * 10000 = 7.68M parámetros
# Total: ~15M parámetros adicionales solo en embeddings
```

#### 2. **Longitud de Secuencia**
```python
# Atención cuadrática: O(n²)
# Sudoku: 81² = 6,561 operaciones
# Texto: 1024² = 1,048,576 operaciones (~160x más)
```

#### 3. **Convergencia de Entrenamiento**
```python
# HRM converge rápido en Sudoku (pocos parámetros, datos estructurados)
# Texto requiere:
# - Más épocas de entrenamiento
# - Learning rates más pequeños
# - Más datos para generalización
```

### Limitaciones Conceptuales

#### 1. **Naturaleza No-Autorregresiva**
- ✅ **Ventaja**: Generación paralela, más rápida
- ❌ **Limitación**: Difícil mantener coherencia en texto largo
- ❌ **Limitación**: No puede corregir errores anteriores

#### 2. **Razonamiento Jerárquico en Texto**
```python
# HRM asume separación clara:
# H-Level: Planificación abstracta
# L-Level: Ejecución detallada

# En texto, esta separación puede no ser natural:
# - Sintaxis y semántica están entrelazadas
# - Decisiones locales afectan estructura global
# - Diferentes tipos de texto requieren diferentes patrones
```

#### 3. **Evaluación de Calidad**
```python
# Sudoku: Solución correcta o incorrecta (binario)
# Texto: Múltiples respuestas válidas, calidad subjetiva

# Métricas necesarias:
# - BLEU, ROUGE para calidad
# - Coherencia semántica  
# - Fluidez y naturalidad
# - Relevancia contextual
```

### Limitaciones de Datos

#### 1. **Eficiencia de Datos**
- HRM aprende Sudoku con 1000 ejemplos
- Texto puede requerir 10K-100K ejemplos para generalizar
- Vocabulario grande dificulta few-shot learning

#### 2. **Diversidad Lingüística**
- Modelos de caracteres luchan con idiomas no latinos
- Tokenización híbrida requiere vocabularios específicos por idioma
- Transferencia entre idiomas limitada

---

## 🚀 Guía de Implementación

### Fase 1: Preparación del Entorno

#### 1.1 Instalación de Dependencias
```bash
# Clonar repositorio HRM base
git clone https://github.com/sapientinc/HRM.git
cd HRM

# Instalar dependencias base
pip install -r requirements.txt

# Dependencias adicionales para texto
pip install nltk transformers datasets
```

#### 1.2 Estructura de Proyecto
```bash
# Crear estructura para HRLM
mkdir -p models/tokenizers
mkdir -p dataset/text
mkdir -p config/text
mkdir -p examples/text
mkdir -p scripts/text
```

### Fase 2: Implementación del Tokenizador

#### 2.1 Tokenizador de Caracteres
```python
# Crear models/tokenizers/char_tokenizer.py
# (Ver implementación completa en sección anterior)

# Probar tokenizador
python -c "
from models.tokenizers.char_tokenizer import CharTokenizer
tokenizer = CharTokenizer()
text = 'Hello world!'
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
print(f'Original: {text}')
print(f'Encoded: {encoded}')
print(f'Decoded: {decoded}')
print(f'Vocab size: {tokenizer.vocab_size}')
"
```

### Fase 3: Construcción del Dataset

#### 3.1 Dataset de Ejemplo
```python
# Crear dataset/text/build_simple_dataset.py
from models.tokenizers.char_tokenizer import CharTokenizer
from dataset.build_text_dataset import build_text_dataset

# Datos de ejemplo para traducción
input_texts = [
    "Translate to Spanish: Hello",
    "Translate to Spanish: Goodbye", 
    "Translate to Spanish: Thank you",
    "Translate to Spanish: Good morning",
    "Translate to Spanish: How are you?"
] * 200  # Repetir para tener más datos

output_texts = [
    "Hola",
    "Adiós",
    "Gracias", 
    "Buenos días",
    "¿Cómo estás?"
] * 200

# Construir dataset
tokenizer = CharTokenizer()
metadata = build_text_dataset(
    input_texts,
    output_texts,
    tokenizer,
    output_dir="data/translation-simple",
    max_seq_len=256
)
```

#### 3.2 Verificar Dataset
```python
# Verificar que el dataset se construyó correctamente
python -c "
import numpy as np
import json

# Cargar datos
inputs = np.load('data/translation-simple/train/inputs.npy')
labels = np.load('data/translation-simple/train/labels.npy')

with open('data/translation-simple/train/metadata.json') as f:
    metadata = json.load(f)

print(f'Inputs shape: {inputs.shape}')
print(f'Labels shape: {labels.shape}')
print(f'Vocab size: {metadata[\"vocab_size\"]}')
print(f'Seq len: {metadata[\"seq_len\"]}')
"
```

### Fase 4: Configuración del Modelo

#### 4.1 Configuración HRLM
```yaml
# Crear config/text/hrlm_v1.yaml
name: hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1
loss:
  name: losses@ACTLossHead
  loss_type: softmax_cross_entropy

# Configuración adaptada para texto
halt_exploration_prob: 0.1
halt_max_steps: 32

H_cycles: 4
L_cycles: 4

H_layers: 6
L_layers: 6

hidden_size: 768
num_heads: 12
expansion: 4

puzzle_emb_ndim: 0  # Sin puzzle embeddings para texto
pos_encodings: rope
```

#### 4.2 Configuración de Entrenamiento
```yaml
# Crear config/text/cfg_text_pretrain.yaml
defaults:
  - arch: text/hrlm_v1
  - _self_

# Datos
data_path: data/translation-simple

# Hiperparámetros adaptados para texto
global_batch_size: 32
epochs: 50000
eval_interval: 5000

lr: 1e-5
lr_min_ratio: 0.1
lr_warmup_steps: 10000

weight_decay: 0.01
beta1: 0.9
beta2: 0.95

# Sin puzzle embeddings
puzzle_emb_lr: 0
puzzle_emb_weight_decay: 0
```

### Fase 5: Entrenamiento

#### 5.1 Script de Entrenamiento
```python
# Crear scripts/text/train_text.py
import sys
sys.path.append('.')

from pretrain import launch
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../../config/text", config_name="cfg_text_pretrain", version_base=None)
def main(cfg: DictConfig):
    launch(cfg)

if __name__ == "__main__":
    main()
```

#### 5.2 Ejecutar Entrenamiento
```bash
# Entrenamiento en GPU única
cd scripts/text
python train_text.py

# Entrenamiento distribuido (múltiples GPUs)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 4 train_text.py
```

### Fase 6: Evaluación e Inferencia

#### 6.1 Script de Evaluación
```python
# Crear scripts/text/evaluate_text.py
from models.tokenizers.char_tokenizer import CharTokenizer
from evaluate import launch as evaluate_launch

def evaluate_text_model(checkpoint_path):
    # Cargar tokenizador
    tokenizer = CharTokenizer()
    
    # Evaluar modelo
    results = evaluate_launch(checkpoint=checkpoint_path)
    
    return results
```

#### 6.2 Script de Inferencia
```python
# Crear scripts/text/inference_text.py
import torch
from models.tokenizers.char_tokenizer import CharTokenizer

class HRLMInference:
    def __init__(self, checkpoint_path):
        self.tokenizer = CharTokenizer()
        # Cargar modelo (implementar carga desde checkpoint)
        
    def predict(self, input_text):
        # Implementar inferencia
        # (Ver implementación completa en sección anterior)
        pass

# Ejemplo de uso
if __name__ == "__main__":
    model = HRLMInference("checkpoints/hrlm/step_50000")
    
    result = model.predict("Translate to Spanish: Hello world")
    print(f"Input: Translate to Spanish: Hello world")
    print(f"Output: {result}")
```

---

## ⚙️ Configuración Recomendada

### Configuración Base (Desarrollo)

```yaml
# Para experimentación inicial
arch:
  hidden_size: 512
  H_cycles: 2
  L_cycles: 2
  H_layers: 4
  L_layers: 4
  num_heads: 8
  halt_max_steps: 16

training:
  global_batch_size: 16
  lr: 5e-5
  epochs: 10000
  
data:
  vocab_size: 100    # Tokenizador de caracteres básico
  seq_len: 256       # Secuencias cortas
```

### Configuración Intermedia (Producción Pequeña)

```yaml
# Para tareas reales pequeñas
arch:
  hidden_size: 768
  H_cycles: 3
  L_cycles: 3  
  H_layers: 6
  L_layers: 6
  num_heads: 12
  halt_max_steps: 32

training:
  global_batch_size: 32
  lr: 1e-5
  epochs: 50000

data:
  vocab_size: 1000   # Tokenizador híbrido
  seq_len: 512       # Secuencias medianas
```

### Configuración Avanzada (Producción Grande)

```yaml
# Para tareas complejas
arch:
  hidden_size: 1024
  H_cycles: 4
  L_cycles: 4
  H_layers: 8
  L_layers: 8
  num_heads: 16
  halt_max_steps: 64

training:
  global_batch_size: 64
  lr: 5e-6
  epochs: 100000

data:
  vocab_size: 10000  # BPE/WordPiece
  seq_len: 1024      # Secuencias largas
```

### Recursos Computacionales

| Configuración | GPU Mínima | Memoria GPU | Tiempo Entrenamiento |
|---------------|------------|-------------|---------------------|
| **Base** | RTX 3070 | 8GB | ~6 horas |
| **Intermedia** | RTX 4080 | 16GB | ~24 horas |
| **Avanzada** | A100 | 40GB | ~72 horas |

---

## 🗺️ Roadmap de Desarrollo

### Fase 1: Fundamentos (Semanas 1-4)

#### Semana 1: Preparación
- [ ] Configurar entorno de desarrollo
- [ ] Implementar tokenizador de caracteres
- [ ] Crear dataset de traducción simple
- [ ] Verificar compatibilidad con HRM base

#### Semana 2: Modelo Base
- [ ] Adaptar configuración HRM para texto
- [ ] Modificar embeddings y heads de salida
- [ ] Implementar función de pérdida para texto
- [ ] Probar entrenamiento en dataset pequeño

#### Semana 3: Entrenamiento
- [ ] Entrenar modelo en traducción simple
- [ ] Implementar métricas de evaluación
- [ ] Optimizar hiperparámetros
- [ ] Validar convergencia

#### Semana 4: Inferencia
- [ ] Implementar pipeline de inferencia
- [ ] Crear sistema de decodificación
- [ ] Probar generación de texto
- [ ] Documentar resultados iniciales

### Fase 2: Optimización (Semanas 5-8)

#### Semana 5: Tokenización Avanzada
- [ ] Implementar tokenizador híbrido
- [ ] Comparar rendimiento vs caracteres
- [ ] Optimizar vocabulario
- [ ] Evaluar eficiencia de memoria

#### Semana 6: Arquitectura
- [ ] Experimentar con más capas/ciclos
- [ ] Optimizar mecanismo ACT para texto
- [ ] Probar diferentes configuraciones
- [ ] Analizar trade-offs rendimiento/velocidad

#### Semana 7: Datos y Entrenamiento
- [ ] Expandir datasets de entrenamiento
- [ ] Implementar data augmentation
- [ ] Optimizar pipeline de datos
- [ ] Mejorar estabilidad de entrenamiento

#### Semana 8: Evaluación
- [ ] Implementar métricas avanzadas (BLEU, ROUGE)
- [ ] Crear benchmark de evaluación
- [ ] Comparar con baselines
- [ ] Analizar casos de fallo

### Fase 3: Aplicaciones (Semanas 9-12)

#### Semana 9: Casos de Uso
- [ ] Implementar Q&A factual
- [ ] Crear sistema de corrección gramatical
- [ ] Probar extracción de información
- [ ] Evaluar en tareas estructuradas

#### Semana 10: Escalabilidad
- [ ] Optimizar para secuencias largas
- [ ] Implementar entrenamiento distribuido
- [ ] Probar en datasets grandes
- [ ] Optimizar memoria y velocidad

#### Semana 11: Robustez
- [ ] Probar en diferentes dominios
- [ ] Evaluar generalización
- [ ] Implementar manejo de errores
- [ ] Crear tests de regresión

#### Semana 12: Producción
- [ ] Crear API de inferencia
- [ ] Implementar sistema de logging
- [ ] Crear documentación de usuario
- [ ] Preparar release inicial

### Fase 4: Investigación Avanzada (Semanas 13+)

#### Investigación Continua
- [ ] Explorar generación autorregresiva híbrida
- [ ] Investigar adaptación a idiomas no latinos
- [ ] Experimentar con arquitecturas multimodales
- [ ] Desarrollar técnicas de fine-tuning eficiente

#### Optimizaciones Avanzadas
- [ ] Implementar attention sparse para secuencias largas
- [ ] Explorar quantización y pruning
- [ ] Desarrollar técnicas de distillation
- [ ] Investigar arquitecturas neurales evolutivas

---

## 📊 Métricas de Éxito

### Métricas Técnicas

#### Rendimiento del Modelo
- **Exactitud**: >90% en tareas de traducción simple
- **BLEU Score**: >0.7 para traducciones de calidad
- **Convergencia**: <10K pasos para tareas simples
- **Memoria**: <2GB GPU para configuración base

#### Eficiencia de Datos
- **Few-shot**: Aprendizaje efectivo con <5K ejemplos
- **Generalización**: >80% exactitud en datos no vistos
- **Transferencia**: Adaptación rápida a nuevas tareas

### Métricas de Negocio

#### Usabilidad
- **Latencia**: <100ms para inferencia en secuencias cortas
- **Throughput**: >100 ejemplos/segundo en batch
- **Escalabilidad**: Soporte para múltiples usuarios concurrentes

#### Calidad
- **Coherencia**: Texto generado coherente y gramaticalmente correcto
- **Relevancia**: Respuestas apropiadas al contexto
- **Diversidad**: Capacidad de generar respuestas variadas

---

## 🔗 Referencias y Recursos

### Papers Fundamentales
1. **HRM Original**: "Hierarchical Reasoning Model" - Wang et al. (2025)
2. **Adaptive Computation Time**: "Adaptive Computation Time for Recurrent Neural Networks" - Graves (2016)
3. **Transformer Architecture**: "Attention Is All You Need" - Vaswani et al. (2017)

### Implementaciones de Referencia
- **HRM Base**: [sapientinc/HRM](https://github.com/sapientinc/HRM)
- **FlashAttention**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **Transformers**: [huggingface/transformers](https://github.com/huggingface/transformers)

### Datasets Sugeridos
- **Traducción**: WMT datasets, OPUS collections
- **Q&A**: SQuAD, Natural Questions
- **Corrección**: CoNLL-2014, JFLEG
- **Resumen**: CNN/DailyMail, XSum

### Herramientas y Librerías
```python
# Tokenización
import transformers  # BPE/WordPiece tokenizers
import sentencepiece  # SentencePiece tokenizer

# Evaluación
import nltk  # BLEU, ROUGE scores
import sacrebleu  # Standardized BLEU
import bert_score  # Semantic similarity

# Visualización
import wandb  # Experiment tracking
import tensorboard  # Training visualization
import matplotlib  # Custom plots
```

---

## 📝 Conclusiones

### Viabilidad Técnica: ✅ CONFIRMADA

La adaptación de HRM para procesamiento de texto es **técnicamente viable** y mantiene las ventajas fundamentales del modelo original:

1. **Eficiencia de Datos**: Potencial para aprendizaje few-shot en tareas de texto
2. **Razonamiento Jerárquico**: Separación útil entre planificación y ejecución
3. **Computación Adaptativa**: ACT permite procesamiento variable según complejidad
4. **Estabilidad**: Arquitectura recurrente probada y estable

### Casos de Uso Óptimos

HRLM es especialmente adecuado para:
- ✅ **Tareas estructuradas** con patrones claros
- ✅ **Transformaciones determinísticas** de texto
- ✅ **Dominios específicos** con vocabulario limitado
- ✅ **Aplicaciones few-shot** con datos limitados

### Próximos Pasos

1. **Implementar prototipo** siguiendo la guía de implementación
2. **Validar en tareas simples** como traducción básica
3. **Optimizar configuración** para casos de uso específicos
4. **Expandir gradualmente** a tareas más complejas
5. **Contribuir al ecosistema** HRM con extensiones de texto

### Impacto Esperado

HRLM tiene el potencial de:
- **Democratizar NLP** con modelos eficientes en datos
- **Acelerar prototipado** de aplicaciones de texto
- **Reducir costos** de entrenamiento para tareas específicas
- **Avanzar investigación** en razonamiento jerárquico para texto

---

*Documentación creada para guiar la implementación de Hierarchical Reasoning Language Model (HRLM) basado en la arquitectura HRM original.*

**Versión**: 1.0  
**Fecha**: Enero 2025  
**Autor**: Análisis basado en arquitectura HRM de sapientinc  
**Licencia**: Seguir licencia del proyecto HRM original