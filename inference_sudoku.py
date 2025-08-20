#!/usr/bin/env python3
"""
Script de inferencia simple para resolver Sudokus usando el modelo HRM preentrenado.
"""

import os
import sys
import yaml
import json
import numpy as np
import torch
import pydantic
from typing import Optional, Any, Sequence, List

# SOLUCIÓN LIMPIA: Monkey-patch inteligente de flash_attn que detecta CPU automáticamente
import torch.nn.functional as F

def cpu_fallback_flash_attn(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), alibi_slopes=None, deterministic=False, return_attn_probs=False):
    """Fallback de atención estándar para CPU cuando flash_attn no puede ejecutarse"""
    # Transponer para compatibilidad: [b, s, h, d] -> [b, h, s, d]
    q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Calcular atención estándar
    scale = softmax_scale or (q.size(-1) ** -0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if causal:
        seq_len = q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    
    # Aplicar dropout si se especifica
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=False)
    
    out = torch.matmul(attn, v)
    
    # Transponer de vuelta: [b, h, s, d] -> [b, s, h, d] y hacer contiguo
    out = out.transpose(1, 2).contiguous()
    
    if return_attn_probs:
        return out, attn.transpose(1, 2).contiguous()
    return out

# Importar flash_attn y hacer monkey-patch inteligente
try:
    import flash_attn
    from flash_attn import flash_attn_func
    
    # Guardar la función original
    _original_flash_attn_func = flash_attn_func
    
    def smart_flash_attn_func(*args, **kwargs):
        """Wrapper inteligente que detecta CPU y usa fallback automáticamente"""
        # Detectar si estamos en CPU chequeando el primer tensor
        if args:  # q es el primer argumento
            q = args[0]
            if q.device.type == 'cpu':
                # Usar fallback para CPU
                return cpu_fallback_flash_attn(*args, **kwargs)
        
        # Si estamos en CUDA, usar la implementación original
        try:
            return _original_flash_attn_func(*args, **kwargs)
        except NotImplementedError as e:
            if 'CPU' in str(e):
                # Fallback automático si detecta error de CPU
                return cpu_fallback_flash_attn(*args, **kwargs)
            else:
                raise  # Re-lanzar otros errores
    
    # Reemplazar la función en todos los lugares donde se importa
    flash_attn.flash_attn_func = smart_flash_attn_func
    
    # También patchear flash_attn_interface si existe
    try:
        import flash_attn_interface
        flash_attn_interface.flash_attn_func = smart_flash_attn_func
    except ImportError:
        pass
    
    # Patchear el módulo en sys.modules para futuras importaciones
    sys.modules['flash_attn'].flash_attn_func = smart_flash_attn_func
    
    print("🔧 Monkey-patch inteligente de flash_attn aplicado (CPU auto-fallback)")
    
except ImportError:
    print("❌ flash_attn no está instalado")
    sys.exit(1)

# Evitar importación directa de pretrain.py que requiere adam_atan2
# En su lugar, importaremos solo lo necesario
sys.path.append(os.path.dirname(__file__))

from utils.functions import load_model_class
from puzzle_dataset import PuzzleDatasetMetadata


def load_model(checkpoint_path):
    """Carga el modelo desde el checkpoint."""
    
    config_path = os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Crear metadata compatible con Sudoku
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=10 + 1,  # PAD + "0" ... "9"
        
        pad_id=0,
        ignore_label_id=0,
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=1,  # Para inferencia individual
        mean_puzzle_examples=1,
        sets=["all"]
    )
    
    # Cargar el modelo directamente
    model_class = load_model_class(config_dict['arch']['name'])
    
    # Preparar la configuración para el modelo
    model_config = {**config_dict['arch']}
    model_config.update({
        'batch_size': 1,  # Para inferencia individual
        'seq_len': metadata.seq_len,
        'puzzle_emb_ndim': model_config.get('puzzle_emb_ndim', 512),
        'num_puzzle_identifiers': metadata.num_puzzle_identifiers,
        'vocab_size': metadata.vocab_size,
    })
    
    model = model_class(model_config)
    
    # Cargar pesos del checkpoint
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Limpiar prefijos del checkpoint
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # Remover prefijo _orig_mod. si existe (modelos compilados)
            clean_key = k.removeprefix("_orig_mod.")
            # Remover prefijo model. si existe 
            clean_key = clean_key.removeprefix("model.")
            cleaned_state_dict[clean_key] = v
        
        model.load_state_dict(cleaned_state_dict, strict=True)
    except Exception as e:
        print(f"Error al cargar estado del modelo: {e}")
        # Intentar con strict=False
        try:
            model.load_state_dict(cleaned_state_dict, strict=False)
            print("Cargado con strict=False")
        except Exception as e2:
            print(f"Error incluso con strict=False: {e2}")
            raise
    
    model.eval()
    
    return model, config_dict, metadata


def sudoku_to_input(sudoku_grid):
    """Convierte una matriz de Sudoku 9x9 a formato de entrada del modelo."""
    # El modelo espera un formato específico, necesitamos revisar cómo se procesan los datos
    # Por ahora, asumimos que es una matriz 9x9 con 0 para celdas vacías
    return sudoku_grid.astype(np.int32)


def solve_sudoku(model, config, metadata, sudoku_grid):
    """Resuelve un Sudoku usando el modelo entrenado."""
    
    print("Sudoku a resolver:")
    print_sudoku(sudoku_grid)
    
    # Preparar entrada - el modelo espera un diccionario batch
    # Convertir 9x9 a secuencia de 81 elementos
    # IMPORTANTE: El dataset suma 1 a todos los valores (PAD=0, luego 1-9 para dígitos 0-8 del Sudoku)
    input_sequence = sudoku_grid.flatten() + 1  # Convertir 0-9 a 1-10
    input_tensor = torch.from_numpy(input_sequence).unsqueeze(0).long()
    
    # Crear el batch en el formato esperado por el modelo
    batch = {
        "inputs": input_tensor,
        "puzzle_identifiers": torch.zeros(1, dtype=torch.long),  # Single puzzle type
    }
    
    print(f"Forma de entrada: {input_tensor.shape}")
    
    with torch.no_grad():
        try:
            # Inicializar carry para el modelo ACT
            carry = model.initial_carry(batch)
            
            # El modelo ACT requiere múltiples iteraciones para converger
            max_steps = config['arch']['halt_max_steps']
            print(f"Ejecutando inferencia ACT con máximo {max_steps} pasos...")
            
            # Ejecutar múltiples iteraciones hasta convergencia
            for step in range(max_steps):
                carry, outputs = model(carry, batch)
                
                # Mostrar progreso
                current_steps = carry.steps[0].item()
                is_halted = carry.halted[0].item()
                print(f"  Paso {step + 1}: steps={current_steps}, halted={is_halted}")
                
                # Verificar si el modelo ha decidido parar (todas las secuencias están halted)
                if torch.all(carry.halted):
                    print(f"✅ Convergencia alcanzada en el paso {step + 1}")
                    break
                    
                if step == max_steps - 1:
                    print(f"⚠️  Máximo de pasos ({max_steps}) alcanzado sin convergencia explícita")
            
            # Los outputs finales contienen las predicciones después de convergencia
            
            print(f"Tipo de salida: {type(outputs)}")
            if isinstance(outputs, dict):
                print(f"Claves de salida: {outputs.keys()}")
            
            # Extraer la solución - usamos los logits que son las predicciones del modelo
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']  # Shape: [batch_size, seq_len, vocab_size]
                
                print(f"Forma de logits: {logits.shape}")
                
                # Tomar argmax sobre la dimensión del vocabulario
                predictions = logits.argmax(dim=-1)  # Shape: [batch_size, seq_len]
                
                print(f"Forma de predicciones: {predictions.shape}")
                
                # Extraer el primer (y único) batch
                solution_flat = predictions[0].cpu().numpy()  # Shape: [seq_len]
                
                print(f"Predicciones raw: {solution_flat[:10]}...")  # Mostrar primeros 10 valores
                
                # Convertir de 1-10 de vuelta a 0-9 (el dataset suma 1 a todos los valores)
                solution_flat = solution_flat - 1
                
                # Clip para asegurar valores válidos
                solution_flat = np.clip(solution_flat, 0, 9)
                
                print(f"Predicciones procesadas: {solution_flat[:10]}...")
                
                # Reshape a 9x9
                if len(solution_flat) >= 81:
                    solution_grid = solution_flat[:81].reshape(9, 9)
                else:
                    print(f"Error: Solución tiene solo {len(solution_flat)} elementos, esperaba 81")
                    return None
            else:
                print("Error: No se encontraron logits en la salida del modelo")
                return None
            
            print("\nSolución generada:")
            print_sudoku(solution_grid)
            
            # Validar la solución
            print("\n🔍 VALIDANDO SOLUCIÓN...")
            is_valid, errors = validate_sudoku(solution_grid)
            is_complete = is_sudoku_complete(solution_grid)
            
            if is_complete:
                if is_valid:
                    print("✅ ¡SOLUCIÓN VÁLIDA Y COMPLETA!")
                    print("🎉 El Sudoku ha sido resuelto correctamente")
                else:
                    print("❌ SOLUCIÓN COMPLETA PERO INVÁLIDA")
                    print("🚫 Errores encontrados:")
                    for error in errors:
                        print(f"   • {error}")
            else:
                empty_cells = np.sum(solution_grid == 0)
                print(f"⚠️  SOLUCIÓN INCOMPLETA ({empty_cells} celdas vacías)")
                if errors:
                    print("🚫 Errores adicionales:")
                    for error in errors:
                        print(f"   • {error}")
                else:
                    print("✅ Pero las celdas llenas son válidas")
            
            return solution_grid
                
        except Exception as e:
            print(f"Error durante la inferencia: {e}")
            import traceback
            traceback.print_exc()
            return None


def validate_sudoku(grid):
    """
    Valida si una solución de Sudoku es correcta.
    
    Args:
        grid: numpy array 9x9 con la solución propuesta
        
    Returns:
        tuple: (is_valid: bool, errors: list)
    """
    errors = []
    
    # Verificar que todos los valores están en el rango 1-9 (0 para celdas vacías)
    if not np.all((grid >= 0) & (grid <= 9)):
        errors.append("Valores fuera del rango 0-9 encontrados")
        return False, errors
    
    # Verificar filas
    for i in range(9):
        row = grid[i, :]
        # Solo considerar valores no-cero para validación
        non_zero = row[row != 0]
        if len(non_zero) != len(np.unique(non_zero)):
            errors.append(f"Fila {i+1} tiene números duplicados")
        
        # Si la fila está completa, verificar que tenga todos los números 1-9
        if len(non_zero) == 9:
            expected = set(range(1, 10))
            actual = set(non_zero)
            if actual != expected:
                missing = expected - actual
                extra = actual - expected
                if missing:
                    errors.append(f"Fila {i+1} le faltan números: {sorted(missing)}")
                if extra:
                    errors.append(f"Fila {i+1} tiene números inválidos: {sorted(extra)}")
    
    # Verificar columnas
    for j in range(9):
        col = grid[:, j]
        non_zero = col[col != 0]
        if len(non_zero) != len(np.unique(non_zero)):
            errors.append(f"Columna {j+1} tiene números duplicados")
            
        # Si la columna está completa, verificar que tenga todos los números 1-9
        if len(non_zero) == 9:
            expected = set(range(1, 10))
            actual = set(non_zero)
            if actual != expected:
                missing = expected - actual
                extra = actual - expected
                if missing:
                    errors.append(f"Columna {j+1} le faltan números: {sorted(missing)}")
                if extra:
                    errors.append(f"Columna {j+1} tiene números inválidos: {sorted(extra)}")
    
    # Verificar cajas 3x3
    for box_row in range(3):
        for box_col in range(3):
            # Extraer la caja 3x3
            box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
            non_zero = box[box != 0]
            
            if len(non_zero) != len(np.unique(non_zero)):
                errors.append(f"Caja ({box_row+1}, {box_col+1}) tiene números duplicados")
                
            # Si la caja está completa, verificar que tenga todos los números 1-9
            if len(non_zero) == 9:
                expected = set(range(1, 10))
                actual = set(non_zero)
                if actual != expected:
                    missing = expected - actual
                    extra = actual - expected
                    if missing:
                        errors.append(f"Caja ({box_row+1}, {box_col+1}) le faltan números: {sorted(missing)}")
                    if extra:
                        errors.append(f"Caja ({box_row+1}, {box_col+1}) tiene números inválidos: {sorted(extra)}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def is_sudoku_complete(grid):
    """
    Verifica si el Sudoku está completamente resuelto (sin celdas vacías).
    
    Args:
        grid: numpy array 9x9
        
    Returns:
        bool: True si está completo, False si tiene celdas vacías (0)
    """
    return not np.any(grid == 0)


def print_sudoku(grid):
    """Imprime un Sudoku de forma legible."""
    print("┌───────┬───────┬───────┐")
    for i in range(9):
        if i == 3 or i == 6:
            print("├───────┼───────┼───────┤")
        
        row = "│ "
        for j in range(9):
            if j == 3 or j == 6:
                row += "│ "
            
            if grid[i, j] == 0:
                row += ". "
            else:
                row += f"{grid[i, j]} "
        
        row += "│"
        print(row)
    
    print("└───────┴───────┴───────┘")


def test_validator():
    """Función para probar el validador con ejemplos conocidos."""
    print("🧪 PROBANDO VALIDADOR DE SUDOKU")
    print("="*40)
    
    # Sudoku válido y completo
    valid_complete = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])
    
    # Sudoku con error en fila (5 repetido)
    invalid_row = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 5],  # 5 repetido
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])
    
    # Sudoku incompleto pero válido
    incomplete_valid = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    test_cases = [
        ("Sudoku válido y completo", valid_complete),
        ("Sudoku inválido (fila con duplicados)", invalid_row), 
        ("Sudoku incompleto pero válido", incomplete_valid)
    ]
    
    for name, grid in test_cases:
        print(f"\n📋 {name}:")
        print_sudoku(grid)
        is_valid, errors = validate_sudoku(grid)
        is_complete = is_sudoku_complete(grid)
        
        print(f"   Completo: {'✅' if is_complete else '❌'}")
        print(f"   Válido: {'✅' if is_valid else '❌'}")
        if errors:
            print("   Errores:")
            for error in errors:
                print(f"      • {error}")
        print()


def main():
    """Función principal."""
    import sys
    
    # Verificar si se quiere probar el validador
    if len(sys.argv) > 1 and sys.argv[1] == "--test-validator":
        test_validator()
        return
        
    # Ruta al checkpoint
    checkpoint_path = "/home/jal/github/HRM/checkpoints/sudoku/checkpoint"
    
    print("Cargando modelo HRM para Sudoku...")
    try:
        model, config, metadata = load_model(checkpoint_path)
        print("✅ Modelo cargado exitosamente!")
        print(f"Configuración: {config['arch']['name']}")
        print(f"Tamaño oculto: {config['arch']['hidden_size']}")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return
    
    # Ejemplo de Sudoku difícil (puedes cambiar este por cualquier otro)
    example_sudoku = np.array([
        [0, 0, 0, 0, 0, 0, 6, 8, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 0],
        [7, 0, 0, 0, 9, 0, 5, 0, 0],
        [5, 7, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 2],
        [0, 0, 5, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    print("\n" + "="*50)
    print("RESOLVIENDO SUDOKU CON HRM")
    print("="*50)
    
    # Resolver el Sudoku
    try:
        solution = solve_sudoku(model, config, metadata, example_sudoku)
        if solution is not None:
            print("\n🏁 Proceso de inferencia completado.")
        else:
            print("\n❌ Error durante el proceso de inferencia")
    except Exception as e:
        print(f"\n❌ Error durante la inferencia: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
