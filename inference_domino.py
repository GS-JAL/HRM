#!/usr/bin/env python3
"""
Inferencia del modelo de dominó - Hierarchical Reasoning Model (HRM)

Este script permite usar el modelo HRM entrenado para predecir jugadas óptimas de dominó.

Requisitos:
    - Modelo entrenado disponible en checkpoints/domino/
    
Uso:
    python3 inference_domino.py

El modelo predice la jugada óptima dado:
    - Estado actual de la mesa (fichas colocadas)
    - Fichas disponibles del jugador  
    - Extremos donde se puede jugar
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Tuple, Optional

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import launch as evaluate_launch
from puzzle_dataset import PuzzleDatasetMetadata


def load_domino_model(checkpoint_path: str):
    """Cargar modelo de dominó entrenado"""
    # TODO: Implementar carga del modelo cuando esté disponible el checkpoint
    # Similar a inference_sudoku.py
    print(f"⚠️  Cargando modelo desde {checkpoint_path}")
    print("⚠️  Esta funcionalidad requiere un checkpoint entrenado")
    return None


def encode_domino_state(
    table_tiles: List[Tuple[int, int]], 
    player_tiles: List[Tuple[int, int]], 
    left_end: Optional[int], 
    right_end: Optional[int]
) -> np.ndarray:
    """
    Codificar el estado del juego de dominó en el formato esperado por el modelo
    
    Args:
        table_tiles: Lista de fichas en la mesa
        player_tiles: Lista de fichas del jugador
        left_end: Extremo izquierdo disponible para jugar
        right_end: Extremo derecho disponible para jugar
    
    Returns:
        Array numpy con la codificación del estado (72 elementos)
    """
    input_seq = np.zeros(72, dtype=np.int32)
    
    # Codificar fichas de la mesa (posiciones 0-55)
    table_pos = 0
    for tile in table_tiles[:28]:  # Máximo 28 fichas
        input_seq[table_pos] = tile[0] + 1      # +1 para evitar 0 (padding)
        input_seq[table_pos + 1] = tile[1] + 1
        table_pos += 2
    
    # Codificar fichas del jugador (posiciones 56-69)
    player_pos = 56
    for tile in player_tiles[:7]:   # Máximo 7 fichas
        input_seq[player_pos] = tile[0] + 1
        input_seq[player_pos + 1] = tile[1] + 1
        player_pos += 2
    
    # Codificar extremos (posiciones 70-71)
    input_seq[70] = (left_end + 1) if left_end is not None else 0
    input_seq[71] = (right_end + 1) if right_end is not None else 0
    
    return input_seq


def decode_domino_prediction(output: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
    """
    Decodificar la predicción del modelo
    
    Args:
        output: Array numpy con la predicción (3 elementos)
    
    Returns:
        Tupla con (ficha_optima, posicion_optima)
    """
    if output[0] == 0 or output[1] == 0:  # No hay predicción válida
        return None, None
    
    # Decodificar ficha (restar 1 para volver a valores 0-6)
    optimal_tile = (output[0] - 1, output[1] - 1)
    
    # Decodificar posición
    position_map = {1: 'left', 2: 'right', 3: 'first'}
    optimal_position = position_map.get(output[2], None)
    
    return optimal_tile, optimal_position


def predict_optimal_move(
    model, 
    table_tiles: List[Tuple[int, int]], 
    player_tiles: List[Tuple[int, int]], 
    left_end: Optional[int], 
    right_end: Optional[int]
) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
    """
    Predecir la jugada óptima dado el estado del juego
    
    Args:
        model: Modelo entrenado
        table_tiles: Fichas en la mesa
        player_tiles: Fichas del jugador
        left_end: Extremo izquierdo disponible
        right_end: Extremo derecho disponible
    
    Returns:
        Tupla con (ficha_optima, posicion_optima)
    """
    if model is None:
        print("⚠️  Modelo no cargado")
        return None, None
    
    # Codificar estado
    input_state = encode_domino_state(table_tiles, player_tiles, left_end, right_end)
    
    # TODO: Implementar predicción con el modelo
    # input_tensor = torch.from_numpy(input_state).unsqueeze(0)
    # with torch.no_grad():
    #     output = model(input_tensor)
    # output_np = output.cpu().numpy()[0]
    
    # Por ahora, retorna predicción dummy
    print("⚠️  Predicción dummy - requiere modelo entrenado")
    return (3, 4), 'right'


def print_game_state(
    table_tiles: List[Tuple[int, int]], 
    player_tiles: List[Tuple[int, int]], 
    left_end: Optional[int], 
    right_end: Optional[int]
):
    """Imprimir el estado actual del juego de forma legible"""
    print("🎲 Estado actual del juego:")
    print(f"   Mesa: {table_tiles}")
    print(f"   Jugador: {player_tiles}")
    print(f"   Extremos: {left_end} - {right_end}")


def main():
    """Función principal para demostración"""
    print("🎯 Demo de inferencia de dominó")
    print()
    
    # Estado de ejemplo
    table_tiles = [(1, 2), (2, 3), (3, 5)]
    player_tiles = [(0, 1), (3, 4), (5, 6), (2, 4)]
    left_end = 1
    right_end = 5
    
    # Mostrar estado
    print_game_state(table_tiles, player_tiles, left_end, right_end)
    print()
    
    # Cargar modelo (dummy por ahora)
    model = load_domino_model("checkpoints/domino/")
    
    # Predecir jugada óptima
    optimal_tile, optimal_position = predict_optimal_move(
        model, table_tiles, player_tiles, left_end, right_end
    )
    
    print("🎯 Predicción:")
    print(f"   Ficha óptima: {optimal_tile}")
    print(f"   Posición: {optimal_position}")


if __name__ == "__main__":
    main()
