import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from detection import center_of_box

def find_strokes_indices(player_1_boxes, ball_positions):
    """
    Detecta frames de golpes usando la trayectoria de la pelota (posición, velocidad, aceleración)
    y la proximidad a la caja del jugador.
    """
    ball_x, ball_y = ball_positions[:, 0], ball_positions[:, 1]

    # Suavizado de la trayectoria de la pelota 
    smooth_x = signal.savgol_filter(ball_x, window_length=7, polyorder=2) # Aumentado window_length para mayor suavizado DFLT: 5
    smooth_y = signal.savgol_filter(ball_y, window_length=7, polyorder=2)

    # Interpolación de la posición de la pelota
    x_coords_for_interp = np.arange(0, len(smooth_y))
    # Eliminar NaN para la interpolación
    valid_indices = ~np.isnan(smooth_y) & ~np.isnan(smooth_x)
    
    x_valid = x_coords_for_interp[valid_indices]
    y_valid = smooth_y[valid_indices]
    x_valid_ball = smooth_x[valid_indices]
    
    '''
    # Manejar caso de no suficientes puntos para interpolar
    if len(x_valid) < 2:
        return [] # No hay suficientes datos para detectar golpes
    '''
        
    ball_f_y = interp1d(x_valid, y_valid, kind='cubic', fill_value="extrapolate")
    ball_f_x = interp1d(x_valid, x_valid_ball, kind='cubic', fill_value="extrapolate")
    
    xnew = np.linspace(0, len(ball_y), num=len(ball_y), endpoint=True)

    # Coordenadas interpoladas de la pelota para todos los frames
    interpolated_ball_x = ball_f_x(xnew)
    interpolated_ball_y = ball_f_y(xnew)
    
    # Calcular velocidad y aceleración de la pelota (usando diferencias)
    # np.diff da la diferencia entre elementos consecutivos, lo que aproxima la velocidad.
    # Para la aceleración, tomamos la diferencia de la velocidad.
    
    # Velocidad (delta_pos / delta_frame_index, asumiendo delta_frame_index=1)
    # Añadimos un valor al principio para mantener la misma longitud que los frames originales
    vel_x = np.diff(interpolated_ball_x, prepend=interpolated_ball_x[0])
    vel_y = np.diff(interpolated_ball_y, prepend=interpolated_ball_y[0])
    
    # Magnitud de la velocidad (rapidez)
    speed = np.sqrt(vel_x**2 + vel_y**2)
    
    # Aceleración
    accel_x = np.diff(vel_x, prepend=vel_x[0])
    accel_y = np.diff(vel_y, prepend=vel_y[0])
    
    # Magnitud de la aceleración
    acceleration_magnitude = np.sqrt(accel_x**2 + accel_y**2)

    # --- Criterios para la detección de golpes ---

    # 1. Picos en la magnitud de la aceleración (cambios abruptos en velocidad/dirección)
    # Ajusta los parámetros 'height' y 'distance' según la respuesta típica de tus videos
    # height: umbral mínimo para considerar un pico de aceleración
    # distance: mínima distancia entre picos de aceleración para evitar múltiples detecciones del mismo golpe
    ACCEL_PEAK_HEIGHT = np.percentile(acceleration_magnitude[~np.isnan(acceleration_magnitude)], 80) # DEFAULT: el 90 percentil de las aceleraciones
    ACCEL_PEAK_DISTANCE = 6 # DEFAULT: Mínimo 10 frames entre picos fuertes de aceleración

    stroke_candidate_indices, _ = find_peaks(acceleration_magnitude, height=ACCEL_PEAK_HEIGHT, distance=ACCEL_PEAK_DISTANCE)
    
    # 2. Distancia de la pelota al jugador
    dists = []
    for i in range(len(player_1_boxes)):
        player_box = player_1_boxes[i]
        if player_box is not None and not np.isnan(player_box).any():
            player_center = center_of_box(player_box)
            ball_pos = np.array([interpolated_ball_x[i], interpolated_ball_y[i]])
            
            # Asegurar que ball_pos no sea NaN
            if not np.isnan(ball_pos).any():
                box_dist = np.linalg.norm(player_center - ball_pos)
                dists.append(box_dist)
            else:
                dists.append(np.nan)
        else:
            dists.append(np.nan)
            
    dists = np.array(dists)

    strokes_1_indices = []
    
    # Umbral de proximidad de la pelota al jugador. Valor típico podría ser 100-200 píxeles, o basado en la altura de la caja del jugador.
    PLAYER_PROXIMITY_THRESHOLD = 220 # DEFAULT: 150

    # Iterar sobre los candidatos a golpe (donde hubo alta aceleración)
    for candidate_idx in stroke_candidate_indices:
        # Asegurarse de que haya datos válidos en este índice
        if candidate_idx < len(dists) and not np.isnan(dists[candidate_idx]):
            # Obtener la altura de la caja del jugador en el frame del golpe
            player_box_at_stroke = player_1_boxes[candidate_idx]
            if player_box_at_stroke is not None and not np.isnan(player_box_at_stroke).any():
                player_box_height = max(player_box_at_stroke[3] - player_box_at_stroke[1], 100) # Mínimo de altura para evitar división por cero o cajas muy pequeñas

                # Umbral de distancia dinámico
                # Puedes ajustar la proporción (e.g., 0.8) o usar un valor fijo como PLAYER_PROXIMITY_THRESHOLD
                current_proximity_threshold = min(PLAYER_PROXIMITY_THRESHOLD, player_box_height * 0.8)
                
                if dists[candidate_idx] < current_proximity_threshold:
                    strokes_1_indices.append(candidate_idx)

    # --- Filtrado final para eliminar golpes muy cercanos ---
    # Esto ya lo tenías y es bueno mantenerlo para eliminar duplicados o detecciones ruidosas.
    while True:
        diffs = np.diff(strokes_1_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40: # Si dos golpes están a menos de 40 frames de distancia
                # Conservar el que tuvo menor distancia al jugador (más probable que sea el golpe real)
                # Asegurarse de que los índices sean válidos para acceder a dists
                if strokes_1_indices[i] < len(dists) and strokes_1_indices[i+1] < len(dists):
                    dist1 = dists[strokes_1_indices[i]]
                    dist2 = dists[strokes_1_indices[i+1]]
                    # Si alguna distancia es NaN, asumimos que no es un buen candidato.
                    # np.nanargmin devuelve el índice del mínimo, ignorando NaNs
                    if np.isnan(dist1) and np.isnan(dist2):
                        # Ambos son NaN, eliminamos el segundo por defecto o el que tenga mayor índice
                        to_del.append(i + 1)
                    elif np.isnan(dist1): # Si dist1 es NaN, preferimos dist2
                        to_del.append(i)
                    elif np.isnan(dist2): # Si dist2 es NaN, preferimos dist1
                        to_del.append(i+1)
                    else: # Ambos son números válidos
                        if dist1 > dist2:
                            to_del.append(i) # Elimina el primero si es mayor (menos cercano)
                        else:
                            to_del.append(i + 1) # Elimina el segundo si es mayor
                else: # Si los índices son inválidos para dists, marcamos para eliminar el segundo por seguridad
                    to_del.append(i + 1)

        # Eliminar duplicados de to_del y ordenar de mayor a menor para eliminar sin afectar índices
        to_del = sorted(list(set(to_del)), reverse=True)
        for idx in to_del:
            if idx < len(strokes_1_indices): # Asegurarse que el índice existe antes de borrar
                strokes_1_indices = np.delete(strokes_1_indices, idx).tolist() # Convertir a lista para manejo de np.diff en próxima iteración

        if len(to_del) == 0:
            break
    
    # Asegurar que strokes_1_indices es una lista para consistencia
    if isinstance(strokes_1_indices, np.ndarray):
        strokes_1_indices = strokes_1_indices.tolist()

    return strokes_1_indices


def detect_type_of_stroke (player_1_boxes, ball_positions, stroke_1_indices):
    strokes_types = {}

    ball_positions_np = np.asarray(ball_positions) #convierto a numpy array

    # Umbral para Smash/Saque: Ajusta este valor según tus videos.
    # Un valor fijo (ej: 50 píxeles por encima de la cabeza) puede ser más estable si la perspectiva es fija.
    SMASH_SERVE_HEIGHT_OFFSET = 20 # Píxeles por encima de la cabeza del jugador para considerarlo un smash/saque.

    # Umbral de tolerancia horizontal para Smash/Saque:
    # Si la pelota está más lejos de este valor (en X) del centro del jugador, NO es un smash.
    # Por ejemplo, 0.2 significa que la pelota debe estar dentro del 20% del ancho del jugador (aproximado).
    SMASH_HORIZONTAL_TOLERANCE_FACTOR = 1

    for frame_idx in stroke_1_indices:
        # 1. Obtener la caja del jugador y la posición de la pelota para el frame del golpe
        player_box = player_1_boxes[frame_idx] if frame_idx < len(player_1_boxes) else None
        ball_pos = ball_positions_np[frame_idx] if frame_idx < len(ball_positions_np) else None

        # Verificar si hay datos válidos para este frame
        if player_box is None or ball_pos is None or np.isnan(ball_pos).any():
            strokes_types[frame_idx] = "Desconocido (sin datos)"
            continue # Pasa al siguiente golpe

        # 2. Calcular el centro del jugador
        player_center_x, player_center_y = center_of_box(player_box)

        # 3. Detectar Smash/Saque
        # La parte superior de la caja (y_min) es la altura de la cabeza del jugador
        player_head_y = player_box[1] # y_min de la box del jugador

        # Calcular el ancho aproximado del jugador para la tolerancia horizontal
        player_width = player_box[2] - player_box[0] # x_max - x_min

        # Calcular la distancia horizontal de la pelota al centro del jugador
        horizontal_dist_to_player_center = abs(ball_pos[0] - player_center_x)

        # Si la pelota está significativamente por encima de la cabeza del jugador y pasa la tolerancia horizontal
        if (ball_pos[1] < (player_head_y - SMASH_SERVE_HEIGHT_OFFSET)) and (horizontal_dist_to_player_center < (player_width * SMASH_HORIZONTAL_TOLERANCE_FACTOR)):
            strokes_types[frame_idx] = "Smash/Saque"
        else:
            # 4. Detectar Drive o Revés
            # Comparamos la X de la pelota con la X del centro del jugador.
            if ball_pos[0] > player_center_x:
                strokes_types[frame_idx] = "Drive"
            else:
                strokes_types[frame_idx] = "Reves"

    return strokes_types