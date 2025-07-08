import argparse
import queue
import pandas as pd 
from collections import Counter
import imutils
import os
from PIL import Image, ImageDraw
import cv2 
import numpy as np
import torch
import sys
import time
import matplotlib.pyplot as plt

from sktime.datatypes._panel._convert import from_2d_array_to_nested
from detector_cancha import CourtDetector
from Models.tracknet import trackNet
from TrackPlayers.trackplayers import *
from utils import get_video_properties, get_dtype
from detection import *
from pickle import load
import deteccion_golpes

TEMP_FOLDER = "temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)


def _run_analysis(input_path):
  print("Iniciando analisis...")
  s= time.time()

  input_video_path = input_path
  output_video_path = os.path.join("temp", "output.mp4")
  bounce = 1


  n_classes = 256
  save_weights_path = 'WeightsTracknet/model.h5'
  yolo_classes = 'Yolov3/yolov3.txt'
  yolo_weights = 'Yolov3/yolov3.weights'
  yolo_config = 'Yolov3/yolov3.cfg'

  if output_video_path == "":
      # El output se da vacio
      output_video_path = input_video_path.split('.')[0] + "VideoOutput/video_output.mp4"


  # Cargar modelo tracknet
  width, height = 640, 360 # ancho y alto en el que trabaja TrackNet
  modelFN = trackNet 
  m = modelFN(n_classes, input_height=height, input_width=width)
  m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
  m.load_weights(save_weights_path)

  # carga yolov3 para deteccion jugadores
  LABELS = open(yolo_classes).read().strip().split("\n")
  net = cv2.dnn.readNet(yolo_weights, yolo_config)

  # inicializa los detectores
  court_detector = CourtDetector()
  dtype = get_dtype()
  detection_model = DetectionModel(dtype=dtype)

  # tomar fps&video tamaño
  video = cv2.VideoCapture(input_video_path)
  fps = int(video.get(cv2.CAP_PROP_FPS))
  print('fps : {}'.format(fps))
  output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

  # guarda las imagenes de prediccion como video
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

  # obtener propiedades del video
  fps, length, v_width, v_height = get_video_properties(video)

  frames = []
  frame_i = 0

  print('PARTE 1: Detectando cancha y jugadores...')
  while True:
    ret, frame = video.read()
    frame_i += 1

    if ret:
      if frame_i == 1:
        lines = court_detector.detect(frame)
      else: # Luego trackearlos
        lines = court_detector.track_court(frame) #Sigue la cancha basandose en la deteccion inicial
      
      #Detacta jugadores en cada frame y guarda sus boxes
      detection_model.detect_player_1(frame, court_detector)
      detection_model.detect_top_persons(frame, court_detector, frame_i)
      
      #Dibuja la cancha
      for i in range(0, len(lines), 4):
        x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
        cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 3)

      #cada frame se guarda en frames --> almacenado en RAM   
      new_frame = cv2.resize(frame, (v_width, v_height))
      frames.append(new_frame)
    else:
      break
  video.release()
  print("Parte 1 finalizada")

  detection_model.find_player_2_box() #Separa la caja del jugador de las demas personas en el frame

  # segunda parte
  player1_boxes = detection_model.player_1_boxes
  player2_boxes = detection_model.player_2_boxes

  last = time.time()
  currentFrame = 0
  t = []
  coords = []
  frames_w_ball = []

  # Almacena las ultimas 8 coordenadas de la pelota para dibujar la trayectoria
  q = queue.deque()
  for i in range(0, 8):
      q.appendleft(None)


  for img in frames:
      currentFrame += 1
      print('Tracking the ball: {}'.format(round( ((currentFrame) / total) * 100, 2)))

      # img es el frame que TrackNet predecira la posicion
      # como necesitamos cambiar el tamaño y tipo de img, lo copiamos a output_img
      img_mod = img

      # cambiar el tamaño de la imagen a (640, 360)
      img_mod = cv2.resize(img_mod, (width, height))
      # convierto img a float32
      img_mod = img_mod.astype(np.float32)

      #Reordeno para TrackNet. Espera primero los canales
      X = np.rollaxis(img_mod, 2, 0)
      # ejecuto modelo de tracknet para predecir mapa de calor donde podria estar la pelota
      pr = m.predict(np.array([X]))[0]
      pr = pr.reshape((height, width, n_classes)).argmax(axis=2) # rearmamos la imagen para tracknet
      pr = pr.astype(np.uint8) #cv2 image espera un numpy.unit8

      # redimensionar el mapa de calor al tamaño original
      heatmap = cv2.resize(pr, (output_width, output_height))
      ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY) # se convierte el mapa de calor a una imagen binaria

      # busca circulos en el mapa de calor, que son donde estaria la pelota
      circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                maxRadius=7)

      # chequeo si se detecto la pelota
      if circles is not None:
          # Si solo se detecto un circulo, se guarda la coordenada
          if len(circles) == 1:

              x = int(circles[0][0][0])
              y = int(circles[0][0][1])

              coords.append([x,y])
              t.append(time.time()-last)

              # se guarda en la cola q para trayectoria
              q.appendleft([x, y])
              q.pop()

          else: #Si se detectaron multiples
              coords.append(None)
              t.append(time.time()-last)

              q.appendleft(None)
              q.pop()
      else: # Si no se detecto la pelota
          coords.append(None)
          t.append(time.time()-last)

          q.appendleft(None)
          q.pop()

      #Dibujo las boxes de los jugadores
      img = mark_player_box(img, player1_boxes, currentFrame -1)
      img = mark_player_box(img, player2_boxes, currentFrame -1)
      
      PIL_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      PIL_image = Image.fromarray(PIL_image)

      #Se itera sobre los ultimos 8 frames y dibuja la trayectoria de la pelota guardada en la cola q
      for i in range(0, 8):
          if q[i] is not None:
              draw_x = q[i][0]
              draw_y = q[i][1]
              bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
              draw = ImageDraw.Draw(PIL_image)
              draw.ellipse(bbox, outline='yellow')
              del draw

      # Se escribe en el formato de salida
      img = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
      #output_video.write(img)
      frames_w_ball.append(img)


  print("Parte 2 finalizada")


  for _ in range(3):        #Para limpiar y suavizar la trayectoria de coords (pelota) eliminando valores atípicos (outliers)
    x, y = diff_xy(coords)
    remove_outliers(x, y, coords)

  # interpolar coordenadas
  coords = interpolation(coords)


  #Parte 3: Deteccion golpes
  #Preparamos las coordenadas para el modelo de deteccion de golpes
  ball_positions_array = np.full((len(coords), 2), np.nan)

  for i, coord in enumerate(coords):
      if coord is not None:
          ball_positions_array[i] = coord 

  p1  = deteccion_golpes.find_strokes_indices(player1_boxes, ball_positions_array)
  player1_stroke_types = deteccion_golpes.detect_type_of_stroke(player1_boxes, ball_positions_array, p1)
  currentFrame = 0

  for img in frames_w_ball:
      print('Detectando golpes: {}'.format(round( ((currentFrame +1) / total) * 100, 2)))
      img_wStroke = img

      #Se marca cuando detecta un golpe
      for i in range(-5, 10):
        if currentFrame + i in p1:
            cv2.putText(img_wStroke, 'Stroke detected', 
                        (int(player1_boxes[currentFrame][0]) - 10, int(player1_boxes[currentFrame][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)
            
            stroke_type_text = player1_stroke_types.get(currentFrame + i, "Desconocido") # Default "Desconocido"

            cv2.putText(img_wStroke, stroke_type_text,
                        (int(player1_boxes[currentFrame][0]) - 10, int(player1_boxes[currentFrame][1]) + 20), # 20 píxeles más abajo
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)

        '''if currentFrame + i in p2: #Si se quiere marcar los golpes del jugador 2
            cv2.putText(img_wStroke, 'Stroke detected',
                        (int(f_x(currentFrame)) - 30, int(f_y(currentFrame)) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)''' 

      output_video.write(img_wStroke)

      # avanzo al proximo frame y reinicio el bucle
      currentFrame += 1

  # Esta todo hecho, libera video de salida
  output_video.release()
  print("Parte 3 finalizada")


  # Parte 4: Deteccion de piques
  if bounce == 1:

    # calcula velocidad (Adentro del if bounce == 1, si luego lo necesito para otra funcion, sacarlo!)
    Vx = []
    Vy = []
    V = []
    #frames = [*range(len(coords))]

    for i in range(len(coords)-1):
      p1 = coords[i]
      p2 = coords[i+1]
      t1 = t[i]
      t2 = t[i+1]
      x = (p1[0]-p2[0])/(t1-t2)
      y = (p1[1]-p2[1])/(t1-t2)
      Vx.append(x)
      Vy.append(y)

    for i in range(len(Vx)):
      vx = Vx[i]
      vy = Vy[i]
      v = (vx**2+vy**2)**0.5
      V.append(v)

    xy = coords[:] #Hasta aca el calculo de la velocidad

    # Predecir rebotes
    test_df = pd.DataFrame({'x': [coord[0] for coord in xy[:-1]], 'y':[coord[1] for coord in xy[:-1]], 'V': V}) #Construyo DataFrame de pandas con las coordenadas y la velocidad de la pelota

    # para cada coordenada, guardo las 20 anteriores. --> secuencia de datos para que el modelo de aprendizaje automático puede analizar para identificar patrones de botes.
    for i in range(20, 0, -1): 
      test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
    for i in range(20, 0, -1): 
      test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
    for i in range(20, 0, -1): 
      test_df[f'lagV_{i}'] = test_df['V'].shift(i, fill_value=0)

    test_df.drop(['x', 'y', 'V'], axis=1, inplace=True)

    Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
          'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
          'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
          'lagX_2', 'lagX_1']]
    Xs = from_2d_array_to_nested(Xs.to_numpy()) #convierto en formato anidado de sktime

    Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
          'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
          'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
          'lagY_3', 'lagY_2', 'lagY_1']]
    Ys = from_2d_array_to_nested(Ys.to_numpy())

    Vs = test_df[['lagV_20', 'lagV_19', 'lagV_18',
          'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
          'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
          'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
    Vs = from_2d_array_to_nested(Vs.to_numpy())

    X = pd.concat([Xs, Ys, Vs], axis=1)

    # Cargo modelo para determinar si es pique o no.
    clf = load(open('clf.pkl', 'rb'))

    predicted = clf.predict(X) #El clasificador predice, para cada secuencia de 20 frames, si hay un pique.
    idx = list(np.where(predicted == 1)[0])
    idx = np.array(idx) - 10 #Cmpensa retardo en el modelo de predicción
    

    video = cv2.VideoCapture(output_video_path)
    output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print(fps)
    print(length)

    output_video = cv2.VideoWriter('VideoOutput/video_output_wBounce.mp4', fourcc, fps, (output_width, output_height))
    i = 0
    while True: #iterar sobre los frames del video, marcando los piques
      ret, frame = video.read()
      if ret:
        # if coords[i] is not None:
        if i in idx:
          center_coordinates = int(xy[i][0]), int(xy[i][1])
          radius = 2
          color = (255, 0, 0)
          thickness = -1
          cv2.circle(frame, center_coordinates, 10, color, thickness)

        i += 1
        output_video.write(frame)
      else:
        break

    video.release()
    output_video.release()

  print("Finalizado con exito!")
  print(f'Tiempo de ejecucion total : {time.time() - s} segundos')
  
  return player1_stroke_types


def generar_estadisticas(strokes_counts, output_dir="temp"): #Genera los gráficos y estadísticas para la UI basados en los conteos de golpes
    
    conteo = Counter(strokes_counts.values())
    
    drives = conteo.get('Drive', 0)
    backhands = conteo.get('Reves', 0)
    smash_saque = conteo.get('Smash/Saque', 0)  
    
    total_drives_backhands = drives + backhands

    # Solo generamos este gráfico si hay datos
    drives_reveses_chart_path = None
    if total_drives_backhands > 0:
        labels_pie1 = ['Drives', 'Reveses']
        sizes_pie1 = [drives, backhands]
        colors_pie1 = ['#4CAF50', '#2196F3'] # Verde para drives, Azul para reveses
        explode_pie1 = (0.05, 0) 

        plt.figure(figsize=(6, 6)) 
        plt.pie(sizes_pie1, explode=explode_pie1, labels=labels_pie1, colors=colors_pie1, autopct='%1.1f%%',
                shadow=True, startangle=140, textprops={'color': 'white', 'fontname': 'Arial', 'fontweight': 'bold'})
        plt.axis('equal') 
        plt.title('Cantidad de Drives vs. Reveses', color='white', fontname='Arial', fontweight='bold', fontsize=16)
        
        drives_reveses_chart_path = os.path.join(output_dir, "drives_vs_backhands_pie_chart.png")
        plt.savefig(drives_reveses_chart_path, bbox_inches='tight', transparent=True) 
        plt.close()
        print(f"Gráfico de tarta (Drives vs Reveses) guardado en: {drives_reveses_chart_path}")
    else:
        print("No hay suficientes datos para el gráfico de Drives vs Reveses.")


    # --- 2. Gráfico de barras: Errores no Forzados vs Tiros Ganadores ---
    # Datos inventados para probar UI. Aun no se lograron generar estos datos!
    unforced_errors = 1
    winners = 0 # Valor original

    unforced_winners_chart_path = None
    
    labels_bar = ['Errores No Forzados', 'Tiros Ganadores']
    
    # Preparar los conteos para plotear: si winners es 0, lo ponemos en un valor pequeño (0.05) para que se vea la barra.
    # El 0.05 es un valor arbitrario, puedes ajustarlo si quieres un "bordesito" más o menos visible.
    plot_winners = winners if winners > 0 else 0.05 
    counts_bar_for_plot = [unforced_errors, plot_winners] # Usamos estos conteos para plt.bar

    colors_bar = ['#FFC107', '#00BCD4'] 

    plt.figure(figsize=(6, 4)) 
    bars = plt.bar(labels_bar, counts_bar_for_plot, color=colors_bar) # Guardamos la referencia a las barras

    # --- MODIFICACIONES PARA EL EJE Y ---
    plt.ylim(0, 5) # Establecer el rango del eje Y de 0 a 5
    # Establecer los ticks discretos del eje Y en 0, 1, 2, 3, 4, 5
    plt.yticks(np.arange(0, 6, 1), color='white', fontname='Arial') 

    plt.ylabel('Cantidad', color='white', fontname='Arial')
    plt.title('Errores No Forzados vs. Tiros Ganadores', color='white', fontname='Arial', fontweight='bold', fontsize=16)
    plt.xticks(color='white', fontname='Arial') 
    plt.grid(axis='y', linestyle='--', alpha=0.7) 
    
    ax = plt.gca() 
    ax.set_facecolor('none') 
    plt.gcf().patch.set_alpha(0.0) 

    # --- Añadir etiquetas numéricas encima de cada barra ---
    # Usaremos los valores originales (1 y 0) para estas etiquetas.
    original_bar_counts = [unforced_errors, winners]
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        # Posicionar el texto ligeramente por encima de la barra.
        # Si la barra es 0.05 (bordesito), el texto "0" estará justo encima de ella.
        plt.text(bar.get_x() + bar.get_width()/2, # Centro X de la barra
                 yval + 0.1, # Un poco por encima del tope de la barra (ajusta 0.1 si es necesario)
                 str(original_bar_counts[i]), # El texto es el valor original (0 o 1)
                 ha='center', va='bottom', color='white', fontsize=14, fontweight='bold')


    unforced_winners_chart_path = os.path.join(output_dir, "unforced_vs_winners_bar_chart.png")
    plt.savefig(unforced_winners_chart_path, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Gráfico de barras (Errores vs Ganadores) guardado en: {unforced_winners_chart_path}")


    # --- 3. Texto: Voleas jugadas ---
    #Datos inventados para probar UI. Aun no se lograron generar estos datos!
    volleys_played = 0 # Puedes poner cualquier número aquí

    # --- 4. Gráfico de Errores por Tipo de Golpe ---
    # Datos inventados para probar UI. Aun no se lograron generar estos datos!
    drive_errors = 1
    backhand_errors = 0 # Este es el valor que queremos que se vea como franja mínima

    errors_by_stroke_chart_path = None
    
    labels_pie2 = ['Errores con Drive', 'Errores con Reves']
    original_error_counts = [drive_errors, backhand_errors] # Guardamos los conteos originales

    # --- CAMBIOS AQUÍ para el gráfico de Errores por Tipo de Golpe ---
    # 1. Preparar los tamaños para el ploteo: si el original es 0, usamos 0.001 para que se dibuje.
    sizes_pie2_for_plot = [max(val, 0.002) for val in original_error_counts] 
    
    # Si la suma de los valores originales es 0, no generamos el gráfico para evitar un pie chart vacío.
    # (aunque con 0.001 ambos, sumarán 0.002, es más claro chequear el original)
    if sum(original_error_counts) == 0:
        print("No hay errores con drive o reves para generar el gráfico de errores por golpe.")
        errors_by_stroke_chart_path = None
    else:
        colors_pie2 = ["#E2714F", '#673AB7'] 
        explode_pie2 = (0.05, 0) 

        plt.figure(figsize=(6, 6))
        
        # 2. Crear las etiquetas que se mostrarán en el gráfico, incluyendo las cantidades originales.
        display_labels = []
        for i, label_name in enumerate(labels_pie2):
            display_labels.append(f"{label_name}: {original_error_counts[i]}") # Formato: "Nombre\n(Cantidad)"

        plt.pie(sizes_pie2_for_plot, # Usamos los tamaños modificados para el ploteo
                explode=explode_pie2, 
                labels=display_labels, # <-- Usamos nuestras etiquetas personalizadas
                colors=colors_pie2, 
                autopct='', # <-- Deshabilitamos autopct para que no muestre el porcentaje
                shadow=True, 
                startangle=140, 
                textprops={'color': 'white', 'fontname': 'Arial', 'fontweight': 'bold'})
        
        plt.axis('equal')
        plt.title('Errores Cometidos por Tipo de Golpe', color='white', fontname='Arial', fontweight='bold', fontsize=16)
        
        errors_by_stroke_chart_path = os.path.join(output_dir, "errors_by_stroke_pie_chart.png")
        plt.savefig(errors_by_stroke_chart_path, bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Gráfico de tarta (Errores por Golpe) guardado en: {errors_by_stroke_chart_path}")




    # --- Retornar las rutas de las imágenes y los datos textuales ---
    return {
        'drives_reveses_chart': drives_reveses_chart_path,
        'unforced_winners_chart': unforced_winners_chart_path,
        'volleys_played': volleys_played, # El dato en sí, no una ruta de imagen
        'errors_by_stroke_chart': errors_by_stroke_chart_path
    }

def main():
    
    # Estos valores serían temporales, de prueba
    input_path = "VideoInput/video_input11.mp4"

    _run_analysis(input_path)
