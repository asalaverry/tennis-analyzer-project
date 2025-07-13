# Tennis Match Analyzer

## Descripción del Proyecto

**Tennis Match Analyzer** es una aplicación de escritorio diseñada para el análisis detallado de partidos de tenis a partir de videos. Desarrollada en Python, esta herramienta combina procesamiento de video avanzado con una interfaz de usuario intuitiva para proporcionar estadísticas valiosas sobre el rendimiento de los jugadores.

El proyecto está diseñado para procesar secuencias de video, detectando los golpes del jugador, el movimiento, y el pique de la pelota. Luego se presenta un resumen estadístico y visual del desempeño, facilitando el análisis táctico y la revisión post-partido. 

## Características del proyecto

* **Análisis Automático de Golpes:** La idea del proyecto, es que a partir del seguimiento de la pelota de tenis y la deteccion de los jugadores y cancha, podemos obtener muchos datos interesantes, como el tipo de golpe realizado, la direccion (Paralelo o cruzado) y si fue punto o no.  
Con esta informacion, se pueden extraer una gran variedad de estadisticas, que pueden ser muy utiles para analizar los patrones de juego de un tenista, y buscar los puntos debiles y fuertes de un rival o propios.  
_Ahora mismo el proyecto se encuentra en desarrollo, y muchas de estas funciones aun no estan (Ver lista DONE/TO DO)_  

* **Representación Visual de Datos:** Generación de gráficos informativos (diagramas de tarta y gráficos de barras) utilizando `Matplotlib` para una comprensión rápida de las estadísticas del partido.


## Tecnologías Utilizadas

* **Python:** Lenguaje de programación principal.
* **CustomTkinter:** Framework para la construcción de la interfaz gráfica (GUI) moderna.
* **YOLO:** Para la detección de objetos y el seguimiento en el video.
* **Numpy:** Para operaciones numéricas eficientes y manejo de datos.
* **Pandas:** Para el análisis y manipulación de datos estadísticos.
* **Matplotlib:** Para la generación de gráficos estadísticos (PNGs).
* **Pillow (PIL):** Utilizado para el manejo de imágenes, incluyendo la carga de gráficos y el establecimiento del icono de la aplicación.
* **OpenCV (cv2):** (Si se implementó el procesamiento de video) Para la lectura, procesamiento y análisis del video.

## Creditos
* **OpenCV (cv2):** (Si se implementó el procesamiento de video) Para la lectura, procesamiento y análisis del video.