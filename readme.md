# Tennis Match Analyzer

## Descripción del Proyecto

**Tennis Match Analyzer** es una aplicación de escritorio diseñada para el análisis detallado de partidos de tenis a partir de videos. Desarrollada en Python, esta herramienta combina procesamiento de video avanzado con una interfaz de usuario intuitiva para proporcionar estadísticas valiosas sobre el rendimiento de los jugadores.

El proyecto está diseñado para procesar secuencias de video, detectando los golpes del jugador, el movimiento, y el pique de la pelota. Luego se presenta un resumen estadístico y visual del desempeño, facilitando el análisis táctico y la revisión post-partido. 

_Este proyecto fue hecho como entrega para un trabajo universitario. No se si continuare trabajando en el, pero todo aporte es bienvenido_

## Características del Proyecto

* **Análisis Automático de Golpes:** La idea del proyecto, es que a partir del seguimiento de la pelota de tenis y la deteccion de los jugadores y cancha, podemos obtener muchos datos interesantes, como el tipo de golpe realizado, la direccion (Paralelo o cruzado) y si fue punto o no.  
Con esta informacion, se pueden extraer una gran variedad de estadisticas, que pueden ser muy utiles para analizar los patrones de juego de un tenista, y buscar los puntos debiles y fuertes de un rival o propios.  
_Ahora mismo el proyecto se encuentra en desarrollo, y muchas de estas funciones aun no estan (Ver lista DONE/TO DO)_  

* **Representación Visual de Datos:** Generación de gráficos informativos (diagramas de tarta y gráficos de barras) utilizando `Matplotlib` para una comprensión rápida de las estadísticas del partido.


## Tecnologías Utilizadas

* **Python:** Lenguaje de programación principal.
* **CustomTkinter:** Framework para la construcción de la interfaz gráfica (GUI).
* **YOLO:** Para la detección de objetos y el seguimiento en el video.
* **Numpy:** Para operaciones numéricas eficientes y manejo de datos.
* **Pandas:** Para el análisis y manipulación de datos estadísticos.
* **Matplotlib:** Para la generación de gráficos estadísticos.
* **Pillow (PIL):** Utilizado para el manejo de imágenes.
* **OpenCV (cv2):** Para la lectura, procesamiento y análisis del video.

## Status del Proyecto

**DONE**  
- Deteccion de posicion de jugador
- Deteccion de cancha
- Deteccion de pelota y piques
- Deteccion de golpe y tipo de golpe (Drive, reves, smash/saque)
- Interfaz de usuario y estadisticas

**TO DO**
- Deteccion de voleas
- Deteccion de direccion de golpe
- Deteccion de si es punto o no (Y razon de "no punto")
- Mejorar algoritmo de deteccion de golpe y tipo de golpe
- Mejorar deteccion de pique 
- Deteccion de momentos de partido (Cuando la pelota esta en juego y cuando no)

  
![example](https://github.com/user-attachments/assets/75fff057-ce94-4410-9f3c-0d45936c0f1e)

## Repositorios que Fueron de Ayuda
* **[ArtLabss/tennis-tracking](https://github.com/ArtLabss/tennis-tracking)**
* **[yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet)**
* **[avivcaspi/TennisProject](https://github.com/avivcaspi/TennisProject)**
