import os
import customtkinter as ctk
from tkinter import filedialog
import threading
import cv2
from PIL import Image, ImageTk
from main import _run_analysis, generar_estadisticas
import time

class TennisAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Analizador de Partidos de Tenis")
        self.set_app_icon("logo.ico")
        self.geometry("1200x600")

        self.file_path = None
        self.output_path = None
        self.cap = None
        self.playing = False

        self.analysis_overlay = None
        self.analysis_label = None
        self.time_label = None
        self.animation_step = 0
        self.animation_id = None # Para cancelar la animación
        self.start_time = None # Para el contador de tiempo
        self.time_update_id = None # Para cancelar la actualización de tiempo

        self.stats_frame = None # Frame para contener todas las estadísticas
        self.drives_reveses_chart_label = None
        self.unforced_winners_chart_label = None
        self.volleys_label = None
        self.errors_by_stroke_chart_label = None
        

        self.create_widgets()

    def create_widgets(self):
        self.main_scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.main_scroll_frame.pack(pady=0, padx=0, fill="both", expand=True)
        # Configurar la única columna de main_scroll_frame para que el contenido se expanda horizontalmente
        self.main_scroll_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(self.main_scroll_frame, text="Analizador Partidos de Tenis", font=('SF Pro Display', 30, 'bold'))
        self.title_label.pack(pady=10)

        self.video_frame = ctk.CTkFrame(self.main_scroll_frame, width=640, height=360, border_width=2, border_color="#1C5AA0") 
        self.video_frame.pack(pady= (14,3), expand=False) 

        # ETIQUETA DONDE SE MOSTRARÁ EL VIDEO
        # MODIFICAR: Añadir padx y pady para dejar espacio al borde del frame
        self.video_label = ctk.CTkLabel(self.video_frame, text="", width=640, height=360)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=2, pady=2) # <--- AÑADE ESTO (el valor debe ser igual o mayor al border_width)
        
        # Configurar las filas y columnas para que la etiqueta se expanda
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        # BOTÓN DE DESCARGA (inicialmente oculto)
        download_img = ctk.CTkImage(light_image=Image.open("download_icon.png"), size=(30, 30))
        self.download_button = ctk.CTkLabel(self.video_frame, text="", image=download_img, width=30, height=30, bg_color="transparent", fg_color="transparent", cursor="hand2")
        self.download_button.place(relx=0.98, rely=0.95, anchor="se")
        self.download_button.bind("<Button-1>", lambda e: self.download_output())
        self.download_button.lower()


        # BOTÓN "CARGAR VIDEO" CON ÍCONO
        upload_img = ctk.CTkImage(light_image=Image.open("upload_icon.png"), size=(20, 20))  # <- usa tu propio ícono
        self.upload_button = ctk.CTkButton(self.video_label, text=" Cargar video", image=upload_img, command=self.select_file, font=('SF Pro Display', 14, 'bold'))
        self.upload_button.place(relx=0.5, rely=0.5, anchor="center")


        # CONTROLES DE REPRODUCCIÓN (ocultos al inicio)
        self.player_controls = ctk.CTkFrame(self.main_scroll_frame)
        self.play_button = ctk.CTkButton(self.player_controls, text="Play", command=self.toggle_playback, width=80, font=('SF Pro Display', 14, 'bold'))
        self.play_button.grid(row=0, column=0, padx=10)

        self.slider = ctk.CTkSlider(self.player_controls, from_=0, to=100, command=self.seek_video, width=500)
        self.slider.grid(row=0, column=1, padx=10)

        # CONTROLES DE ANÁLISIS
        self.controls_frame = ctk.CTkFrame(self.main_scroll_frame, width=400, height=200)
        self.controls_frame.pack(pady=30)

        self.start_button = ctk.CTkButton(self.controls_frame, width=220, height=32, text="Iniciar análisis", command=self.start_analysis, fg_color="green", state="disabled", font=('SF Pro Display', 14, 'bold'))
        self.start_button.grid(row=0, column=1, padx=20)

        self.reset_button = ctk.CTkButton(self.controls_frame, width=220, height=32, text="Reiniciar", command=self.reset_ui, fg_color="#8B0000", state="disabled", font=('SF Pro Display', 14, 'bold'))
        self.reset_button.grid(row=0, column=0, padx=20)

        # BOTÓN DE AYUDA
        self.help_button = ctk.CTkButton(self, text="?", width=30, height=30, command=self.show_help)
        self.help_button.place(relx=0.97, rely=0.03, anchor="ne")

        # --- AÑADIR EL OVERLAY Y LOS LABELS DE ESTADO ---
        # El overlay (fondo semitransparente)
        self.analysis_overlay = ctk.CTkFrame(self.video_label, fg_color="#0B315C")
        self.analysis_overlay.place(relx=0.5, rely=0.5, anchor="center", relwidth=1, relheight=1)
        self.analysis_overlay.lower() # Asegurarse de que esté debajo del texto inicialmente (aunque invisible)

        # El texto "Analizando..."
        self.analysis_label = ctk.CTkLabel(self.analysis_overlay, text="Analizando", font=('SF Pro Display', 25, 'bold'), text_color="white")
        self.analysis_label.place(relx=0.5, rely=0.45, anchor="center")

        # El contador de tiempo
        self.time_label = ctk.CTkLabel(self.analysis_overlay, text="00:00", font=('SF Pro Display', 20, 'bold'), text_color="white")
        self.time_label.place(relx=0.5, rely=0.55, anchor="center")



         # --- NUEVA SECCIÓN: Estadísticas (abajo, scrollable) ---
        # --- SECCIÓN DE ESTADÍSTICAS (ABJO, SCROLLABLE) ---
        # Este frame (stats_frame) es el CTkScrollableFrame que contendrá todas las estadísticas
        # Es hijo de self.main_scroll_frame, y se asume que se packea, como lo tienes actualmente.

        self.stats_frame = ctk.CTkFrame(self.main_scroll_frame, fg_color="transparent", border_width=2, border_color="#1C5AA0")
        self.stats_frame.pack(pady=70, padx=10, fill="both", expand=True) 

        # Configurar el grid INTERNO dentro de self.stats_frame para organizar los gráficos y el texto
        # Ahora necesitamos 3 columnas para los gráficos horizontales
        self.stats_frame.grid_columnconfigure(0, weight=1) # Columna para Drives vs Reveses
        self.stats_frame.grid_columnconfigure(1, weight=1) # Columna para Errores vs Ganadores
        self.stats_frame.grid_columnconfigure(2, weight=1) # Columna para Errores por Golpe

        self.stats_title_label = ctk.CTkLabel(self.stats_frame, text="Estadísticas del partido:", 
                                               font=('SF Pro Display', 22, 'bold'), text_color="white")
        # Lo colocamos en la primera fila, abarcando las 3 columnas, y con pady para separar
        self.stats_title_label.grid(row=0, column=0, columnspan=3, pady=40, padx=50, sticky="w") # sticky="w" para alinearlo a la izquierda

        # Gráfico Drives vs Reveses (row=0, column=0)
        self.drives_reveses_chart_label = ctk.CTkLabel(self.stats_frame, text="")
        self.drives_reveses_chart_label.grid(row=1, column=0, pady=10, padx=10, sticky="nsew") # CAMBIO: column=0

        # Gráfico Errores No Forzados vs Tiros Ganadores (row=0, column=1)
        self.unforced_winners_chart_label = ctk.CTkLabel(self.stats_frame, text="")
        self.unforced_winners_chart_label.grid(row=2, column=0, pady=30, padx=10, sticky="nsew") # CAMBIO: row=0, column=1

        # Gráfico Errores por Tipo de Golpe (row=0, column=2)
        self.errors_by_stroke_chart_label = ctk.CTkLabel(self.stats_frame, text="")
        self.errors_by_stroke_chart_label.grid(row=1, column=1, pady=10, padx=10, sticky="nsew") # CAMBIO: row=0, column=2
        
        # Texto Voleas jugadas (debajo de los gráficos, abarcando las 3 columnas)
        self.volleys_label = ctk.CTkLabel(self.stats_frame, text="", font=('SF Pro Display', 18), text_color="white")
        self.volleys_label.grid(row=2, column=1, pady=30, padx=10, sticky="ew") # CAMBIO: row=1 (nueva fila), columnspan=3
        
        # Inicialmente ocultar el frame de estadísticas
        self.hide_statistics() 



        # Inicialmente ocultar el overlay y sus contenidos
        self.hide_analysis_overlay()

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            self.file_path = file_path
            self.output_path = None
            self.upload_button.place_forget()

            self.play_video(file_path)
            self.start_button.configure(state="normal")

    def start_analysis(self):
        self.start_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.player_controls.pack_forget()
        self.show_analysis_overlay()
        threading.Thread(target=self.run_analysis).start()

    def run_analysis(self):
        try:
            strokes_counts = _run_analysis(self.file_path) 

            if strokes_counts: # Asegúrate de que strokes_counts no sea None
                stats_results = generar_estadisticas(strokes_counts)
            else:
                stats_results = None
            
            self.output_path = os.path.join("temp", "output.mp4")
            self.play_video(self.output_path)
            self.reset_button.configure(state="normal")
            self.after(0, lambda: self.download_button.lift())

            if stats_results:
                self.after(0, lambda: self.display_statistics(stats_results)) # Muestra las estadísticas
            else:
                self.after(0, lambda: self.video_label.configure(text="Análisis completado, pero no se generaron estadísticas.", image=None))

            # OCULTAR EL OVERLAY DE ANÁLISIS CUANDO TERMINA
            self.after(0, self.hide_analysis_overlay) # <--- AÑADIR ESTA LÍNEA
            
            # Una vez terminado el análisis, reproduce el video de salida
            self.after(0, lambda: self.play_video(self.output_path))
            self.after(0, lambda: self.reset_button.configure(state="normal"))
        except Exception as e:
            # En caso de error, también ocultar el overlay
            self.after(0, lambda: self.download_button.lower())
            self.after(0, self.hide_analysis_overlay) # <--- AÑADIR ESTA LÍNEA
            self.after(0, lambda: self.video_label.configure(text=f"Error en el análisis: {e}", image=None))
            self.after(0, lambda: self.reset_button.configure(state="normal"))
            print(f"Error durante el análisis: {e}")

    def reset_ui(self):
        if self.cap:
            self.cap.release()
        self.file_path = None
        self.output_path = None
        self.video_label.configure(image="", text="")

        self.upload_button.place(relx=0.5, rely=0.5, anchor="center")
        self.start_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.slider.set(0)
        self.playing = False
        self.player_controls.pack_forget()
        self.hide_analysis_overlay()
        self.hide_statistics()
        self.download_button.lower()

    def show_help(self):
        help_win = ctk.CTkToplevel(self)
        help_win.title("Ayuda")
        help_win.geometry("420x300")
        help_win.resizable(False, False)

        help_win.grab_set()    # Captura los eventos para esta ventana
        help_win.transient(self) 

        label = ctk.CTkLabel(help_win, text="""Se requiere que el video a cargar muestre el partido de tenis desde una vista superior a la cancha. Como las transmisiones de televisión.
                             
📹 Formato admitido: .mp4, .avi
                             
📐 Resolución sugerida: 720p o 1080p
                             
⏱ La duracion del analisis dependera de la duracion del video y la potencia de tu equipo!

Haz clic en 'Cargar video' para seleccionar un archivo.
Luego haz clic en 'Iniciar análisis' para procesarlo.""", justify="left", font=("Arial", 14), wraplength=380)
        label.pack(padx=20, pady=20)


    def show_analysis_overlay(self):
        # Hace visible el overlay y comienza la animación/contador
        self.analysis_overlay.lift() # Lo trae al frente
        self.analysis_overlay.place(relx=0.5, rely=0.5, anchor="center", relwidth=1, relheight=1)
        
        self.analysis_label.configure(text="Analizando") # Resetear el texto
        self.time_label.configure(text="00:00")
        
        self.animation_step = 0
        self.animate_analysis_text()
        
        self.start_time = time.time() # Iniciar el contador
        self.update_analysis_time()

    def hide_analysis_overlay(self):
        # Oculta el overlay y detiene las animaciones
        self.analysis_overlay.place_forget()
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
        if self.time_update_id:
            self.after_cancel(self.time_update_id)
            self.time_update_id = None

    def animate_analysis_text(self):
        dots = ""
        if self.animation_step == 0:
            dots = "."
        elif self.animation_step == 1:
            dots = ".."
        elif self.animation_step == 2:
            dots = "..."
        elif self.animation_step == 3:
            dots = ".."
        elif self.animation_step == 4:
            dots = "."
        
        self.analysis_label.configure(text=f"Analizando{dots}")
        self.animation_step = (self.animation_step + 1) % 5 # Cicla de 0 a 4
        self.animation_id = self.after(600, self.animate_analysis_text) # Cambia cada 300ms

    def update_analysis_time(self):
        if self.start_time is not None:
            elapsed_time = int(time.time() - self.start_time)
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            self.time_label.configure(text=f"{minutes:02d}:{seconds:02d}")
            self.time_update_id = self.after(1000, self.update_analysis_time) # Actualiza cada segundo

    def toggle_playback(self):
        if not self.cap:
            return
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play", font=('SF Pro Display', 14, 'bold'))
        if self.playing:
            self.update_frame()

    def seek_video(self, value):
        if self.cap:
            frame_number = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) * (value / 100))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.update_frame()

    def play_video(self, path):
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.video_label.configure(image=None, text="Error: No se pudo abrir el video.", compound="center", font=('SF Pro Display', 20, 'bold'))
            self.upload_button.place(relx=0.5, rely=0.5, anchor="center") 
            self.start_button.configure(state="disabled")
            return

        self.slider.set(0)
        self.playing = False 
        self.play_button.configure(text="Play")
        self.update_frame()

        # EMPAQUETAR player_controls AHORA (Y SOLO AHORA)
        # Esto lo coloca después de self.video_frame y ANTES de self.controls_frame.
        # Ajusta el pady para que quede más cerca del video.
        self.player_controls.pack(pady=(3,0), before=self.controls_frame) # <--- CAMBIO CRUCIAL AQUÍ

        self.video_label.configure(compound="center", text="") 

    def update_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            frame_height, frame_width = frame.shape[:2]
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()

            if label_width <= 1 or label_height <= 1:
                label_width = 640 # Tamaño que definiste para self.video_frame / self.video_label
                label_height = 360

            # Calcular las relaciones de aspecto
            video_aspect_ratio = frame_width / frame_height
            label_aspect_ratio = label_width / label_height

            # Lógica de redimensionamiento para "cubrir" el espacio
            # Queremos que el video llene completamente el ancho o la altura del label
            # y que la otra dimensión se ajuste proporcionalmente, pudiendo sobresalir.
            if video_aspect_ratio > label_aspect_ratio:
                # El video es más ancho que el label. Ajustar por altura y luego recortar el ancho.
                new_height = label_height
                new_width = int(new_height * video_aspect_ratio)
            else:
                # El video es más alto que el label. Ajustar por ancho y luego recortar el alto.
                new_width = label_width
                new_height = int(new_width / video_aspect_ratio)

            # Redimensionar el frame
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Recortar el frame redimensionado para que se ajuste exactamente al label
            # Calcular los puntos de inicio para el recorte
            start_x = (new_width - label_width) // 2
            start_y = (new_height - label_height) // 2

            cropped_frame = resized_frame[start_y:start_y + label_height, start_x:start_x + label_width]

            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cropped_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            
            # Asegurarse de que el texto de "Cargando video" se borre al cargar el frame
            self.video_label.configure(image=imgtk, text="", compound="center") 

            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total > 0:
                self.slider.set((pos / total) * 100)
            
            if self.playing:
                self.after(30, self.update_frame)
        else:
            self.playing = False
            self.play_button.configure(text="Play")

    def download_output(self):
        if self.output_path and os.path.exists(self.output_path):
            save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
            if save_path:
                try:
                    import shutil
                    shutil.copy(self.output_path, save_path)
                    print(f"Video guardado en: {save_path}")
                except Exception as e:
                    print(f"Error al guardar el video: {e}")
        else:
            print("No hay video de salida para guardar.")

    def display_statistics(self, stats_results):
        """Muestra los gráficos y el texto de las estadísticas en la UI."""
        # Hacer visible el frame contenedor de estadísticas.
        # Como stats_frame es hijo de main_scroll_frame y se gestiona con pack,
        # para mostrarlo después de un hide_statistics(), simplemente se vuelve a packear.
        # Si lo ocultaste con grid_remove() (que es lo que haremos), lo volvemos a mostrar con grid().
        self.stats_frame.pack(pady=10, padx=10, fill="both", expand=True) # Asegurarse de que esté packeado

        # --- Cargar y mostrar los gráficos ---
        # Asegúrate de que las rutas en stats_results sean correctas y existan los archivos PNG.

        # Gráfico Drives vs Reveses
        if stats_results and stats_results.get('drives_reveses_chart'):
            try:
                img = Image.open(stats_results['drives_reveses_chart'])
                # Ajusta el tamaño (width, height) para que se vea decente horizontalmente
                # Considerando que hay 3 gráficos, 300x300px o similar podría estar bien.
                ctk_img = ctk.CTkImage(img, size=(292.2, 301.2))
                self.drives_reveses_chart_label.configure(image=ctk_img, text="")
                self.drives_reveses_chart_label.image = ctk_img # ¡IMPORTANTE! Mantener la referencia
                self.drives_reveses_chart_label.grid() # Asegurarse de que el label individual esté visible
            except FileNotFoundError:
                self.drives_reveses_chart_label.configure(image=None, text="Gráfico no encontrado.")
                self.drives_reveses_chart_label.grid() # Mostrar el mensaje de error
            except Exception as e:
                self.drives_reveses_chart_label.configure(image=None, text=f"Error cargando Drives/Reveses: {e}")
                self.drives_reveses_chart_label.grid()

        else:
            self.drives_reveses_chart_label.configure(image=None, text="Datos de Drives/Reveses no disponibles.")
            self.drives_reveses_chart_label.grid()


        # Gráfico Errores No Forzados vs Tiros Ganadores
        if stats_results and stats_results.get('unforced_winners_chart'):
            try:
                img = Image.open(stats_results['unforced_winners_chart'])
                ctk_img = ctk.CTkImage(img, size=(313.2, 223.2)) # Este es más rectangular, ajusta el tamaño
                self.unforced_winners_chart_label.configure(image=ctk_img, text="")
                self.unforced_winners_chart_label.image = ctk_img
                self.unforced_winners_chart_label.grid()
            except FileNotFoundError:
                self.unforced_winners_chart_label.configure(image=None, text="Gráfico no encontrado.")
                self.unforced_winners_chart_label.grid()
            except Exception as e:
                self.unforced_winners_chart_label.configure(image=None, text=f"Error cargando Errores/Ganadores: {e}")
                self.unforced_winners_chart_label.grid()
        else:
            self.unforced_winners_chart_label.configure(image=None, text="Datos de Errores/Ganadores no disponibles.")
            self.unforced_winners_chart_label.grid()


        # Gráfico Errores por Tipo de Golpe
        if stats_results and stats_results.get('errors_by_stroke_chart'):
            try:
                img = Image.open(stats_results['errors_by_stroke_chart'])
                ctk_img = ctk.CTkImage(img, size=(392.4, 301.2))
                self.errors_by_stroke_chart_label.configure(image=ctk_img, text="")
                self.errors_by_stroke_chart_label.image = ctk_img
                self.errors_by_stroke_chart_label.grid()
            except FileNotFoundError:
                self.errors_by_stroke_chart_label.configure(image=None, text="Gráfico no encontrado.")
                self.errors_by_stroke_chart_label.grid()
            except Exception as e:
                self.errors_by_stroke_chart_label.configure(image=None, text=f"Error cargando Errores por Golpe: {e}")
                self.errors_by_stroke_chart_label.grid()
        else:
            self.errors_by_stroke_chart_label.configure(image=None, text="Datos de Errores por Golpe no disponibles.")
            self.errors_by_stroke_chart_label.grid()

        # Texto Voleas jugadas
        if stats_results and stats_results.get('volleys_played') is not None:
            self.volleys_label.configure(text=f"Voleas jugadas: {stats_results['volleys_played']} \n\nPrimeros saques jugados: No se registraron saques")
        else:
            self.volleys_label.configure(text="Voleas jugadas: No disponible.")
        self.volleys_label.grid() # Asegurarse de que el label esté visible


    def hide_statistics(self):
        """Oculta el frame de estadísticas y limpia sus contenidos."""
        # Ocultar el frame principal de estadísticas.
        # Si lo mostraste con pack(), lo ocultas con pack_forget().
        self.stats_frame.pack_forget() 

        # Limpiar los labels de imagen y texto para liberar memoria y la UI
        # No es estrictamente necesario ocultarlos con grid_remove() si el padre ya está oculto,
        # pero limpiarlos es una buena práctica.
        if self.drives_reveses_chart_label:
            self.drives_reveses_chart_label.configure(image=None, text="")
            # Opcional: si quieres ocultar el espacio que ocupa: self.drives_reveses_chart_label.grid_remove()
        if self.unforced_winners_chart_label:
            self.unforced_winners_chart_label.configure(image=None, text="")
            # Opcional: self.unforced_winners_chart_label.grid_remove()
        if self.volleys_label:
            self.volleys_label.configure(text="")
            # Opcional: self.volleys_label.grid_remove()
        if self.errors_by_stroke_chart_label:
            self.errors_by_stroke_chart_label.configure(image=None, text="")
            # Opcional: self.errors_by_stroke_chart_label.grid_remove()

    def set_app_icon(self, icon_path):
        """
        Establece el icono de la aplicación para la ventana principal.
        Prioriza el uso de iconphoto para PNG/GIF, con opción a .ico para Windows.
        """
        try:
            # Verificar si el archivo del icono existe
            if not os.path.exists(icon_path):
                print(f"Advertencia: El archivo de icono '{icon_path}' no se encontró. No se pudo establecer el icono.")
                return

            # Si es un archivo PNG o GIF, usar wm_iconphoto (más moderno y cross-platform)
            if icon_path.lower().endswith(('.png', '.gif')):
                pil_image = Image.open(icon_path)
                
                # Opcional: Redimensionar el icono si es muy grande. Los iconos suelen ser pequeños (ej. 64x64, 128x128).
                # if pil_image.width > 128 or pil_image.height > 128:
                #     pil_image = pil_image.resize((128, 128), Image.LANCZOS) # Image.LANCZOS para mejor calidad al redimensionar

                tk_photo = ImageTk.PhotoImage(pil_image)
                
                # El primer argumento (True) significa que este icono se usará para todos los tamaños posibles
                self.wm_iconphoto(True, tk_photo) 
                print(f"Icono de aplicación PNG/GIF '{icon_path}' establecido.")
            
            # Si es un archivo ICO, usar wm_iconbitmap (más específico de Windows)
            elif icon_path.lower().endswith('.ico'):
                self.wm_iconbitmap(icon_path)
                print(f"Icono de aplicación ICO '{icon_path}' establecido.")
            else:
                print(f"Advertencia: Formato de icono no compatible para '{icon_path}'. Use .png, .gif o .ico.")

        except Exception as e:
            print(f"Error al establecer el icono de la aplicación: {e}")


if __name__ == "__main__":
    app = TennisAnalyzerApp()
    app.mainloop()