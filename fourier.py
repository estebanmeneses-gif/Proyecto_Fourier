"""
==========================================================================
 RECONSTRUCCIÓN DE SEÑALES USANDO SERIES DE FOURIER
 Universidad del Cauca
 Facultad de Ingeniería Electrónica y Telecomunicaciones

 Integrante: ESTEBAN CAMILO MENESES SANCHEZ
--------------------------------------------------------------------------
 Descripción:
   Aplicación con interfaz gráfica (Tkinter + Matplotlib) que reconstruye
   tres señales periódicas de amplitud 1 (cuadrada, triangular y diente de
   sierra) a partir de la suma parcial de su serie de Fourier con N
   armónicos. Muestra:
     1. La señal original x(t)
     2. La señal reconstruida x̂(t)
     3. La señal de error |x(t) - x̂(t)|
     4. El espectro de línea (|coeficiente| vs múltiplos de f0)
   y calcula el error medio cuadrático (MSE) y las potencias de la señal
   original y de la reconstruida.

 Restricción del proyecto (RNF2): no se usan funciones de alto nivel
 (ni FFT, ni generadores de señales de scipy). Los coeficientes se
 calcularon a mano y las métricas se obtienen con sumatorias explícitas.
==========================================================================
"""

import csv                      # Para exportar la tabla de resultados
import numpy as np              # Operaciones numéricas básicas (sin, cos, floor...)
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class FourierApp:
    """Ventana principal de la aplicación. Contiene la interfaz, la
    generación de señales, la reconstrucción por Fourier y las métricas."""

    def __init__(self, root):

        # ==========================================================
        # CONFIGURACIÓN PRINCIPAL
        # ==========================================================

        self.root = root
        self.root.title("Reconstrucción de Señales - Series de Fourier")
        self.root.geometry("1250x850")
        self.root.minsize(1100, 750)

        # Periodo de las señales. El enunciado permite escogerlo libremente
        # (no es un parámetro modificable por el usuario); se eligió T = 2 s.
        self.T = 2.0
        self.f0 = 1.0 / self.T            # Frecuencia fundamental: f0 = 1/T = 0.5 Hz
        self.w0 = 2.0 * np.pi / self.T    # Frecuencia angular: w0 = 2π/T = π rad/s

        # Valores de N exigidos por el enunciado (RF3)
        self.armonicos_valores = [1, 2, 3, 5, 10, 20, 50, 100, 150, 200]
        # Señales A, B y C del enunciado (RF2)
        self.senales = ["Señal Cuadrada", "Señal Triangular", "Señal Diente de Sierra"]

        # Frecuencia de muestreo fija (muestras por segundo).
        # Con N = 200 el armónico más alto es 200·f0 = 100 Hz, así que
        # fs = 2000 da 20 muestras por ciclo de ese armónico: suficiente
        # para dibujarlo bien y muy por encima del mínimo de Nyquist (2 por ciclo).
        self.fs = 2000.0
        # Intervalo máximo de visualización: limita la cantidad de muestras
        # para que la aplicación no se vuelva lenta ni se quede sin memoria.
        self.t_max_ventana = 50.0

        self.crear_interfaz()

        # Valores iniciales: señal cuadrada y N = 10 (índice 4 de la lista)
        self.cb_senal.current(0)
        self.cb_armonicos.current(4)

        # Primer dibujo al abrir la aplicación
        self.actualizar_graficas()

    # ==============================================================
    # INTERFAZ
    # ==============================================================

    def crear_interfaz(self):
        """Construye la ventana: título, fila de controles (señal, N,
        intervalo de tiempo y botones), fila de métricas y la figura con
        las 4 gráficas (RF1)."""

        # --- Título y datos fijos del sistema ---
        titulo = tk.Label(self.root, text="RECONSTRUCCIÓN DE SEÑALES MEDIANTE SERIES DE FOURIER",
                          font=("Arial", 16, "bold"))
        titulo.pack(pady=8)

        subtitulo = tk.Label(self.root,
                             text=f"Periodo T = {self.T:.1f} s    |    Frecuencia fundamental f₀ = {self.f0:.2f} Hz",
                             font=("Arial", 10))
        subtitulo.pack(pady=2)

        # --- Controles. Los Combobox son "readonly": el usuario solo puede
        #     escoger valores válidos, lo que evita errores de entrada. ---
        control_frame = tk.LabelFrame(self.root, text="Parámetros de simulación", padx=10, pady=8)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(control_frame, text="Señal:").grid(row=0, column=0, padx=5, pady=5)
        self.cb_senal = ttk.Combobox(control_frame, values=self.senales, state="readonly", width=24)
        self.cb_senal.grid(row=0, column=1, padx=5)

        tk.Label(control_frame, text="N Armónicos:").grid(row=0, column=2, padx=5)
        self.cb_armonicos = ttk.Combobox(control_frame, values=self.armonicos_valores,
                                         state="readonly", width=10)
        self.cb_armonicos.grid(row=0, column=3, padx=5)

        tk.Label(control_frame, text="t inicial (s):").grid(row=0, column=4, padx=5)
        self.entry_ti = ttk.Entry(control_frame, width=8)
        self.entry_ti.insert(0, "0")
        self.entry_ti.grid(row=0, column=5, padx=5)

        tk.Label(control_frame, text="t final (s):").grid(row=0, column=6, padx=5)
        self.entry_tf = ttk.Entry(control_frame, width=8)
        self.entry_tf.insert(0, "6")
        self.entry_tf.grid(row=0, column=7, padx=5)

        btn_actualizar = tk.Button(control_frame, text="CALCULAR", command=self.actualizar_graficas,
                                   bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), padx=12)
        btn_actualizar.grid(row=0, column=8, padx=15)

        btn_tabla = tk.Button(control_frame, text="GENERAR TABLA", command=self.generar_tabla,
                              font=("Arial", 10, "bold"), padx=12)
        btn_tabla.grid(row=0, column=9, padx=5)

        btn_analisis = tk.Button(control_frame, text="ANÁLISIS N", command=self.mostrar_analisis,
                                 font=("Arial", 10, "bold"), padx=12)
        btn_analisis.grid(row=0, column=10, padx=5)

        # --- Métricas de desempeño (RF5) ---
        metricas_frame = tk.LabelFrame(self.root,
                                       text="Métricas de desempeño (calculadas sobre un periodo)",
                                       padx=10, pady=5)
        metricas_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.lbl_mse = tk.Label(metricas_frame, text="MSE: ---", font=("Arial", 11, "bold"))
        self.lbl_mse.pack(side=tk.LEFT, padx=20)

        self.lbl_pot_orig = tk.Label(metricas_frame, text="Potencia original: ---",
                                     font=("Arial", 11, "bold"))
        self.lbl_pot_orig.pack(side=tk.LEFT, padx=20)

        self.lbl_pot_rec = tk.Label(metricas_frame, text="Potencia reconstruida: ---",
                                    font=("Arial", 11, "bold"))
        self.lbl_pot_rec.pack(side=tk.LEFT, padx=20)

        # --- Figura con 4 gráficas en cuadrícula 2x2 (RF4) ---
        # ax1: original | ax2: reconstruida | ax3: error | ax4: espectro
        self.fig = plt.Figure(figsize=(11, 6.5), dpi=100)

        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Actualización inmediata (RF1): al cambiar la señal o N, o al
        # presionar Enter en los campos de tiempo, se recalcula sin pulsar CALCULAR.
        self.cb_senal.bind("<<ComboboxSelected>>", lambda e: self.actualizar_graficas())
        self.cb_armonicos.bind("<<ComboboxSelected>>", lambda e: self.actualizar_graficas())
        self.entry_ti.bind("<Return>", lambda e: self.actualizar_graficas())
        self.entry_tf.bind("<Return>", lambda e: self.actualizar_graficas())

    # ==============================================================
    # UTILIDADES DE TIEMPO Y VALIDACIÓN
    # ==============================================================

    def crear_vector_tiempo(self, ti, tf):
        """Vector de tiempo con frecuencia de muestreo fija y muestras en el
        punto medio de cada intervalo (evita muestrear justo en las
        discontinuidades)."""
        # Número de muestras = duración del intervalo × muestras por segundo
        n = int(round((tf - ti) * self.fs))
        if n < 2:
            raise ValueError("El intervalo de tiempo es demasiado pequeño.")
        # t_k = ti + (k + 0.5)/fs, con k = 0, 1, ..., n-1
        return ti + (np.arange(n) + 0.5) / self.fs

    def leer_intervalo(self):
        """Lee t inicial y t final de la interfaz y valida que sean números
        finitos, que tf > ti y que el intervalo no supere el máximo (RNF1).
        Si algo falla, lanza ValueError con un mensaje para el usuario."""
        try:
            ti = float(self.entry_ti.get())
            tf = float(self.entry_tf.get())
        except ValueError:
            raise ValueError("t inicial y t final deben ser números (ej. 0 y 6).")
        if not (np.isfinite(ti) and np.isfinite(tf)):
            raise ValueError("t inicial y t final deben ser valores finitos.")
        if tf <= ti:
            raise ValueError("El tiempo final debe ser mayor que el tiempo inicial.")
        if tf - ti > self.t_max_ventana:
            raise ValueError(f"El intervalo máximo permitido es de {self.t_max_ventana:.0f} s.")
        return ti, tf

    def leer_armonicos(self):
        """Devuelve el número de armónicos N seleccionado."""
        try:
            return int(self.cb_armonicos.get())
        except ValueError:
            raise ValueError("Selecciona un número de armónicos válido.")

    def leer_senal(self):
        """Devuelve el nombre de la señal seleccionada."""
        tipo = self.cb_senal.get()
        if tipo not in self.senales:
            raise ValueError("Selecciona una señal válida.")
        return tipo

    # ==============================================================
    # GENERACIÓN DE SEÑAL ORIGINAL
    # ==============================================================

    def generar_senal_original(self, t, tipo):
        """Genera la señal ideal x(t), de periodo T y amplitud 1, con
        operaciones elementales (sin funciones de alto nivel, RNF2).

        Truco usado en la triangular y la sierra:
            u = t/T                 -> tiempo medido en "periodos"
            floor(u + 0.5)          -> redondea u al entero más cercano
            u - floor(u + 0.5)      -> queda en [-0.5, 0.5): es la posición
                                       dentro del periodo, centrada en 0.
        Así, la señal se repite automáticamente cada T segundos."""
        if "Cuadrada" in tipo:
            # Señal A: +1 en la primera mitad del periodo y -1 en la segunda.
            # Se toma el signo de sin(w0·t). Es una señal IMPAR.
            x = np.where(np.sin(self.w0 * t) >= 0, 1.0, -1.0)
        elif "Triangular" in tipo:
            # Señal B: 1 - 4·|posición en el periodo|.
            # Vale +1 en t = 0, -1 en t = ±T/2 y varía linealmente. Es PAR.
            x = 1 - 4 * np.abs((t / self.T) - np.floor((t / self.T) + 0.5))
        elif "Sierra" in tipo:
            # Señal C: rampa 2·(posición en el periodo), sube de -1 a +1
            # y cae bruscamente al terminar cada periodo. Es IMPAR.
            x = 2 * ((t / self.T) - np.floor(0.5 + (t / self.T)))
        else:
            raise ValueError("Tipo de señal no reconocido.")
        return x

    # ==============================================================
    # SERIE DE FOURIER
    # ==============================================================

    def reconstruir_fourier(self, t, N, tipo):
        """Suma parcial de la serie trigonométrica de Fourier:
            x̂(t) = a0 + Σ_{n=1..N} [ an·cos(n·w0·t) + bn·sin(n·w0·t) ]
        Coeficientes (calculados a mano; las tres señales tienen a0 = 0):
          Cuadrada  (impar):  bn = 4/(nπ) si n impar, 0 si n par;      an = 0
          Triangular (par):   an = 8/(nπ)² si n impar, 0 si n par;     bn = 0
          Diente de sierra:   bn = 2(-1)^(n+1)/(nπ) para todo n;       an = 0
        Devuelve la señal reconstruida y el espectro de línea (|coef| vs n·f0)."""
        x_rec = np.zeros_like(t)   # Acumulador de la suma parcial (empieza en a0 = 0)
        frecuencias = []           # Eje x del espectro: n·f0
        amplitudes = []            # Eje y del espectro: |an| o |bn|

        if "Cuadrada" in tipo:
            # Señal impar -> solo senos
            for n in range(1, N + 1):
                if n % 2 != 0:
                    bn = 4.0 / (n * np.pi)
                    # Se suma el n-ésimo armónico a la reconstrucción
                    x_rec += bn * np.sin(n * self.w0 * t)
                    amplitud = abs(bn)
                else:
                    # Los armónicos pares valen cero; se guardan igual para
                    # que el espectro los muestre en 0 (por eso N=1 y N=2
                    # dan el mismo resultado)
                    amplitud = 0.0
                frecuencias.append(n * self.f0)
                amplitudes.append(amplitud)

        elif "Triangular" in tipo:
            # Señal par -> solo cosenos. Coeficientes ~1/n²: convergen rápido
            for n in range(1, N + 1):
                if n % 2 != 0:
                    an = 8.0 / ((np.pi * n) ** 2)
                    x_rec += an * np.cos(n * self.w0 * t)
                    amplitud = abs(an)
                else:
                    amplitud = 0.0
                frecuencias.append(n * self.f0)
                amplitudes.append(amplitud)

        elif "Sierra" in tipo:
            # Señal impar -> solo senos, pero aparecen TODOS los armónicos
            # con signo alternado: (-1)^(n+1) = +1, -1, +1, ...
            for n in range(1, N + 1):
                bn = 2.0 * ((-1) ** (n + 1)) / (n * np.pi)
                x_rec += bn * np.sin(n * self.w0 * t)
                amplitud = abs(bn)
                frecuencias.append(n * self.f0)
                amplitudes.append(amplitud)
        else:
            raise ValueError("Tipo de señal no reconocido.")

        return x_rec, np.array(frecuencias), np.array(amplitudes)

    # ==============================================================
    # MÉTRICAS (cálculo manual, sin funciones de alto nivel)
    # ==============================================================

    def calcular_metricas_manual(self, x_orig, x_rec):
        """Calcula las métricas del RF5 con sumatorias explícitas (RNF2):

            MSE            = sqrt( (1/N)·Σ (x[n] - x̂[n])² )
            P_original     = (1/N)·Σ x[n]²
            P_reconstruida = (1/N)·Σ x̂[n]²

        Nota: el enunciado define el MSE CON raíz cuadrada (en rigor eso es
        el RMSE); se respeta la fórmula dada por el profesor.
        Por el teorema de Parseval, P_reconstruida -> P_original cuando N
        crece."""
        suma_error_sq = 0.0
        suma_pot_orig = 0.0
        suma_pot_rec = 0.0
        n_muestras = len(x_orig)

        # Un solo recorrido acumula las tres sumatorias
        for i in range(n_muestras):
            suma_error_sq += (x_orig[i] - x_rec[i]) ** 2
            suma_pot_orig += x_orig[i] ** 2
            suma_pot_rec += x_rec[i] ** 2

        mse = np.sqrt(suma_error_sq / n_muestras)   # Raíz del error cuadrático promedio
        pot_orig = suma_pot_orig / n_muestras       # Potencia promedio de x(t)
        pot_rec = suma_pot_rec / n_muestras         # Potencia promedio de x̂(t)

        return mse, pot_orig, pot_rec

    def metricas_un_periodo(self, tipo, N):
        """Métricas sobre exactamente un periodo: no dependen de la ventana
        de visualización."""
        t = self.crear_vector_tiempo(0.0, self.T)
        x_orig = self.generar_senal_original(t, tipo)
        x_rec, _, _ = self.reconstruir_fourier(t, N, tipo)
        return self.calcular_metricas_manual(x_orig, x_rec)

    # ==============================================================
    # OBTENER DATOS
    # ==============================================================

    def obtener_datos(self):
        """Lee y valida los parámetros de la interfaz y genera, sobre el
        intervalo elegido, la señal original, la reconstruida y el espectro."""
        N =self.leer_armonicos()
        ti, tf = self.leer_intervalo()
        tipo = self.leer_senal()

        t = self.crear_vector_tiempo(ti, tf)
        x_orig = self.generar_senal_original(t, tipo)
        x_rec, frecuencias, amplitudes = self.reconstruir_fourier(t, N, tipo)

        return t, x_orig, x_rec, frecuencias, amplitudes, N, tipo

    # ==============================================================
    # ACTUALIZAR GRÁFICAS
    # ==============================================================

    def actualizar_graficas(self):
        """Recalcula todo y redibuja las 4 gráficas y las métricas.
        Cualquier error se muestra en una ventana emergente y la aplicación
        sigue funcionando (RNF1)."""
        try:
            t, x_orig, x_rec, frecuencias, amplitudes, N, tipo = self.obtener_datos()

            # Señal de error punto a punto: |x(t) - x̂(t)|
            error_abs = np.abs(x_orig - x_rec)
            # Las métricas se calculan sobre un periodo (no sobre la ventana)
            mse, potencia_orig, potencia_rec = self.metricas_un_periodo(tipo, N)

            # Borrar lo dibujado antes
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()

            self.ax1.plot(t, x_orig, linewidth=1.5)
            self.ax1.set_title("1. Señal Original")
            self.ax1.set_xlabel("Tiempo (s)")
            self.ax1.set_ylabel("Amplitud")
            self.ax1.grid(True)

            self.ax2.plot(t, x_rec, linewidth=1.5)
            self.ax2.set_title(f"2. Reconstrucción - N = {N}")
            self.ax2.set_xlabel("Tiempo (s)")
            self.ax2.set_ylabel("Amplitud")
            self.ax2.grid(True)

            self.ax3.plot(t, error_abs, linewidth=1.2)
            self.ax3.axhline(0, linestyle="--", linewidth=1)
            self.ax3.set_title("3. Señal de Error |e(t)|")
            self.ax3.set_xlabel("Tiempo (s)")
            self.ax3.set_ylabel("|x(t) - x̂(t)|")
            self.ax3.grid(True)

            # Espectro de línea en múltiplos de la frecuencia fundamental
            # Se muestran TODOS los armónicos usados (1..N)
            self.ax4.stem(frecuencias / self.f0, amplitudes, basefmt=" ")
            self.ax4.set_xlim(0, N + 1)
            self.ax4.set_ylim(0, None)
            self.ax4.set_title("4. Espectro de Línea")
            self.ax4.set_xlabel(f"n (múltiplos de f₀ = {self.f0:.2f} Hz)")
            self.ax4.set_ylabel("|Coeficiente|")
            self.ax4.grid(True)

            self.lbl_mse.config(text=f"MSE: {mse:.8f}")
            self.lbl_pot_orig.config(text=f"Potencia original: {potencia_orig:.8f}")
            self.lbl_pot_rec.config(text=f"Potencia reconstruida: {potencia_rec:.8f}")

            self.fig.tight_layout()
            self.canvas.draw()

        except ValueError as e:
            # Errores de validación de los datos ingresados por el usuario
            messagebox.showerror("Error de entrada", str(e))
        except Exception as e:
            # Cualquier otro error inesperado: se informa sin cerrar el programa
            messagebox.showerror("Error", f"Ocurrió un error durante el cálculo:\n\n{e}")

    # ==============================================================
    # RESULTADOS PARA LA TABLA
    # ==============================================================

    def calcular_resultados(self, tipo):
        """Calcula MSE y potencias para cada N exigido en la tabla del
        enunciado (1, 2, 3, 5, 10, 20, 50, 100, 150, 200)."""
        resultados = []
        for N in self.armonicos_valores:
            mse, p_orig, p_rec = self.metricas_un_periodo(tipo, N)
            resultados.append({"N": N, "MSE": mse, "P_original": p_orig, "P_reconstruida": p_rec})
        return resultados

    # ==============================================================
    # VENTANA DE TABLA (las tres señales + exportar CSV)
    # ==============================================================

    def generar_tabla(self):
        """Abre una ventana con la tabla de resultados (una pestaña por
        señal) y un botón para exportarla a CSV."""
        try:
            datos = {tipo: self.calcular_resultados(tipo) for tipo in self.senales}

            ventana = tk.Toplevel(self.root)
            ventana.title("Tabla de Resultados - Series de Fourier")
            ventana.geometry("850x560")

            tk.Label(ventana, text="TABLA DE RESULTADOS", font=("Arial", 14, "bold")).pack(pady=10)

            notebook = ttk.Notebook(ventana)
            notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

            columnas = ("N", "MSE", "P_original", "P_reconstruida")

            for tipo in self.senales:
                frame = tk.Frame(notebook)
                notebook.add(frame, text=tipo)

                tabla = ttk.Treeview(frame, columns=columnas, show="headings", height=12)
                tabla.heading("N", text="N Armónicos")
                tabla.heading("MSE", text="Error Medio Cuadrático")
                tabla.heading("P_original", text="Potencia Original")
                tabla.heading("P_reconstruida", text="Potencia Reconstruida")

                tabla.column("N", width=100, anchor="center")
                tabla.column("MSE", width=190, anchor="center")
                tabla.column("P_original", width=180, anchor="center")
                tabla.column("P_reconstruida", width=200, anchor="center")

                for r in datos[tipo]:
                    tabla.insert("", "end", values=(r["N"], f"{r['MSE']:.8f}",
                                                    f"{r['P_original']:.8f}",
                                                    f"{r['P_reconstruida']:.8f}"))
                tabla.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            btn_csv = tk.Button(ventana, text="EXPORTAR CSV",
                                command=lambda: self.exportar_csv(datos),
                                font=("Arial", 10, "bold"), padx=12)
            btn_csv.pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar la tabla:\n\n{e}")

    def exportar_csv(self, datos):
        """Guarda la tabla de las tres señales en un archivo .csv (se puede
        abrir en Excel para hacer las gráficas de la sustentación)."""
        try:
            ruta = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV", "*.csv")],
                                                title="Guardar tabla de resultados")
            if not ruta:      # El usuario canceló el diálogo
                return
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                escritor = csv.writer(f)
                escritor.writerow(["Señal", "N armónicos", "MSE", "Potencia original",
                                   "Potencia reconstruida"])
                for tipo, filas in datos.items():
                    for r in filas:
                        escritor.writerow([tipo, r["N"], f"{r['MSE']:.8f}",
                                           f"{r['P_original']:.8f}", f"{r['P_reconstruida']:.8f}"])
            messagebox.showinfo("Exportación", "Tabla guardada correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el archivo:\n\n{e}")

    # ==============================================================
    # GRÁFICAS DE ANÁLISIS (las tres señales)
    # ==============================================================

    def mostrar_analisis(self):
        """Abre una ventana con las gráficas MSE vs N y Potencia vs N para
        las tres señales (escala logarítmica en N) y un texto de
        interpretación."""
        try:
            datos = {tipo: self.calcular_resultados(tipo) for tipo in self.senales}

            ventana = tk.Toplevel(self.root)
            ventana.title("Análisis de Resultados")
            ventana.geometry("1100x800")

            texto = (
                "INTERPRETACIÓN DE RESULTADOS:\n\n"
                "1. Convergencia: al aumentar N, el MSE disminuye y la reconstrucción se acerca a la señal original. "
                "La triangular converge más rápido (coeficientes ~1/n²) que la cuadrada y la sierra (~1/n).\n"
                "2. Parseval: la potencia reconstruida tiende a la potencia original (1, 1/3 y 1/3) al añadir armónicos.\n"
                "3. Fenómeno de Gibbs: en las discontinuidades (cuadrada y sierra) aparece un sobreimpulso cercano al 9 % "
                "que no desaparece al aumentar N; solo se hace más angosto."
            )
            tk.Label(ventana, text=texto, font=("Arial", 10, "italic"), justify="left",
                     bg="#e8f4f8", padx=15, pady=10, relief="groove",
                     wraplength=1050).pack(fill=tk.X, padx=15, pady=10)

            fig = plt.Figure(figsize=(10, 6), dpi=100)
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            marcadores = ["o", "s", "^"]
            for tipo, m in zip(self.senales, marcadores):
                res = datos[tipo]
                N = [r["N"] for r in res]
                ax1.plot(N, [r["MSE"] for r in res], marker=m, linewidth=1.5, label=tipo)
                ax2.plot(N, [r["P_reconstruida"] for r in res], marker=m, linewidth=1.5,
                         label=f"{tipo} (reconstruida)")
                ax2.axhline(res[0]["P_original"], linestyle="--", linewidth=1, color="gray")

            ax1.set_title("Error Medio Cuadrático vs Número de Armónicos")
            ax1.set_xlabel("Número de armónicos N")
            ax1.set_ylabel("MSE")
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.grid(True, which="both")
            ax1.legend()

            ax2.set_title("Potencia reconstruida vs Número de Armónicos (líneas punteadas: potencia original)")
            ax2.set_xlabel("Número de armónicos N")
            ax2.set_ylabel("Potencia")
            ax2.set_xscale("log")
            ax2.grid(True, which="both")
            ax2.legend()

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=ventana)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar el análisis:\n\n{e}")


# ==================================================================
# PUNTO DE ENTRADA: crea la ventana y arranca el ciclo de eventos
# ==================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = FourierApp(root)
    root.mainloop()