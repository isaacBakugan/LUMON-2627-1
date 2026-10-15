# Nombre del integrante: Alexander Ramirez
# Cédula del integrante: 29611077

import csv
import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# CONSTANTES DEL PROGRAMA
# -----------------------------------------------------------------------------
# El enunciado no pide al usuario escoger una tasa de aprendizaje y tampoco
# incluye este valor entre los hiperparametros del archivo de red. Por eso se
# usa una tasa fija y documentada aqui.
TASA_APRENDIZAJE = 0.5

# Semilla fija para que crear dos veces la misma arquitectura produzca pesos
# iniciales reproducibles. Esto facilita estudiar y explicar los resultados.
SEMILLA_ALEATORIA = 42


# =============================================================================
# UTILIDADES DE ENTRADA POR CONSOLA
# =============================================================================

def leer_entero(mensaje, minimo=1):
    """Lee un numero entero desde la consola y valida un valor minimo."""
    while True:
        texto = input(mensaje).strip()
        try:
            valor = int(texto)
            if valor < minimo:
                print("El valor debe ser mayor o igual que", minimo)
                continue
            return valor
        except ValueError:
            print("Entrada invalida. Debe escribir un numero entero.")


def pedir_ruta_existente(mensaje):
    """Solicita una ruta hasta que el archivo exista."""
    while True:
        ruta = input(mensaje).strip()
        if os.path.isfile(ruta):
            return ruta
        print("No se encontro el archivo. Revise la ruta e intente otra vez.")


# =============================================================================
# LECTURA Y VALIDACION DE LOS ARCHIVOS DE DATOS
# =============================================================================

def cargar_datos_csv(ruta):
    """Carga un CSV de datos con el formato x1,x2,...,xn,y.

    Las primeras columnas son las entradas y la ultima es la clase esperada.
    Se permite una fila de encabezado y se ignoran filas completamente vacias.

    Importante: el archivo 3-d_4-class_test.csv suministrado con la tarea tiene
    varias filas ',,,' al final. Esta funcion las ignora de forma segura.
    """
    filas_numericas = []
    encabezado = None
    columnas_esperadas = None

    with open(ruta, "r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.reader(archivo)

        for numero_linea, fila in enumerate(lector, start=1):
            # Una fila como ,,, contiene celdas pero todas estan vacias.
            if not fila or all(celda.strip() == "" for celda in fila):
                continue

            try:
                valores = [float(celda.strip()) for celda in fila]
            except ValueError:
                # Se permite encabezado solamente antes de los datos.
                if encabezado is None and len(filas_numericas) == 0:
                    encabezado = [celda.strip() for celda in fila]
                    continue
                raise ValueError(
                    "La linea " + str(numero_linea) +
                    " contiene un valor no numerico."
                )

            if columnas_esperadas is None:
                columnas_esperadas = len(valores)
                if columnas_esperadas < 2:
                    raise ValueError(
                        "El CSV debe contener al menos una entrada y una clase."
                    )

            if len(valores) != columnas_esperadas:
                raise ValueError(
                    "La linea " + str(numero_linea) +
                    " tiene una cantidad de columnas diferente."
                )

            filas_numericas.append(valores)

    if len(filas_numericas) == 0:
        raise ValueError("El archivo no contiene datos numericos.")

    datos = np.array(filas_numericas, dtype=float)
    entradas = datos[:, :-1]
    clases = datos[:, -1]

    if encabezado is None or len(encabezado) != datos.shape[1]:
        encabezado = ["x" + str(i + 1) for i in range(entradas.shape[1])]
        encabezado.append("y")

    return entradas, clases, encabezado


def clases_ordenadas(vector_clases):
    """Devuelve las clases distintas en orden ascendente como una lista."""
    return sorted(float(valor) for valor in np.unique(vector_clases))


def validar_datos_para_red(red, entradas, salidas):
    """Comprueba que dimensiones y cantidad de clases coincidan con la red.

    Si no coinciden, devuelve False e imprime exactamente como deberia
    configurarse la red para ese archivo.
    """
    entradas_archivo = entradas.shape[1]
    clases_archivo = clases_ordenadas(salidas)
    salidas_archivo = len(clases_archivo)

    if (entradas_archivo != red.neuronas_entrada or
            salidas_archivo != red.neuronas_salida):
        print("\nERROR: la configuracion de la red no corresponde al archivo.")
        print("Este archivo requiere:")
        print("  Neuronas de entrada (u):", entradas_archivo)
        print("  Neuronas de salida  (v):", salidas_archivo)
        print("La red actual tiene:")
        print("  Neuronas de entrada (u):", red.neuronas_entrada)
        print("  Neuronas de salida  (v):", red.neuronas_salida)
        return False

    return True


def convertir_a_one_hot(salidas, etiquetas):
    """Convierte clases numericas a una matriz one-hot.

    Ejemplo con etiquetas [-1, 1]:
      -1 -> [1, 0]
       1 -> [0, 1]
    """
    mapa = {etiqueta: indice for indice, etiqueta in enumerate(etiquetas)}
    indices = np.array([mapa[float(valor)] for valor in salidas], dtype=int)

    one_hot = np.zeros((len(salidas), len(etiquetas)), dtype=float)
    one_hot[np.arange(len(salidas)), indices] = 1.0

    return one_hot, indices


# =============================================================================
# FUNCIONES DE ACTIVACION
# =============================================================================

def sigmoide(z):
    """Funcion sigmoide usada en todas las capas ocultas."""
    # Recortar evita overflow numerico en exp() sin cambiar la idea matematica.
    z_seguro = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z_seguro))


def derivada_sigmoide(desde_activacion):
    """Derivada de sigmoide usando su salida ya calculada: a*(1-a)."""
    return desde_activacion * (1.0 - desde_activacion)


def softmax(z):
    """Convierte los valores de salida en probabilidades que suman 1."""
    # Restar el maximo de cada fila mejora la estabilidad numerica.
    z_estable = z - np.max(z, axis=1, keepdims=True)
    exponenciales = np.exp(z_estable)
    return exponenciales / np.sum(exponenciales, axis=1, keepdims=True)


def entropia_cruzada(probabilidades, objetivos_one_hot):
    """Calcula el error de clasificacion usado durante el entrenamiento."""
    epsilon = 1e-12
    probabilidades_seguras = np.clip(probabilidades, epsilon, 1.0)
    errores = objetivos_one_hot * np.log(probabilidades_seguras)
    return -float(np.mean(np.sum(errores, axis=1)))


# =============================================================================
# PERCEPTRON MULTICAPA
# =============================================================================

class PerceptronMulticapa:
    """Red neuronal multicapa programada desde cero.

    Los pesos se guardan como matrices:
        filas    = conexiones que llegan desde la capa anterior
        columnas = neuronas de la capa actual

    El formato CSV definido en el enunciado no incluye una columna para bias.
    Para respetar exactamente ese formato, esta implementacion entrena y guarda
    solamente los pesos de conexion especificados por la tarea.
    """

    def __init__(self, neuronas_entrada, neuronas_salida,
                 capas_ocultas, neuronas_por_capa,
                 epocas_acumuladas=0, inicializar=True):
        self.neuronas_entrada = int(neuronas_entrada)
        self.neuronas_salida = int(neuronas_salida)
        self.capas_ocultas = int(capas_ocultas)
        self.neuronas_por_capa = int(neuronas_por_capa)
        self.epocas_acumuladas = int(epocas_acumuladas)
        self.pesos = []

        if inicializar:
            self.inicializar_pesos()

    def dimensiones_capas(self):
        """Devuelve [entrada, ocultas..., salida]."""
        return (
            [self.neuronas_entrada] +
            [self.neuronas_por_capa] * self.capas_ocultas +
            [self.neuronas_salida]
        )

    def inicializar_pesos(self):
        """Crea los pesos iniciales usando inicializacion Xavier uniforme."""
        generador = np.random.default_rng(SEMILLA_ALEATORIA)
        dimensiones = self.dimensiones_capas()
        self.pesos = []

        for cantidad_anterior, cantidad_actual in zip(
                dimensiones[:-1], dimensiones[1:]):
            limite = np.sqrt(6.0 / (cantidad_anterior + cantidad_actual))
            matriz = generador.uniform(
                -limite,
                limite,
                size=(cantidad_anterior, cantidad_actual)
            )
            self.pesos.append(matriz)

    def alimentacion_hacia_delante(self, entradas):
        """Realiza el forward pass y conserva activaciones para backpropagation.

        Retorna una lista:
          activaciones[0]  = entradas originales
          activaciones[1]  = primera capa oculta
          ...
          activaciones[-1] = probabilidades softmax de salida
        """
        activaciones = [entradas]
        actual = entradas

        # Todas las matrices menos la ultima pertenecen a capas ocultas.
        for matriz_pesos in self.pesos[:-1]:
            suma_ponderada = actual @ matriz_pesos
            actual = sigmoide(suma_ponderada)
            activaciones.append(actual)

        # La ultima capa usa softmax para clasificacion multiclase.
        suma_salida = actual @ self.pesos[-1]
        probabilidades = softmax(suma_salida)
        activaciones.append(probabilidades)

        return activaciones

    def predecir_indices(self, entradas):
        """Devuelve el indice de la neurona de salida con mayor probabilidad."""
        probabilidades = self.alimentacion_hacia_delante(entradas)[-1]
        return np.argmax(probabilidades, axis=1)

    def entrenar(self, entradas, objetivos_one_hot, numero_epocas):
        """Entrena la red mediante retropropagacion y descenso de gradiente.

        La retropropagacion se implementa manualmente. NumPy solamente realiza
        las multiplicaciones de matrices; no realiza el aprendizaje por nosotros.
        """
        errores_por_epoca = []
        cantidad_ejemplos = entradas.shape[0]

        for _ in range(numero_epocas):
            # -------------------------------------------------------------
            # 1) FORWARD PASS
            # -------------------------------------------------------------
            activaciones = self.alimentacion_hacia_delante(entradas)
            predicciones = activaciones[-1]

            error = entropia_cruzada(predicciones, objetivos_one_hot)
            errores_por_epoca.append(error)

            # -------------------------------------------------------------
            # 2) BACKPROPAGATION
            # -------------------------------------------------------------
            # Para softmax + entropia cruzada, la derivada final se simplifica
            # a prediccion - objetivo.
            delta = (predicciones - objetivos_one_hot) / cantidad_ejemplos
            gradientes = [None] * len(self.pesos)

            # Recorremos las capas desde la salida hacia la entrada.
            indice_capa = len(self.pesos) - 1

            while indice_capa >= 0:
                activacion_anterior = activaciones[indice_capa]
                gradientes[indice_capa] = activacion_anterior.T @ delta

                # Si aun hay una capa oculta anterior, propagamos su error.
                if indice_capa > 0:
                    error_anterior = delta @ self.pesos[indice_capa].T
                    activacion_oculta = activaciones[indice_capa]
                    delta = (
                        error_anterior *
                        derivada_sigmoide(activacion_oculta)
                    )

                indice_capa -= 1

            # -------------------------------------------------------------
            # 3) ACTUALIZACION DE PESOS
            # -------------------------------------------------------------
            for i in range(len(self.pesos)):
                self.pesos[i] = (
                    self.pesos[i] -
                    TASA_APRENDIZAJE * gradientes[i]
                )

        # Es importante acumular, no reemplazar, para que entrenamientos
        # posteriores queden reflejados en el hiperparametro e.
        self.epocas_acumuladas += numero_epocas

        return errores_por_epoca

    def imprimir_hiperparametros(self):
        """Muestra los cinco hiperparametros solicitados por la tarea."""
        print("\n========================================")
        print("HIPERPARAMETROS DE LA RED")
        print("========================================")
        print("u - Neuronas de entrada:", self.neuronas_entrada)
        print("v - Neuronas de salida:", self.neuronas_salida)
        print("L - Capas ocultas:", self.capas_ocultas)
        print("b - Neuronas por capa:", self.neuronas_por_capa)
        print("e - Epocas acumuladas:", self.epocas_acumuladas)
        print("Tasa de aprendizaje fija:", TASA_APRENDIZAJE)

    def guardar(self, ruta):
        """Guarda la red en el CSV solicitado por el enunciado.

        Formato:
          tipo,capa,posicion,peso_0,...,peso_(c-1)

        donde c = max(u, b).
        """x
        if not ruta.lower().endswith(".csv"):
            ruta += ".csv"

        c = max(self.neuronas_entrada, self.neuronas_por_capa)
        encabezado = ["tipo", "capa", "posicion"]
        encabezado += ["peso_" + str(i) for i in range(c)]

        with open(ruta, "w", encoding="utf-8", newline="") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(encabezado)

            # Primeras 5 filas: hiperparametros. Las columnas desde la tercera
            # en adelante quedan vacias, como indica el enunciado.
            for clave, valor in [
                ("u", self.neuronas_entrada),
                ("v", self.neuronas_salida),
                ("L", self.capas_ocultas),
                ("b", self.neuronas_por_capa),
                ("e", self.epocas_acumuladas),
            ]:
                escritor.writerow([clave, valor] + [""] * (c + 1))

            # Capas ocultas: h, numero de capa, posicion, pesos...
            for indice_oculta in range(self.capas_ocultas):
                numero_capa = indice_oculta + 1
                matriz = self.pesos[indice_oculta]

                for posicion in range(self.neuronas_por_capa):
                    pesos_neurona = matriz[:, posicion].tolist()
                    relleno = [""] * (c - len(pesos_neurona))
                    fila = ["h", numero_capa, posicion]
                    fila += [formatear_peso(p) for p in pesos_neurona]
                    fila += relleno
                    escritor.writerow(fila)

            # Capa de salida. Se numera despues de la ultima hidden.
            numero_capa_salida = self.capas_ocultas + 1
            matriz_salida = self.pesos[-1]

            for posicion in range(self.neuronas_salida):
                pesos_neurona = matriz_salida[:, posicion].tolist()
                relleno = [""] * (c - len(pesos_neurona))
                fila = ["o", numero_capa_salida, posicion]
                fila += [formatear_peso(p) for p in pesos_neurona]
                fila += relleno
                escritor.writerow(fila)

        return ruta

    @classmethod
    def cargar(cls, ruta):
        """Reconstruye una red completa desde un archivo CSV guardado."""
        with open(ruta, "r", encoding="utf-8-sig", newline="") as archivo:
            filas = list(csv.reader(archivo))

        if len(filas) < 6:
            raise ValueError("El archivo de red esta incompleto.")

        encabezado = filas[0]
        if len(encabezado) < 4:
            raise ValueError("El encabezado del archivo de red no es valido.")

        hiperparametros = {}
        for fila in filas[1:6]:
            if len(fila) < 2:
                raise ValueError("Una fila de hiperparametros esta incompleta.")
            hiperparametros[fila[0].strip()] = int(float(fila[1]))

        requeridos = {"u", "v", "L", "b", "e"}
        if set(hiperparametros.keys()) != requeridos:
            raise ValueError(
                "Las primeras cinco filas deben ser u, v, L, b y e."
            )

        u = hiperparametros["u"]
        v = hiperparametros["v"]
        L = hiperparametros["L"]
        b = hiperparametros["b"]
        e = hiperparametros["e"]
        c = max(u, b)

        if len(encabezado) != c + 3:
            raise ValueError(
                "El archivo debe tener 3 + c columnas, con c=max(u,b)=" +
                str(c) + "."
            )

        red = cls(u, v, L, b, epocas_acumuladas=e, inicializar=False)

        dimensiones = red.dimensiones_capas()
        red.pesos = []
        for anterior, actual in zip(dimensiones[:-1], dimensiones[1:]):
            red.pesos.append(np.zeros((anterior, actual), dtype=float))

        neuronas_leidas = set()

        for fila in filas[6:]:
            if not fila or all(celda.strip() == "" for celda in fila):
                continue

            if len(fila) != c + 3:
                raise ValueError("Una fila de neurona tiene columnas incorrectas.")

            tipo = fila[0].strip().lower()
            capa = int(float(fila[1]))
            posicion = int(float(fila[2]))

            if tipo == "h":
                if capa < 1 or capa > L:
                    raise ValueError("Numero de capa hidden invalido.")
                indice_matriz = capa - 1
                cantidad_entradas = u if capa == 1 else b
                cantidad_neuronas = b
            elif tipo == "o":
                if capa != L + 1:
                    raise ValueError("La capa de salida debe ser L+1.")
                indice_matriz = L
                cantidad_entradas = b if L > 0 else u
                cantidad_neuronas = v
            else:
                raise ValueError("Tipo de neurona invalido: " + tipo)

            if posicion < 0 or posicion >= cantidad_neuronas:
                raise ValueError("Posicion de neurona fuera de rango.")

            pesos_texto = fila[3:3 + cantidad_entradas]
            if any(texto.strip() == "" for texto in pesos_texto):
                raise ValueError("Faltan pesos en una neurona del archivo.")

            pesos_neurona = np.array(
                [float(texto) for texto in pesos_texto],
                dtype=float
            )

            red.pesos[indice_matriz][:, posicion] = pesos_neurona
            neuronas_leidas.add((tipo, capa, posicion))

        esperadas = L * b + v
        if len(neuronas_leidas) != esperadas:
            raise ValueError(
                "El archivo contiene " + str(len(neuronas_leidas)) +
                " neuronas, pero la arquitectura requiere " + str(esperadas) + "."
            )

        return red


def formatear_peso(valor):
    """Guarda suficientes decimales para reconstruir el float con precision."""
    return format(float(valor), ".17g")


# =============================================================================
# EVALUACION Y MATRIZ DE CONFUSION
# =============================================================================

def evaluar_red(red, entradas, salidas_reales):
    """Calcula predicciones, aciertos y porcentaje usando las clases del CSV."""
    etiquetas = clases_ordenadas(salidas_reales)
    _, indices_reales = convertir_a_one_hot(salidas_reales, etiquetas)
    indices_predichos = red.predecir_indices(entradas)

    predicciones_etiqueta = np.array(
        [etiquetas[indice] for indice in indices_predichos],
        dtype=float
    )
    correctos = indices_predichos == indices_reales
    exactitud = float(np.mean(correctos))

    return (
        predicciones_etiqueta,
        indices_reales,
        indices_predichos,
        correctos,
        exactitud,
        etiquetas,
    )


def crear_matriz_confusion(indices_reales, indices_predichos, cantidad_clases):
    """Construye manualmente la matriz de confusion."""
    matriz = np.zeros((cantidad_clases, cantidad_clases), dtype=int)

    for real, predicho in zip(indices_reales, indices_predichos):
        matriz[real, predicho] += 1

    return matriz


# =============================================================================
# GRAFICOS
# =============================================================================

def graficar_datos(entradas, salidas, encabezado):
    """Grafica el dataset en 2D o las primeras tres entradas en 3D."""
    cantidad_dimensiones = entradas.shape[1]

    if cantidad_dimensiones >= 3:
        figura = plt.figure("1 - Datos del archivo")
        eje = figura.add_subplot(111, projection="3d")
        puntos = eje.scatter(
            entradas[:, 0], entradas[:, 1], entradas[:, 2],
            c=salidas, cmap="viridis", edgecolors="black", linewidths=0.3
        )
        eje.set_xlabel(encabezado[0])
        eje.set_ylabel(encabezado[1])
        eje.set_zlabel(encabezado[2])
        eje.set_title("Datos del archivo - primeras tres entradas")
        figura.colorbar(puntos, ax=eje, label="Clase esperada")
    else:
        plt.figure("1 - Datos del archivo")
        if cantidad_dimensiones == 2:
            y_grafico = entradas[:, 1]
            etiqueta_y = encabezado[1]
        else:
            y_grafico = np.zeros(len(entradas))
            etiqueta_y = "eje auxiliar"

        puntos = plt.scatter(
            entradas[:, 0], y_grafico,
            c=salidas, cmap="viridis", edgecolors="black", linewidths=0.3
        )
        plt.xlabel(encabezado[0])
        plt.ylabel(etiqueta_y)
        plt.title("Datos del archivo")
        plt.grid(True, alpha=0.25)
        plt.colorbar(puntos, label="Clase esperada")


def graficar_correctos(entradas, correctos, encabezado):
    """Grafica verde para prediccion correcta y rojo para incorrecta."""
    colores = ["green" if correcto else "red" for correcto in correctos]
    cantidad_dimensiones = entradas.shape[1]

    if cantidad_dimensiones >= 3:
        figura = plt.figure("2 - Predicciones correctas/incorrectas")
        eje = figura.add_subplot(111, projection="3d")
        eje.scatter(
            entradas[:, 0], entradas[:, 1], entradas[:, 2],
            c=colores, edgecolors="black", linewidths=0.3
        )
        eje.set_xlabel(encabezado[0])
        eje.set_ylabel(encabezado[1])
        eje.set_zlabel(encabezado[2])
        eje.set_title("Verde = correcto | Rojo = incorrecto")
    else:
        plt.figure("2 - Predicciones correctas/incorrectas")
        if cantidad_dimensiones == 2:
            y_grafico = entradas[:, 1]
            etiqueta_y = encabezado[1]
        else:
            y_grafico = np.zeros(len(entradas))
            etiqueta_y = "eje auxiliar"

        plt.scatter(
            entradas[:, 0], y_grafico,
            c=colores, edgecolors="black", linewidths=0.3
        )
        plt.xlabel(encabezado[0])
        plt.ylabel(etiqueta_y)
        plt.title("Verde = correcto | Rojo = incorrecto")
        plt.grid(True, alpha=0.25)


def graficar_error_entrenamiento(errores):
    """Grafico de linea del error para cada epoca del entrenamiento actual."""
    plt.figure("3 - Error por epoca")
    epocas = np.arange(1, len(errores) + 1)
    plt.plot(epocas, errores)
    plt.xlabel("Epoca")
    plt.ylabel("Entropia cruzada")
    plt.title("Error de entrenamiento por epoca")
    plt.grid(True, alpha=0.25)


def graficar_matriz_confusion(matriz, etiquetas):
    """Muestra la matriz de confusion usando matplotlib.pyplot.matshow."""
    plt.figure("3 - Matriz de confusion")
    plt.matshow(matriz, fignum=plt.gcf().number, cmap="Blues")
    plt.title("Matriz de confusion", pad=18)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")

    nombres = [formatear_etiqueta(etiqueta) for etiqueta in etiquetas]
    posiciones = np.arange(len(etiquetas))
    plt.xticks(posiciones, nombres)
    plt.yticks(posiciones, nombres)

    # Escribimos el numero dentro de cada celda para facilitar la lectura.
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            plt.text(
                columna, fila, str(matriz[fila, columna]),
                ha="center", va="center"
            )

    plt.colorbar(label="Cantidad de ejemplos")


def formatear_etiqueta(valor):
    """Evita mostrar 1.0 cuando la clase realmente es un entero 1."""
    if float(valor).is_integer():
        return str(int(valor))
    return str(valor)


def mostrar_tres_graficos_entrenamiento(
        entradas, salidas, encabezado, correctos, errores):
    """Crea los tres graficos pedidos despues de entrenar."""
    graficar_datos(entradas, salidas, encabezado)
    graficar_correctos(entradas, correctos, encabezado)
    graficar_error_entrenamiento(errores)
    plt.show()
    plt.close("all")


def mostrar_tres_graficos_prueba(
        entradas, salidas, encabezado, correctos,
        indices_reales, indices_predichos, etiquetas):
    """Crea los tres graficos pedidos despues de probar."""
    graficar_datos(entradas, salidas, encabezado)
    graficar_correctos(entradas, correctos, encabezado)

    matriz = crear_matriz_confusion(
        indices_reales,
        indices_predichos,
        len(etiquetas)
    )
    graficar_matriz_confusion(matriz, etiquetas)

    plt.show()
    plt.close("all")


# =============================================================================
# ACCIONES DEL MENU
# =============================================================================

def crear_red_desde_consola():
    """Solicita los cuatro valores de arquitectura requeridos."""
    print("\n--- Crear nuevo perceptron multicapa ---")
    u = leer_entero("Numero de neuronas de entrada: ", minimo=1)
    v = leer_entero("Numero de neuronas de salida: ", minimo=2)
    L = leer_entero("Numero de capas ocultas: ", minimo=1)
    b = leer_entero("Numero de neuronas por capa oculta: ", minimo=1)

    red = PerceptronMulticapa(u, v, L, b)
    print("\nRed creada correctamente.")
    red.imprimir_hiperparametros()
    return red


def cargar_red_desde_consola():
    """Solicita el CSV de una red hasta que pueda cargarse correctamente."""
    print("\n--- Cargar perceptron multicapa ---")

    while True:
        ruta = pedir_ruta_existente("Archivo CSV de la red: ")
        try:
            red = PerceptronMulticapa.cargar(ruta)
            print("\nRed cargada correctamente desde:", ruta)
            red.imprimir_hiperparametros()
            return red
        except (ValueError, OSError) as error:
            print("No se pudo cargar la red:", error)


def accion_entrenar(red):
    """Carga un dataset, valida, entrena, evalua y muestra tres graficos."""
    print("\n--- Entrenamiento por retropropagacion ---")
    ruta = pedir_ruta_existente("Archivo CSV de entrenamiento: ")

    try:
        entradas, salidas, encabezado = cargar_datos_csv(ruta)
    except (ValueError, OSError) as error:
        print("No se pudo cargar el archivo:", error)
        return

    if not validar_datos_para_red(red, entradas, salidas):
        return

    numero_epocas = leer_entero("Numero de epocas: ", minimo=1)
    etiquetas = clases_ordenadas(salidas)
    objetivos, _ = convertir_a_one_hot(salidas, etiquetas)

    print("\nEntrenando...")
    errores = red.entrenar(entradas, objetivos, numero_epocas)

    (
        _, indices_reales, indices_predichos,
        correctos, exactitud, _
    ) = evaluar_red(red, entradas, salidas)

    print("Entrenamiento completado.")
    print("Exactitud sobre el archivo de entrenamiento:",
          round(exactitud * 100.0, 2), "%")
    print("Error final:", round(errores[-1], 8))
    red.imprimir_hiperparametros()

    mostrar_tres_graficos_entrenamiento(
        entradas, salidas, encabezado, correctos, errores
    )


def accion_probar(red):
    """Ejecuta alimentacion hacia delante con un archivo de prueba."""
    print("\n--- Prueba por alimentacion hacia delante ---")
    ruta = pedir_ruta_existente("Archivo CSV de prueba: ")

    try:
        entradas, salidas, encabezado = cargar_datos_csv(ruta)
    except (ValueError, OSError) as error:
        print("No se pudo cargar el archivo:", error)
        return

    if not validar_datos_para_red(red, entradas, salidas):
        return

    (
        _, indices_reales, indices_predichos,
        correctos, exactitud, etiquetas
    ) = evaluar_red(red, entradas, salidas)

    print("\nPrueba completada.")
    print("Predicciones correctas:", int(np.sum(correctos)), "de", len(correctos))
    print("Exactitud:", round(exactitud * 100.0, 2), "%")
    red.imprimir_hiperparametros()

    mostrar_tres_graficos_prueba(
        entradas, salidas, encabezado, correctos,
        indices_reales, indices_predichos, etiquetas
    )


def accion_guardar(red):
    """Solicita una ruta y guarda la red."""
    ruta = input("Nombre o ruta del archivo para guardar la red: ").strip()

    if ruta == "":
        print("Debe indicar un nombre de archivo.")
        return

    try:
        ruta_final = red.guardar(ruta)
        print("Red guardada correctamente en:", ruta_final)
        red.imprimir_hiperparametros()
    except OSError as error:
        print("No se pudo guardar la red:", error)


def menu_red(red):
    """Menu recurrente que permite seguir entrenando la misma red."""
    while True:
        print("\n========================================")
        print("MENU DEL PERCEPTRON MULTICAPA")
        print("========================================")
        print("1. Entrenar con retropropagacion")
        print("2. Probar con alimentacion hacia delante")
        print("3. Guardar la red")
        print("4. Salir")

        opcion = input("Seleccione una opcion: ").strip()

        if opcion == "1":
            accion_entrenar(red)
        elif opcion == "2":
            accion_probar(red)
        elif opcion == "3":
            accion_guardar(red)
        elif opcion == "4":
            print("Programa finalizado.")
            break
        else:
            print("Opcion invalida. Seleccione 1, 2, 3 o 4.")


def procedimiento_principal():
    """Punto de entrada general solicitado por la tarea."""
    print("========================================")
    print("TAREA 2 - PERCEPTRON MULTICAPA")
    print("========================================")
    print("1. Crear un nuevo perceptron multicapa")
    print("2. Cargar perceptron multicapa de un archivo")

    while True:
        opcion = input("Seleccione 1 o 2: ").strip()

        if opcion == "1":
            red = crear_red_desde_consola()
            break
        elif opcion == "2":
            red = cargar_red_desde_consola()
            break
        else:
            print("Opcion invalida. Escriba 1 o 2.")

    menu_red(red)


if __name__ == "__main__":
    procedimiento_principal()
