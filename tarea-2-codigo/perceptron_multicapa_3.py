# Nombre del integrante: Diego Mendez
# Cédula del integrante: 30830746

# haga su tarea aqui
import matplotlib.pyplot as plt

TASA_APRENDIZAJE = 0.5

COLORES_CLASES = ["#3b6fb6", "#e08a1e", "#2e9e5b", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#17becf"]


def leer_csv(ruta):
    with open(ruta, "r", encoding="utf-8-sig") as archivo:
        lineas = archivo.readlines()

    filas = []
    for linea in lineas:
        linea = linea.strip()
        if not linea:
            continue
        partes = linea.split(",")
        try:
            valores = [float(p) for p in partes]
        except ValueError:
            continue
        filas.append(valores)

    return filas


def carpeta_del_script():
    """Devuelve la carpeta donde vive este archivo .py"""
    ruta = __file__
    pos = max(ruta.rfind("/"), ruta.rfind("\\"))
    if pos == -1:
        return "."
    return ruta[:pos]


def abrir_csv(entrada):
    try:
        return leer_csv(entrada)
    except FileNotFoundError:
        pass

    pos = max(entrada.rfind("/"), entrada.rfind("\\"))
    nombre = entrada[pos + 1:]
    try:
        return leer_csv(carpeta_del_script() + "/assets/" + nombre)
    except FileNotFoundError:
        return None


def separar_datos(filas):
    if not filas:
        raise ValueError("El archivo esta vacio o no tiene datos numericos.")

    largo = len(filas[0])
    if largo < 2:
        raise ValueError("Cada fila necesita al menos una entrada y una clase (2 columnas).")

    entradas = []
    clases = []
    for fila in filas:
        if len(fila) != largo:
            raise ValueError("Todas las filas deben tener el mismo numero de columnas.")
        clase = fila[-1]
        if not (-1e9 < clase < 1e9) or clase != int(clase):
            raise ValueError("La ultima columna debe ser la clase, como numero entero "
                             "(por ejemplo 0 y 1, 1 a 4, o -1 y 1).")
        entradas.append(fila[:-1])
        clases.append(int(clase))

    return entradas, clases


def configuracion_requerida(entradas, clases):
    """Neuronas de entrada y de salida que pide un archivo de datos: (u, v)."""
    return len(entradas[0]), len(set(clases))


def asignar_indices(clases, v):
    
    etiquetas = sorted(set(clases))
    k = len(etiquetas)

    if k > v:
        raise ValueError(f"El archivo tiene {k} clases distintas y la red solo {v} salidas.")

    if k == v:
        nombres = etiquetas
    elif v == 2 and set(etiquetas) <= {-1, 1}:
        nombres = [-1, 1]
    elif etiquetas[0] >= 1 and etiquetas[-1] <= v:
        nombres = list(range(1, v + 1))
    elif etiquetas[0] >= 0 and etiquetas[-1] <= v - 1:
        nombres = list(range(v))
    else:
        raise ValueError(f"No se puede saber a que salida corresponde cada clase "
                         f"({', '.join(str(e) for e in etiquetas)}) con v={v} salidas.")

    posicion = {nombre: i for i, nombre in enumerate(nombres)}
    return [posicion[c] for c in clases], nombres


def validar_configuracion(red, entradas, clases):
  
    u, v = configuracion_requerida(entradas, clases)
    etiquetas = ", ".join(str(e) for e in sorted(set(clases)))
    if red["u"] != u or red["v"] < v:
        return (f"La red no corresponde con el archivo: la red tiene u={red['u']} entradas y "
                f"v={red['v']} salidas, pero el archivo tiene {u} entradas y {v} clases distintas "
                f"({etiquetas}). Configure la red con u={u} neuronas de entrada y v={v} de salida.")
    try:
        asignar_indices(clases, red["v"])
    except ValueError as error:
        return str(error)
    return None


GENERADOR = {"estado": 3083074630}


def fijar_semilla(semilla):
    GENERADOR["estado"] = semilla % (2 ** 64)


def aleatorio():
    """Numero pseudoaleatorio uniforme en [0, 1)."""
    GENERADOR["estado"] = (6364136223846793005 * GENERADOR["estado"]
                           + 1442695040888963407) % (2 ** 64)
    return (GENERADOR["estado"] >> 11) / (2 ** 53)


def suma_ponderada(entradas, pesos, sesgo):
    """z = sesgo + w1*x1 + w2*x2 + ... + wn*xn"""
    total = sesgo
    for x, w in zip(entradas, pesos):
        total += x * w
    return total


def sigmoide(z):
    """Sigmoide: 1 / (1 + e^-z). Rango de salida: (0, 1)"""
    if z < -700:
        return 0.0
    return 1 / (1 + 2.718281828459045 ** (-z))


def crear_red(u, v, L, b):
    tamanos = [u] + [b] * L + [v]
    pesos = []
    sesgos = []
    for n_entrada, n_salida in zip(tamanos[:-1], tamanos[1:]):
        # Pesos iniciales aleatorios y pequenos: si todos fueran
        # iguales, todas las neuronas de una capa aprenderian lo mismo.
        limite = (6.0 / (n_entrada + n_salida)) ** 0.5
        capa = []
        for _ in range(n_salida):
            capa.append([(2 * aleatorio() - 1) * limite for _ in range(n_entrada)])
        pesos.append(capa)
        sesgos.append([0.0] * n_salida)

    return {"u": u, "v": v, "L": L, "b": b, "epocas": 0, "pesos": pesos, "sesgos": sesgos}


def alimentar(red, entrada):
    """
    Alimentacion hacia delante
    """
    activaciones = [entrada]
    for pesos_capa, sesgos_capa in zip(red["pesos"], red["sesgos"]):
        previa = activaciones[-1]
        activaciones.append([sigmoide(suma_ponderada(previa, w, s))
                             for w, s in zip(pesos_capa, sesgos_capa)])
    return activaciones


def predecir_clase(red, entrada):
    """Clase predicha: la neurona de salida mas activa."""
    salida = alimentar(red, entrada)[-1]
    return salida.index(max(salida))


def retropropagar(red, entrada, clase, tasa):
    activaciones = alimentar(red, entrada)
    salida = activaciones[-1]
    objetivo = [1.0 if j == clase else 0.0 for j in range(red["v"])]
    error = 0.5 * sum((t - a) ** 2 for t, a in zip(objetivo, salida))

    # delta de la capa de salida
    deltas = [(a - t) * a * (1 - a) for a, t in zip(salida, objetivo)]

    for k in range(len(red["pesos"]) - 1, -1, -1):
        previa = activaciones[k]            # lo que entra a la capa k
        pesos_capa = red["pesos"][k]

        # delta de la capa anterior: se calcula ANTES de tocar los pesos de esta
        if k > 0:
            deltas_previa = []
            for i, a in enumerate(previa):
                suma = 0.0
                for j, d in enumerate(deltas):
                    suma += pesos_capa[j][i] * d
                deltas_previa.append(suma * a * (1 - a))

        for j, d in enumerate(deltas):
            for i, a in enumerate(previa):
                pesos_capa[j][i] -= tasa * d * a
            red["sesgos"][k][j] -= tasa * d

        if k > 0:
            deltas = deltas_previa

    return error


def entrenar(red, entradas, clases, epocas, tasa):
    errores = []
    n = len(entradas)
    indices = list(range(n))
    paso = max(1, epocas // 10)

    for e in range(1, epocas + 1):
        # Se baraja el orden en cada epoca (Fisher-Yates)
        for i in range(n - 1, 0, -1):
            j = int(aleatorio() * (i + 1))
            indices[i], indices[j] = indices[j], indices[i]

        total = 0.0
        for i in indices:
            total += retropropagar(red, entradas[i], clases[i], tasa)

        errores.append(total / n)
        red["epocas"] += 1
        if e == 1 or e % paso == 0:
            print(f"  Epoca {red['epocas']}: error medio = {errores[-1]:.5f}")

    return errores


def evaluar(red, entradas, clases):
    """Devuelve (predichos, matriz de confusion, aciertos). Filas = real, columnas = predicho."""
    v = red["v"]
    predichos = [predecir_clase(red, x) for x in entradas]
    matriz = [[0] * v for _ in range(v)]
    aciertos = 0
    for real, predicho in zip(clases, predichos):
        matriz[real][predicho] += 1
        if real == predicho:
            aciertos += 1
    return predichos, matriz, aciertos


def guardar_red(red, ruta):
    """
    Formato: encabezado, 5 filas de hiperparametros (u, v, L, b, e; el valor va
    en la columna 2) y una fila por neurona: tipo, capa, posicion, c pesos
    (c = max(u, b)) y al final el sesgo.
    """
    u, v, L, b = red["u"], red["v"], red["L"], red["b"]
    c = max(u, b)

    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("tipo,capa,pos," + ",".join(f"w{i}" for i in range(c)) + ",sesgo\n")

        for letra, valor in (("u", u), ("v", v), ("L", L), ("b", b), ("e", red["epocas"])):
            archivo.write(f"{letra},{valor}" + "," * (c + 2) + "\n")

        for k, capa in enumerate(red["pesos"]):
            tipo = "o" if k == L else "h"
            for j, pesos_neurona in enumerate(capa):
                celdas = [str(w) for w in pesos_neurona] + [""] * (c - len(pesos_neurona))
                archivo.write(f"{tipo},{k + 1},{j}," + ",".join(celdas)
                              + f",{red['sesgos'][k][j]}\n")


def cargar_red(ruta):
    """
    Lee una red guardada. Devuelve None si el archivo no existe y lanza
    ValueError si su contenido no es una red valida. Ignora cualquier fila
    cuya primera celda no sea u, v, L, b, e, h u o
    """
    candidatos = [ruta]
    pos = max(ruta.rfind("/"), ruta.rfind("\\"))
    nombre = ruta[pos + 1:]
    candidatos.append(carpeta_del_script() + "/" + nombre)
    candidatos.append(carpeta_del_script() + "/assets/" + nombre)

    lineas = None
    for candidato in candidatos:
        try:
            with open(candidato, "r", encoding="utf-8-sig") as archivo:
                lineas = archivo.readlines()
            break
        except FileNotFoundError:
            continue
    if lineas is None:
        return None

    hiper = {}
    filas_neuronas = []
    for linea in lineas:
        celdas = [c.strip() for c in linea.strip().split(",")]
        tipo = celdas[0]
        if tipo in ("u", "v", "L", "b", "e"):
            valor = next((c for c in celdas[1:] if c != ""), None)
            try:
                hiper[tipo] = int(float(valor))
            except (TypeError, ValueError):
                raise ValueError(f"El hiperparametro '{tipo}' no tiene un valor numerico valido.")
        elif tipo in ("h", "o"):
            filas_neuronas.append(celdas)

    for letra in ("u", "v", "L", "b"):
        if letra not in hiper:
            raise ValueError(f"Falta el hiperparametro '{letra}' en el archivo de la red.")
    u, v, L, b = hiper["u"], hiper["v"], hiper["L"], hiper["b"]
    if u < 1 or v < 1 or L < 0 or (L > 0 and b < 1):
        raise ValueError("Los hiperparametros del archivo no son validos.")

    c = max(u, b)
    tamanos = [u] + [b] * L + [v]
    pesos = [[None] * n for n in tamanos[1:]]
    sesgos = [[0.0] * n for n in tamanos[1:]]

    for celdas in filas_neuronas:
        try:
            tipo = celdas[0]
            capa = int(celdas[1])
            posicion = int(celdas[2])
        except (IndexError, ValueError):
            raise ValueError("Hay una fila de neurona con capa o posicion invalida.")

        if capa < 1 or capa > L + 1 or posicion < 0 or posicion >= tamanos[capa]:
            raise ValueError(f"La neurona {tipo},{capa},{posicion} no existe en esta arquitectura.")
        if (tipo == "o") != (capa == L + 1):
            raise ValueError(f"La neurona {tipo},{capa},{posicion} tiene un tipo incorrecto para su capa.")

        n_entrada = tamanos[capa - 1]
        valores = celdas[3:3 + n_entrada]
        try:
            if len(valores) < n_entrada:
                raise ValueError
            pesos[capa - 1][posicion] = [float(x) for x in valores]
            if len(celdas) > 3 + c and celdas[3 + c] != "":
                sesgos[capa - 1][posicion] = float(celdas[3 + c])
        except ValueError:
            raise ValueError(f"La neurona {tipo},{capa},{posicion} no tiene {n_entrada} pesos validos.")

    for k, capa in enumerate(pesos):
        if any(neurona is None for neurona in capa):
            raise ValueError(f"Faltan neuronas en la capa {k + 1} del archivo de la red.")

    return {"u": u, "v": v, "L": L, "b": b, "epocas": hiper.get("e", 0),
            "pesos": pesos, "sesgos": sesgos}


def pedir_entero(mensaje, minimo):
    while True:
        try:
            valor = int(input(mensaje))
        except ValueError:
            print("  -> Ingresa un numero entero.")
            continue
        if valor < minimo:
            print(f"  -> Debe ser al menos {minimo}.")
            continue
        return valor


def crear_red_interactiva():
    print("\n--- Nueva red ---")
    u = pedir_entero("Numero de neuronas de entrada: ", 1)
    v = pedir_entero("Numero de neuronas de salida (una por clase, minimo 2): ", 2)
    L = pedir_entero("Numero de capas ocultas: ", 0)
    b = 0
    if L > 0:
        b = pedir_entero("Numero de neuronas por capa oculta: ", 1)
    return crear_red(u, v, L, b)


def cargar_red_interactiva():
    ruta = input("Archivo de la red: ").strip()
    try:
        red = cargar_red(ruta)
    except ValueError as error:
        print(error)
        return None
    if red is None:
        print("No se encontro el archivo")
    return red


def imprimir_hiperparametros(red):
    print("\n--- Hiperparametros ---")
    print(f"  Neuronas de entrada (u):       {red['u']}")
    print(f"  Neuronas de salida (v):        {red['v']}")
    print(f"  Capas ocultas (L):             {red['L']}")
    print(f"  Neuronas por capa oculta (b):  {red['b']}")
    print(f"  Epocas entrenadas (e):         {red['epocas']}")
    print(f"  Tasa de aprendizaje:           {TASA_APRENDIZAJE}")


def pedir_datos(red):

    ruta = input("Archivo de datos: ").strip()
    filas = abrir_csv(ruta)
    if filas is None:
        print("No se encontro el archivo")
        return None

    try:
        entradas, clases = separar_datos(filas)
    except ValueError as error:
        print(error)
        return None

    mensaje = validar_configuracion(red, entradas, clases)
    if mensaje:
        print(mensaje)
        return None

    indices, nombres = asignar_indices(clases, red["v"])
    return entradas, indices, nombres


def dispersion_por_grupos(x1, x2, grupos, estilos, x3=None):
    """Un scatter por grupo, para que cada uno tenga su entrada en la leyenda."""
    for valor, (color, etiqueta) in estilos.items():
        gx = [a for a, g in zip(x1, grupos) if g == valor]
        gy = [b for b, g in zip(x2, grupos) if g == valor]
        if x3 is None:
            plt.scatter(gx, gy, c=color, label=etiqueta, s=45, alpha=0.8,
                        edgecolors="white", linewidths=0.6)
        else:
            gz = [c for c, g in zip(x3, grupos) if g == valor]
            plt.gca().scatter(gx, gy, gz, c=color, label=etiqueta, s=45, alpha=0.8,
                              edgecolors="white", linewidths=0.6)


def coordenadas(entradas):
    """x1, x2 y x3 (esta ultima solo si hay 3 o mas entradas)."""
    n_entradas = len(entradas[0])
    x1 = [fila[0] for fila in entradas]
    x2 = [fila[1] if n_entradas > 1 else 0.0 for fila in entradas]
    x3 = [fila[2] for fila in entradas] if n_entradas > 2 else None
    return x1, x2, x3


def etiquetar_ejes(n_entradas):
    plt.xlabel("x1")
    plt.ylabel("x2" if n_entradas > 1 else "")
    if n_entradas > 2:
        plt.gca().set_zlabel("x3")


def graficar_datos_crudos(entradas, clases, nombres):
    """Dibuja en el eje actual los datos del archivo, coloreados por su clase real."""
    x1, x2, x3 = coordenadas(entradas)
    estilos = {k: (COLORES_CLASES[k % len(COLORES_CLASES)], f"Clase {nombre}")
               for k, nombre in enumerate(nombres)}
    dispersion_por_grupos(x1, x2, clases, estilos, x3)
    etiquetar_ejes(len(entradas[0]))
    plt.title("Datos del archivo")
    plt.legend(loc="best", fontsize=8)


def graficar_matriz(matriz, nombres):
    """Matriz de confusion con matshow, dibujada en el eje actual."""
    v = len(matriz)
    maximo = max(max(fila) for fila in matriz)
    plt.matshow(matriz, fignum=0, cmap="Blues")
    for i in range(v):
        for j in range(v):
            color = "white" if maximo > 0 and matriz[i][j] > maximo / 2 else "black"
            plt.text(j, i, str(matriz[i][j]), ha="center", va="center", color=color)
    plt.xticks(range(v), [str(n) for n in nombres])
    plt.yticks(range(v), [str(n) for n in nombres])
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de confusion")


def graficar_resultados(entradas, esperados, predichos, aciertos, nombres,
                        errores=None, matriz=None, epoca_inicial=1):
    """
    Tres graficos: datos del archivo, predicciones correctas/incorrectas y
    el error (linea por epoca si se entreno, matriz de confusion si se probo).
    """
    n_entradas = len(entradas[0])
    x1, x2, x3 = coordenadas(entradas)
    en_3d = n_entradas > 2

    coincidencia = ["ok" if e == p else "error" for e, p in zip(esperados, predichos)]
    estilo_coinc = {"ok": ("#2e9e5b", "Correcta"), "error": ("#d64545", "Incorrecta")}

    plt.figure(figsize=(16, 5))

    # 1. Datos del archivo
    plt.subplot(1, 3, 1, projection="3d" if en_3d else None)
    graficar_datos_crudos(entradas, esperados, nombres)
    plt.grid(True, alpha=0.25)

    # 2. Correcto / incorrecto
    plt.subplot(1, 3, 2, projection="3d" if en_3d else None)
    dispersion_por_grupos(x1, x2, coincidencia, estilo_coinc, x3)
    etiquetar_ejes(n_entradas)
    plt.title(f"Predicciones: {aciertos}/{len(entradas)} correctas")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)

    # 3. Error
    plt.subplot(1, 3, 3)
    if errores is not None:
        epocas = list(range(epoca_inicial, epoca_inicial + len(errores)))
        plt.plot(epocas, errores, color="#3b6fb6", linewidth=1.8)
        plt.xlabel("Epoca")
        plt.ylabel("Error medio")
        plt.title("Error por epoca")
        plt.grid(True, alpha=0.25)
    else:
        graficar_matriz(matriz, nombres)

    if n_entradas > 3:
        plt.suptitle("Solo se grafican las primeras 3 dimensiones de entrada")

    plt.tight_layout()
    plt.show()

def opcion_entrenar(red):
    datos = pedir_datos(red)
    if datos is None:
        return False
    entradas, clases, nombres = datos

    epocas = pedir_entero("Numero de epocas: ", 1)
    epoca_inicial = red["epocas"] + 1

    print("\nEntrenando...")
    errores = entrenar(red, entradas, clases, epocas, TASA_APRENDIZAJE)

    predichos, matriz, aciertos = evaluar(red, entradas, clases)
    print(f"\nAciertos sobre los datos de entrenamiento: {aciertos}/{len(clases)} "
          f"({100 * aciertos / len(clases):.1f}%)")
    imprimir_hiperparametros(red)

    graficar_resultados(entradas, clases, predichos, aciertos, nombres,
                        errores=errores, epoca_inicial=epoca_inicial)
    return True


def opcion_probar(red):
    datos = pedir_datos(red)
    if datos is None:
        return
    entradas, clases, nombres = datos

    predichos, matriz, aciertos = evaluar(red, entradas, clases)
    print(f"\nAciertos: {aciertos}/{len(clases)} ({100 * aciertos / len(clases):.1f}%)")
    imprimir_hiperparametros(red)

    graficar_resultados(entradas, clases, predichos, aciertos, nombres, matriz=matriz)


def opcion_guardar(red):
    ruta = input("Nombre del archivo para guardar la red (ej. red.csv): ").strip()
    if ruta == "":
        print("Nombre vacio, no se guardo.")
        return False
    try:
        guardar_red(red, ruta)
    except OSError as error:
        print(f"No se pudo guardar: {error}")
        return False
    print(f"Red guardada en {ruta}")
    imprimir_hiperparametros(red)
    return True


def main():
    print("=== Perceptron multicapa ===")

    red = None
    while red is None:
        print("\n1) Crear un nuevo perceptron multicapa")
        print("2) Cargarlo de un archivo")
        opcion = input("Elige una opcion (1/2): ").strip()
        if opcion == "1":
            red = crear_red_interactiva()
        elif opcion == "2":
            red = cargar_red_interactiva()
        else:
            print("  -> Opcion invalida.")
    imprimir_hiperparametros(red)

    sin_guardar = False
    while True:
        print("\n--- Menu ---")
        print("1) Entrenar con retropropagacion")
        print("2) Probar con alimentacion hacia delante")
        print("3) Guardar la red")
        print("4) Salir")
        opcion = input("Elige una opcion (1-4): ").strip()

        if opcion == "1":
            if opcion_entrenar(red):
                sin_guardar = True
        elif opcion == "2":
            opcion_probar(red)
        elif opcion == "3":
            if opcion_guardar(red):
                sin_guardar = False
        elif opcion == "4":
            if sin_guardar:
                respuesta = input("Hay entrenamiento sin guardar. ¿Salir de todos modos? (s/n): ")
                if respuesta.strip().lower() != "s":
                    continue
            break
        else:
            print("  -> Opcion invalida.")

    print("Fin")


if __name__ == "__main__":
    main()