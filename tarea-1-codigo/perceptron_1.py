# Nombre del integrante: Alexander Gabriel Ramírez Duarte
# Cédula del integrante: 29611077

def convertir_a_numero(texto):
    """Convierte un texto a float.

    Esta funcion existe para centralizar la conversion de los valores
    leidos desde el CSV y desde la consola.
    """
    return float(texto.strip())


def cargar_csv(ruta_archivo):
    """Carga manualmente un CSV y separa entradas de salida esperada.

    Cada fila valida debe tener n columnas:
      - las primeras n-1 son entradas del perceptron
      - la ultima columna es la salida esperada

    El archivo puede tener encabezado. Si la primera fila no se puede
    convertir a numeros, se interpreta como encabezado y se omite.
    """
    datos = []
    encabezados = None
    cantidad_columnas = None

    # utf-8-sig elimina automaticamente el BOM que algunos CSV traen
    # al principio del primer encabezado.
    with open(ruta_archivo, "r", encoding="utf-8-sig") as archivo:
        numero_linea = 0

        for linea in archivo:
            numero_linea += 1
            linea = linea.strip()

            # Ignoramos lineas vacias para evitar errores innecesarios.
            if linea == "":
                continue

            partes = linea.split(",")

            try:
                fila_numerica = []
                for parte in partes:
                    fila_numerica.append(convertir_a_numero(parte))
            except ValueError:
                # Solo aceptamos una fila no numerica si aparece antes
                # de cualquier dato: la tratamos como encabezado.
                if len(datos) == 0 and encabezados is None:
                    encabezados = []
                    for parte in partes:
                        encabezados.append(parte.strip())
                    continue

                raise ValueError(
                    "La linea " + str(numero_linea) +
                    " contiene un valor que no es numerico."
                )

            # La primera fila numerica define cuantas columnas esperamos.
            if cantidad_columnas is None:
                cantidad_columnas = len(fila_numerica)

                if cantidad_columnas < 2:
                    raise ValueError(
                        "El CSV debe tener al menos una entrada y una salida esperada."
                    )

            # Todas las filas deben conservar la misma cantidad de columnas.
            if len(fila_numerica) != cantidad_columnas:
                raise ValueError(
                    "La linea " + str(numero_linea) +
                    " no tiene la misma cantidad de columnas que las demas."
                )

            datos.append(fila_numerica)

    if len(datos) == 0:
        raise ValueError("El archivo CSV no contiene filas de datos numericos.")

    # Si no habia encabezado, generamos nombres sencillos para los ejes.
    if encabezados is None or len(encabezados) != cantidad_columnas:
        encabezados = []
        indice = 1
        while indice < cantidad_columnas:
            encabezados.append("x" + str(indice))
            indice += 1
        encabezados.append("y")

    entradas = []
    salidas_esperadas = []

    for fila in datos:
        entradas.append(fila[:-1])
        salidas_esperadas.append(fila[-1])

    return entradas, salidas_esperadas, encabezados

def suma_ponderada(entradas, pesos, sesgo):
    """Calcula la suma del perceptron desde cero.

    Formula:
        z = sesgo + x1*w1 + x2*w2 + ... + xm*wm

    No usamos sum() a proposito: hacemos visible cada acumulacion para que
    la logica matematica del perceptron quede clara en el codigo.
    """
    resultado = sesgo
    indice = 0

    while indice < len(entradas):
        resultado = resultado + entradas[indice] * pesos[indice]
        indice += 1

    return resultado


def activacion_escalon(valor):
    """Funcion de activacion escalon binario.

    Si la suma ponderada es mayor o igual que cero devuelve 1.
    En caso contrario devuelve 0.
    """
    if valor >= 0:
        return 1.0
    return 0.0


def activacion_fuzzy(valor):
    """Funcion de activacion fuzzy usada para explorar una salida intermedia.

    La funcion tiene tres zonas:
      - valor >=  0.25  -> 1.0
      - valor <= -0.25  -> 0.0
      - entre ambas     -> el valor redondeado a una decima

    Con w1=1, w2=-1 y sesgo=0 esta funcion reproduce exactamente las
    salidas del archivo fuzzy_separables.csv suministrado con la tarea.
    """
    if valor >= 0.25:
        return 1.0

    if valor <= -0.25:
        return 0.0

    return round(valor, 1)


def aplicar_activacion(valor, opcion_activacion):
    """Conecta la opcion escogida por el usuario con su funcion real."""
    if opcion_activacion == "1":
        return activacion_escalon(valor)

    return activacion_fuzzy(valor)


def ejecutar_perceptron(entradas, pesos, sesgo, opcion_activacion):
    """Ejecuta el perceptron para todas las filas del archivo."""
    predicciones = []

    for vector in entradas:
        z = suma_ponderada(vector, pesos, sesgo)
        salida = aplicar_activacion(z, opcion_activacion)
        predicciones.append(salida)

    return predicciones


def valores_coinciden(esperado, predicho):
    """Compara dos salidas usando una tolerancia pequena para floats."""
    return abs(esperado - predicho) < 0.000000001


def calcular_resultados(salidas_esperadas, predicciones):
    """Devuelve coincidencias por fila y cantidad total de aciertos."""
    coincidencias = []
    aciertos = 0

    indice = 0
    while indice < len(salidas_esperadas):
        coincide = valores_coinciden(
            salidas_esperadas[indice],
            predicciones[indice]
        )
        coincidencias.append(coincide)

        if coincide:
            aciertos += 1

        indice += 1

    return coincidencias, aciertos


def preparar_coordenadas(entradas):
    """Obtiene las dos dimensiones que se usaran en los scatter plots.

    - Si existen 2 o mas entradas, usamos x1 y x2.
    - Si solo existe 1 entrada, la graficamos contra y=0.

    De esta forma tambien se cumple el requerimiento de que, si hay mas
    de dos dimensiones de entrada, solo se visualicen las primeras dos.
    """
    coordenadas_x = []
    coordenadas_y = []

    for vector in entradas:
        coordenadas_x.append(vector[0])

        if len(vector) >= 2:
            coordenadas_y.append(vector[1])
        else:
            coordenadas_y.append(0.0)

    return coordenadas_x, coordenadas_y


def configurar_ejes(nombre_x, nombre_y, titulo):
    """Aplica rotulos comunes a cada grafico."""
    plt.xlabel(nombre_x)
    plt.ylabel(nombre_y)
    plt.title(titulo)
    plt.grid(True, alpha=0.25)


def crear_graficos(entradas, salidas_esperadas, predicciones,
                    coincidencias, encabezados):
    """Crea los tres graficos scatter solicitados en el enunciado."""
    coordenadas_x, coordenadas_y = preparar_coordenadas(entradas)

    nombre_x = encabezados[0]
    if len(entradas[0]) >= 2:
        nombre_y = encabezados[1]
    else:
        nombre_y = "eje auxiliar"

    # ------------------------------------------------------------
    # GRAFICO 1: el color de cada punto representa el valor esperado.
    # ------------------------------------------------------------
    plt.figure("1 - Valores esperados")
    puntos_esperados = plt.scatter(
        coordenadas_x,
        coordenadas_y,
        c=salidas_esperadas,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.4
    )
    configurar_ejes(nombre_x, nombre_y, "Valores esperados")
    barra_esperados = plt.colorbar(puntos_esperados)
    barra_esperados.set_label("Salida esperada")

    # ------------------------------------------------------------
    # GRAFICO 2: el color representa la prediccion del perceptron.
    # ------------------------------------------------------------
    plt.figure("2 - Valores predichos")
    puntos_predichos = plt.scatter(
        coordenadas_x,
        coordenadas_y,
        c=predicciones,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.4
    )
    configurar_ejes(nombre_x, nombre_y, "Valores predichos por el perceptron")
    barra_predichos = plt.colorbar(puntos_predichos)
    barra_predichos.set_label("Salida predicha")

    # ------------------------------------------------------------
    # GRAFICO 3: verde = coincide; rojo = no coincide.
    # ------------------------------------------------------------
    colores = []
    for coincide in coincidencias:
        if coincide:
            colores.append("green")
        else:
            colores.append("red")

    plt.figure("3 - Coincidencias")
    plt.scatter(
        coordenadas_x,
        coordenadas_y,
        c=colores,
        edgecolors="black",
        linewidths=0.4
    )
    configurar_ejes(
        nombre_x,
        nombre_y,
        "Coincidencias: verde = correcto, rojo = incorrecto"
    )

    # Creamos los tres graficos primero y luego los mostramos juntos.
    # Al cerrar las ventanas, el programa vuelve a la consola y permite
    # probar otro conjunto de pesos, tal como pide la tarea.
    plt.show()

    # Cerramos las figuras para que la siguiente prueba no reutilice las
    # ventanas anteriores.
    plt.close("all")


def leer_numero(mensaje):
    """Lee un numero por consola y repite hasta recibir un valor valido."""
    while True:
        texto = input(mensaje).strip()

        try:
            return convertir_a_numero(texto)
        except ValueError:
            print("Entrada invalida. Debe escribir un numero, por ejemplo: 1, -0.5 o 2.3")


def leer_pesos(cantidad_entradas):
    """Solicita sesgo y un peso por cada columna de entrada."""
    print("\n--- Configuracion de pesos ---")
    sesgo = leer_numero("Peso del sesgo (bias): ")

    pesos = []
    indice = 0

    while indice < cantidad_entradas:
        peso = leer_numero("Peso w" + str(indice + 1) + ": ")
        pesos.append(peso)
        indice += 1

    return pesos, sesgo


def elegir_activacion():
    """Permite escoger una de las dos funciones de activacion."""
    print("\n--- Funcion de activacion ---")
    print("1. Escalon binario (0 / 1)")
    print("2. Fuzzy (zona intermedia entre -0.25 y 0.25)")

    while True:
        opcion = input("Seleccione 1 o 2: ").strip()

        if opcion == "1" or opcion == "2":
            return opcion

        print("Opcion invalida. Escriba solamente 1 o 2.")


def mostrar_resumen(aciertos, total, pesos, sesgo, opcion_activacion):
    """Muestra en consola el resultado de la prueba actual."""
    porcentaje = (aciertos / total) * 100

    if opcion_activacion == "1":
        nombre_activacion = "Escalon binario"
    else:
        nombre_activacion = "Fuzzy"

    print("\n========================================")
    print("RESULTADO DE LA PRUEBA")
    print("========================================")
    print("Activacion:", nombre_activacion)
    print("Sesgo:", sesgo)
    print("Pesos:", pesos)
    print("Coincidencias:", str(aciertos) + " de " + str(total))
    print("Porcentaje de coincidencia:", round(porcentaje, 2), "%")

    if aciertos == total:
        print("Resultado: todas las salidas coinciden.")
    else:
        print("Resultado: existen puntos que el perceptron no clasifica exactamente.")


def preguntar_si_repetir():
    """Pregunta si el usuario quiere probar otros pesos."""
    while True:
        respuesta = input("\n¿Desea probar otros pesos? (s/n): ").strip().lower()

        if respuesta == "s" or respuesta == "si" or respuesta == "sí":
            return True

        if respuesta == "n" or respuesta == "no":
            return False

        print("Respuesta invalida. Escriba s o n.")


def procedimiento_principal():
    """Coordina todo el flujo de ejecucion solicitado por la tarea."""
    print("========================================")
    print("TAREA 1 - PERCEPTRON INTERACTIVO")
    print("========================================")
    print("El programa espera un CSV con n columnas.")
    print("Las primeras n-1 son entradas y la ultima es la salida esperada.")
    print("\nArchivos de ejemplo incluidos:")
    print("  datos/fuzzy_separables.csv")
    print("  datos/no_separables.csv")

    # Primero cargamos el archivo, como exige el enunciado.
    while True:
        ruta_archivo = input("\nRuta del archivo CSV: ").strip()

        try:
            entradas, salidas_esperadas, encabezados = cargar_csv(ruta_archivo)
            break
        except FileNotFoundError:
            print("No se encontro el archivo. Revise la ruta e intente nuevamente.")
        except ValueError as error:
            print("Error al leer el CSV:", error)

    cantidad_entradas = len(entradas[0])

    print("\nArchivo cargado correctamente.")
    print("Cantidad de vectores:", len(entradas))
    print("Cantidad de entradas por vector:", cantidad_entradas)
    print("Columnas de entrada:", encabezados[:-1])
    print("Columna de salida esperada:", encabezados[-1])

    seguir = True

    # Este ciclo es la parte interactiva: el archivo se mantiene cargado
    # mientras el usuario experimenta con distintos pesos/activaciones.
    while seguir:
        pesos, sesgo = leer_pesos(cantidad_entradas)
        opcion_activacion = elegir_activacion()

        predicciones = ejecutar_perceptron(
            entradas,
            pesos,
            sesgo,
            opcion_activacion
        )

        coincidencias, aciertos = calcular_resultados(
            salidas_esperadas,
            predicciones
        )

        mostrar_resumen(
            aciertos,
            len(salidas_esperadas),
            pesos,
            sesgo,
            opcion_activacion
        )

        crear_graficos(
            entradas,
            salidas_esperadas,
            predicciones,
            coincidencias,
            encabezados
        )

        seguir = preguntar_si_repetir()

    print("\nPrograma finalizado.")


# Este bloque hace que el procedimiento principal se ejecute solamente
# cuando abrimos este archivo directamente con: python perceptron.py
if __name__ == "__main__":
    procedimiento_principal()
