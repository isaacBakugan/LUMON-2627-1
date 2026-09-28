# Nombre del integrante: Diego Mendez
# Cédula del integrante: 30830746


import matplotlib.pyplot as plt

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
    """
    Intenta leer el CSV tal como se escribio. Si no lo encuentra,
    lo busca en la carpeta assets/ que está junto a este script
    """
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
    
def suma_ponderada(entradas, pesos, sesgo):
    """z = sesgo + w1*x1 + w2*x2 + ... + wn*xn"""
    total = sesgo
    for x, w in zip(entradas, pesos):
        total += x * w
    return total


def escalon(z):
    """Funcion escalon (Heaviside). Rango de salida: {0, 1}"""
    return 1.0 if z >= 0 else 0.0


def signo(z):
    """Funcion signo (bipolar). Rango de salida: {-1, 1}"""
    return 1.0 if z >= 0 else -1.0


FUNCIONES_ACTIVACION = {
    "1": ("Heaviside (salida 0 / 1)", escalon),
    "2": ("Bipolar (salida -1 / 1)", signo),
}


def clase_objetivo(y, funcion):
    pertenece_a_clase_1 = y >= 0.5
    if funcion is escalon:
        return 1.0 if pertenece_a_clase_1 else 0.0
    return 1.0 if pertenece_a_clase_1 else -1.0

def pedir_float(mensaje):
    while True:
        try:
            return float(input(mensaje))
        except ValueError:
            print("  -> Ingresa un numero valido.")


def pedir_pesos(n_entradas):
    print("\nIngresa los pesos del perceptron:")
    sesgo = pedir_float("  Sesgo (bias): ")
    pesos = []
    for i in range(n_entradas):
        pesos.append(pedir_float(f"  Peso w{i + 1} (para x{i + 1}): "))
    return sesgo, pesos


def pedir_funcion_activacion():
    print("\nFunciones de activacion:")
    for clave, (nombre, _) in FUNCIONES_ACTIVACION.items():
        print(f"  {clave}) {nombre}")
    while True:
        opcion = input("Elige una funcion de activacion (1/2): ").strip()
        if opcion in FUNCIONES_ACTIVACION:
            nombre, funcion = FUNCIONES_ACTIVACION[opcion]
            print(f"  -> Usando: {nombre}")
            return funcion
        print("  -> Opcion invalida.")

def graficar_datos_crudos(filas, n_entradas):
    """Antes de aplicar el perceptron"""
    x1 = [fila[0] for fila in filas]
    x2 = [fila[1] if n_entradas > 1 else 0.0 for fila in filas]
    y = [fila[-1] for fila in filas]

    plt.figure(figsize=(5, 5))
    plt.scatter(x1, x2, c=y, cmap="bwr", edgecolors="k")
    plt.xlabel("x1")
    plt.ylabel("x2" if n_entradas > 1 else "")
    titulo = "Datos originales (color = valor esperado)"
    if n_entradas > 2:
        titulo += "\n(solo se muestran las primeras 2 dimensiones)"
    plt.title(titulo)
    plt.show()


def graficar_resultados(filas, esperados, predichos, n_entradas):
    x1 = [fila[0] for fila in filas]
    x2 = [fila[1] if n_entradas > 1 else 0.0 for fila in filas]

    colores_esperado = ["tab:blue" if e <= 0 else "tab:orange" for e in esperados]
    colores_predicho = ["tab:blue" if p <= 0 else "tab:orange" for p in predichos]
    colores_acierto = ["green" if e == p else "red" for e, p in zip(esperados, predichos)]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(x1, x2, c=colores_esperado, edgecolors="k")
    plt.title("Valor esperado")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.subplot(1, 3, 2)
    plt.scatter(x1, x2, c=colores_predicho, edgecolors="k")
    plt.title("Valor predicho")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.subplot(1, 3, 3)
    plt.scatter(x1, x2, c=colores_acierto, edgecolors="k")
    plt.title("Coincidencias (verde) / Errores (rojo)")
    plt.xlabel("x1")
    plt.ylabel("x2")

    if n_entradas > 2:
        plt.suptitle("Solo se grafican las primeras 2 dimensiones de entrada")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Programa principal
# ---------------------------------------------------------------------------
def main():
    print("=== Perceptron interactivo ===")
    ruta = input("Ruta del archivo CSV: ").strip()
    filas = abrir_csv(ruta)

    if filas is None:
        print("No se encontro el archivo")
        return

    if not filas:
        print("No es compatible")
        return

    n_columnas = len(filas[0])
    n_entradas = n_columnas - 1
    print(f"Se cargaron {len(filas)} ejemplos con {n_entradas} entrada(s) y 1 salida esperada.")

    graficar_datos_crudos(filas, n_entradas)

    funcion = pedir_funcion_activacion()

    continuar = True
    while continuar:
        sesgo, pesos = pedir_pesos(n_entradas)

        esperados = []
        predichos = []
        aciertos = 0
        for fila in filas:
            entradas = fila[:-1]
            y_real = fila[-1]

            z = suma_ponderada(entradas, pesos, sesgo)
            prediccion = funcion(z)
            objetivo = clase_objetivo(y_real, funcion)

            esperados.append(objetivo)
            predichos.append(prediccion)
            if prediccion == objetivo:
                aciertos += 1

        porcentaje = 100 * aciertos / len(filas)
        print(f"\nAciertos: {aciertos}/{len(filas)} ({porcentaje:.1f}%)")

        graficar_resultados(filas, esperados, predichos, n_entradas)

        respuesta = input("\n¿Probar con otros pesos? (s/n): ").strip().lower()
        continuar = respuesta == "s"

    print("Fin")


if __name__ == "__main__":
    main()
