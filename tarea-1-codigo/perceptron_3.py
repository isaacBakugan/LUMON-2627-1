# Nombre del integrante: Diego Mendez
# Cédula del integrante: 30830746


import matplotlib.pyplot as plt

def leer_csv(ruta):
    """Lee un CSV a mano: devuelve una lista de filas (listas de floats)."""
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

def suma_ponderada(entradas, pesos, sesgo):
    """z = sesgo + w1*x1 + w2*x2 + ... + wn*xn"""
    total = sesgo
    for x, w in zip(entradas, pesos):
        total += x * w
    return total


UMBRAL = 0.5


def sigmoide(z):
    """Sigmoide: 1 / (1 + e^-z). Rango de salida: (0, 1)"""
    if z < -700:  
        return 0.0
    return 1 / (1 + 2.718281828459045 ** (-z))


def relu(z):
    """ReLU: max(0, z). Rango de salida: [0, infinito)"""
    return z if z > 0 else 0.0


FUNCIONES_ACTIVACION = {
    "1": ("Sigmoide", sigmoide),
    "2": ("ReLU", relu),
}


def clase_de(valor):
    """Convierte un valor a clase 0/1: clase 1 si valor >= UMBRAL."""
    return 1.0 if valor >= UMBRAL else 0.0

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
    plt.scatter(x1, x2, c=y, cmap="bwr", s=45, alpha=0.8,
                edgecolors="white", linewidths=0.6)
    plt.colorbar(label="Valor esperado")
    plt.xlabel("x1")
    plt.ylabel("x2" if n_entradas > 1 else "")
    plt.grid(True, alpha=0.25)
    titulo = "Datos originales"
    if n_entradas > 2:
        titulo += "\n(solo se muestran las primeras 2 dimensiones)"
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def dispersion_por_grupos(x1, x2, grupos, estilos):
    """Un scatter por grupo, para que cada uno tenga su entrada en la leyenda."""
    for valor, (color, etiqueta) in estilos.items():
        gx = [a for a, g in zip(x1, grupos) if g == valor]
        gy = [b for b, g in zip(x2, grupos) if g == valor]
        plt.scatter(gx, gy, c=color, label=etiqueta, s=45, alpha=0.8,
                    edgecolors="white", linewidths=0.6)


def dibujar_frontera(sesgo, pesos, z_umbral, limites):
    """Dibuja la recta donde sesgo + w1*x1 + w2*x2 = z_umbral (solo 2 entradas)."""
    xmin, xmax, ymin, ymax = limites
    w1, w2 = pesos
    if w2 != 0:
        y_a = (z_umbral - sesgo - w1 * xmin) / w2
        y_b = (z_umbral - sesgo - w1 * xmax) / w2
        plt.plot([xmin, xmax], [y_a, y_b], "k--", linewidth=1.3, label="Frontera")
    elif w1 != 0:
        x_c = (z_umbral - sesgo) / w1
        plt.plot([x_c, x_c], [ymin, ymax], "k--", linewidth=1.3, label="Frontera")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)


def graficar_resultados(filas, esperados, predichos, n_entradas,
                        sesgo, pesos, z_umbral, aciertos):
    x1 = [fila[0] for fila in filas]
    x2 = [fila[1] if n_entradas > 1 else 0.0 for fila in filas]
    margen = 0.1
    limites = (min(x1) - margen, max(x1) + margen,
               min(x2) - margen, max(x2) + margen)

    clases = {0.0: ("#3b6fb6", "Clase 0"), 1.0: ("#e08a1e", "Clase 1")}
    coincidencia = ["ok" if e == p else "error" for e, p in zip(esperados, predichos)]
    estilo_coinc = {"ok": ("#2e9e5b", "Coincide"), "error": ("#d64545", "No coincide")}
    dibuja_frontera = n_entradas == 2

    plt.figure(figsize=(15, 5))
    paneles = [
        ("Valor esperado", esperados, clases),
        ("Valor predicho", predichos, clases),
        (f"Coincidencias: {aciertos}/{len(filas)}", coincidencia, estilo_coinc),
    ]
    for i, (titulo, grupos, estilos) in enumerate(paneles, start=1):
        plt.subplot(1, 3, i)
        dispersion_por_grupos(x1, x2, grupos, estilos)
        if dibuja_frontera:
            dibujar_frontera(sesgo, pesos, z_umbral, limites)
        plt.title(titulo)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best", fontsize=8)

    if n_entradas > 2:
        plt.suptitle("Solo se grafican las primeras 2 dimensiones de entrada")

    plt.tight_layout()
    plt.show()

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
            prediccion = clase_de(funcion(z))
            objetivo = clase_de(y_real)

            esperados.append(objetivo)
            predichos.append(prediccion)
            if prediccion == objetivo:
                aciertos += 1

        porcentaje = 100 * aciertos / len(filas)
        print(f"\nAciertos: {aciertos}/{len(filas)} ({porcentaje:.1f}%)")

        # Con sigmoide la clase 1 empieza en z = 0; con ReLU, en z = UMBRAL
        z_umbral = 0.0 if funcion is sigmoide else UMBRAL
        graficar_resultados(filas, esperados, predichos, n_entradas,
                            sesgo, pesos, z_umbral, aciertos)

        respuesta = input("\n¿Probar con otros pesos? (s/n): ").strip().lower()
        continuar = respuesta == "s"

    print("Fin")


if __name__ == "__main__":
    main()