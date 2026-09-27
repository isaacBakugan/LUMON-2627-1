# Nombre del integrante: Mario Suso
# Cédula del integrante: 30.751.403

import matplotlib.pyplot as plt

E = 2.718281828459045
LIMITE_EXPONENTE = 700
ARCHIVOS = ["no_separables.csv", "fuzzy_separables.csv"]
CARPETA_ASSETS = __file__.replace("\\", "/").rsplit("/", 1)[0] + "/assets/"


def escalonada(x: float) -> int:
    if x > 0:
        return 1
    return 0


def sigmoide(x: float) -> float:
    if x < -LIMITE_EXPONENTE:
        return 0.0
    return 1 / (1 + E ** (-x))


def suma(entradas: list[float], pesos: list[float], sesgo: float) -> float:
    total = sesgo
    for i in range(len(entradas)):
        total += entradas[i] * pesos[i]
    return total


def leer_csv(ruta: str) -> tuple[list[list[float]], list[float]]:
    entradas = []
    esperados = []
    columnas = None
    with open(ruta, encoding="utf-8") as archivo:
        lineas = archivo.readlines()
    for linea in lineas[1:]:
        linea = linea.strip()
        if linea == "":
            continue
        valores = [float(v) for v in linea.split(",")]
        if columnas is None:
            columnas = len(valores)
        elif len(valores) != columnas:
            raise ValueError("Filas con distinta cantidad de columnas")
        entradas.append(valores[:-1])
        esperados.append(valores[-1])
    return entradas, esperados


def elegir_ruta() -> str:
    while True:
        print("Archivo CSV:")
        print(f"  1) {ARCHIVOS[0]}")
        print(f"  2) {ARCHIVOS[1]}")
        print("  3) Otro archivo")
        opcion = input("Elige 1, 2 o 3: ").strip()
        if opcion == "1":
            return CARPETA_ASSETS + ARCHIVOS[0]
        if opcion == "2":
            return CARPETA_ASSETS + ARCHIVOS[1]
        if opcion == "3":
            return input("Ruta del archivo: ").strip().strip('"')
        print("Opcion invalida. Intenta de nuevo.")


def pedir_datos() -> tuple[list[list[float]], list[float]]:
    while True:
        try:
            entradas, esperados = leer_csv(elegir_ruta())
        except OSError:
            print("No se encontro el archivo. Intenta de nuevo.")
            continue
        except ValueError:
            print("El archivo tiene valores invalidos o filas incompletas. Intenta con otro.")
            continue
        if len(entradas) == 0:
            print("El archivo no tiene datos. Intenta con otro.")
            continue
        return entradas, esperados


def pedir_numero(mensaje: str) -> float:
    while True:
        texto = input(mensaje).strip()
        try:
            return float(texto)
        except ValueError:
            print("Eso no es un numero valido. Intenta de nuevo.")


def pedir_pesos(cantidad: int) -> list[float]:
    pesos = []
    for i in range(cantidad):
        pesos.append(pedir_numero(f"Peso w{i + 1} (columna x{i + 1}): "))
    return pesos


def pedir_activacion():
    while True:
        print("Funcion de activacion:")
        print("  1) Escalonada")
        print("  2) Sigmoide")
        opcion = input("Elige 1 o 2: ").strip()
        if opcion == "1":
            return escalonada
        if opcion == "2":
            return sigmoide
        print("Opcion invalida. Intenta de nuevo.")


def predecir(entradas: list[list[float]], pesos: list[float], sesgo: float, activacion) -> list[float]:
    pronostico = []
    for fila in entradas:
        pronostico.append(activacion(suma(fila, pesos, sesgo)))
    return pronostico


def a_clase(valor: float) -> int:
    if valor >= 0.5:
        return 1
    return 0


def graficar(entradas: list[list[float]], esperados: list[float], pronostico: list[float]) -> None:
    xs = [fila[0] for fila in entradas]
    ys = [fila[1] if len(fila) > 1 else 0 for fila in entradas]

    colores = []
    aciertos = 0
    for i in range(len(esperados)):
        if a_clase(esperados[i]) == a_clase(pronostico[i]):
            colores.append("green")
            aciertos += 1
        else:
            colores.append("red")

    figura, ejes = plt.subplots(1, 3, figsize=(15, 5))

    puntos = ejes[0].scatter(xs, ys, c=esperados, cmap="coolwarm", vmin=0, vmax=1)
    ejes[0].set_title("Valor esperado")
    figura.colorbar(puntos, ax=ejes[0])

    puntos = ejes[1].scatter(xs, ys, c=pronostico, cmap="coolwarm", vmin=0, vmax=1)
    ejes[1].set_title("Valor predicho")
    figura.colorbar(puntos, ax=ejes[1])

    ejes[2].scatter(xs, ys, c=colores)
    ejes[2].set_title(f"Coincidencias: {aciertos}/{len(esperados)}")

    for eje in ejes:
        eje.set_xlabel("x1")
        eje.set_ylabel("x2")

    plt.tight_layout()
    plt.show()


def quiere_repetir() -> bool:
    while True:
        respuesta = input("Probar con otros pesos? (y/n): ").strip().lower()
        if respuesta == "y":
            return True
        if respuesta == "n":
            return False
        print("Responde y o n.")


if __name__ == "__main__":
    entradas, esperados = pedir_datos()
    while True:
        sesgo = pedir_numero("Sesgo: ")
        pesos = pedir_pesos(len(entradas[0]))
        activacion = pedir_activacion()

        pronostico = predecir(entradas, pesos, sesgo, activacion)
        graficar(entradas, esperados, pronostico)

        if not quiere_repetir():
            break
