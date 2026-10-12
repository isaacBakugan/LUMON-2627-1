# Nombre del integrante: Mario Suso
# Cédula del integrante: 30.751.403

import csv
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt

E = 2.718281828459045
LIMITE_EXPONENTE = 700
TASA_APRENDIZAJE = 0.5
ARCHIVOS = [
    "2-d_2-class_train.csv",
    "2-d_2-class_test.csv",
    "3-d_4-class_train.csv",
    "3-d_4-class_test.csv",
]
CARPETA_PROGRAMA = __file__.replace("\\", "/").rsplit("/", 1)[0] + "/"
CARPETA_ASSETS = CARPETA_PROGRAMA + "assets/"


@dataclass
class Red:
    u: int
    v: int
    L: int
    b: int
    pesos: list[list[list[float]]]
    sesgos: list[list[float]]
    epocas: int = 0


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


def resolver_ruta(ruta: str) -> str:
    ruta = ruta.strip().strip('"')
    if os.path.isabs(ruta) or os.path.exists(ruta):
        return ruta
    return CARPETA_PROGRAMA + ruta


def elegir_ruta() -> str:
    while True:
        print("Archivo CSV:")
        for i in range(len(ARCHIVOS)):
            print(f"  {i + 1}) {ARCHIVOS[i]}")
        print(f"  {len(ARCHIVOS) + 1}) Otro archivo")
        opcion = input(f"Elige 1-{len(ARCHIVOS) + 1}: ").strip()
        if opcion.isdigit() and 1 <= int(opcion) <= len(ARCHIVOS):
            return CARPETA_ASSETS + ARCHIVOS[int(opcion) - 1]
        if opcion == str(len(ARCHIVOS) + 1):
            return resolver_ruta(input("Ruta del archivo: "))
        print("Opcion invalida. Intenta de nuevo.")


def pedir_datos(red: Red) -> tuple[list[list[float]], list[int], list[float]]:
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
        etiquetas = sorted(set(esperados))
        if len(entradas[0]) != red.u or len(etiquetas) != red.v:
            print(
                f"El archivo tiene {len(entradas[0])} entradas y {len(etiquetas)} clases. "
                f"Configure la red con {len(entradas[0])} neuronas de entrada y "
                f"{len(etiquetas)} de salida."
            )
            continue
        return entradas, [etiquetas.index(e) for e in esperados], etiquetas


def pedir_entero(mensaje: str) -> int:
    while True:
        texto = input(mensaje).strip()
        if texto.isdigit() and int(texto) >= 1:
            return int(texto)
        print("Debe ser un numero entero mayor o igual a 1. Intenta de nuevo.")


def crear_red(u: int, v: int, L: int, b: int) -> Red:
    tamanos = [u] + [b] * L + [v]
    pesos = []
    sesgos = []
    for k in range(L + 1):
        pesos.append([[random.uniform(-1, 1) for _ in range(tamanos[k])] for _ in range(tamanos[k + 1])])
        sesgos.append([0.0] * tamanos[k + 1])
    return Red(u, v, L, b, pesos, sesgos)


def adelante(red: Red, fila: list[float]) -> list[list[float]]:
    activaciones = [fila]
    for k in range(len(red.pesos)):
        capa = []
        for j in range(len(red.pesos[k])):
            capa.append(sigmoide(suma(activaciones[k], red.pesos[k][j], red.sesgos[k][j])))
        activaciones.append(capa)
    return activaciones


def a_clase(salida: list[float]) -> int:
    return salida.index(max(salida))


def predecir(red: Red, entradas: list[list[float]]) -> list[int]:
    return [a_clase(adelante(red, fila)[-1]) for fila in entradas]


def error_medio(red: Red, entradas: list[list[float]], clases: list[int]) -> float:
    total = 0.0
    for fila, clase in zip(entradas, clases):
        salida = adelante(red, fila)[-1]
        for j in range(red.v):
            objetivo = 1.0 if j == clase else 0.0
            total += (salida[j] - objetivo) ** 2
    return total / len(entradas)


def entrenar(red: Red, entradas: list[list[float]], clases: list[int], epocas: int) -> list[float]:
    errores = []
    orden = list(range(len(entradas)))
    for _ in range(epocas):
        random.shuffle(orden)
        for i in orden:
            act = adelante(red, entradas[i])
            salida = act[-1]
            delta = []
            for j in range(red.v):
                objetivo = 1.0 if j == clases[i] else 0.0
                delta.append((salida[j] - objetivo) * salida[j] * (1 - salida[j]))
            for k in range(len(red.pesos) - 1, -1, -1):
                delta_anterior = []
                if k > 0:
                    for m in range(len(act[k])):
                        error = sum(red.pesos[k][j][m] * delta[j] for j in range(len(delta)))
                        delta_anterior.append(error * act[k][m] * (1 - act[k][m]))
                for j in range(len(delta)):
                    for m in range(len(act[k])):
                        red.pesos[k][j][m] -= TASA_APRENDIZAJE * delta[j] * act[k][m]
                    red.sesgos[k][j] -= TASA_APRENDIZAJE * delta[j]
                delta = delta_anterior
        red.epocas += 1
        errores.append(error_medio(red, entradas, clases))
    return errores


def guardar(red: Red, ruta: str) -> None:
    c = max(red.u, red.b)
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["tipo", "capa", "pos"] + [f"w{i}" for i in range(c)] + ["sesgo"])
        for letra, valor in zip("uvLbe", [red.u, red.v, red.L, red.b, red.epocas]):
            escritor.writerow([letra, valor] + [""] * (c + 2))
        for k in range(len(red.pesos)):
            tipo = "o" if k == red.L else "h"
            for j in range(len(red.pesos[k])):
                pesos = red.pesos[k][j]
                escritor.writerow([tipo, k + 1, j] + pesos + [""] * (c - len(pesos)) + [red.sesgos[k][j]])


def cargar(ruta: str) -> Red:
    with open(ruta, newline="", encoding="utf-8") as archivo:
        filas = list(csv.reader(archivo))[1:]
    u, v, L, b, epocas = (int(float(fila[1])) for fila in filas[:5])
    c = max(u, b)
    tamanos = [u] + [b] * L + [v]
    pesos = []
    sesgos = []
    for k in range(L + 1):
        capa = sorted((f for f in filas[5:] if int(f[1]) == k + 1), key=lambda f: int(f[2]))
        if len(capa) != tamanos[k + 1]:
            raise ValueError(f"La capa {k + 1} no tiene {tamanos[k + 1]} neuronas")
        pesos.append([[float(x) for x in f[3 : 3 + tamanos[k]]] for f in capa])
        sesgos.append([float(f[3 + c]) if len(f) > 3 + c and f[3 + c] else 0.0 for f in capa])
    return Red(u, v, L, b, pesos, sesgos, epocas)


def imprimir_hiperparametros(red: Red) -> None:
    print(
        f"Hiperparametros: u={red.u} (entradas), v={red.v} (salidas), L={red.L} (capas ocultas), "
        f"b={red.b} (neuronas por capa), e={red.epocas} (epocas)"
    )


def graficar(entradas: list[list[float]], clases: list[int], etiquetas: list[float],
             predicciones: list[int], errores: list[float] | None) -> None:
    en_3d = len(entradas[0]) >= 3
    coordenadas = [[fila[i] if i < len(fila) else 0 for fila in entradas] for i in range(3 if en_3d else 2)]

    colores = []
    aciertos = 0
    for real, predicho in zip(clases, predicciones):
        if real == predicho:
            colores.append("green")
            aciertos += 1
        else:
            colores.append("red")

    x, y = coordenadas[0], coordenadas[1]
    z = {"zs": coordenadas[2]} if en_3d else {}
    plt.figure(figsize=(16, 5))
    proyeccion = "3d" if en_3d else None

    plt.subplot(1, 3, 1, projection=proyeccion)
    plt.scatter(x, y, s=36, c=clases, cmap="viridis", **z)
    plt.title("Datos del archivo (color = clase)")

    plt.subplot(1, 3, 2, projection=proyeccion)
    plt.scatter(x, y, s=36, c=colores, **z)
    plt.title(f"Correctas (verde) / incorrectas (rojo): {aciertos}/{len(clases)}")

    plt.subplot(1, 3, 3)
    if errores is not None:
        plt.plot(range(1, len(errores) + 1), errores)
        plt.title("Error por epoca")
        plt.xlabel("Epoca")
        plt.ylabel("Error cuadratico medio")
    else:
        n = len(etiquetas)
        matriz = [[0] * n for _ in range(n)]
        for real, predicho in zip(clases, predicciones):
            matriz[real][predicho] += 1
        plt.matshow(matriz, fignum=0)
        for i in range(n):
            for j in range(n):
                plt.text(j, i, matriz[i][j], ha="center", va="center", color="white")
        plt.xticks(range(n), [f"{e:g}" for e in etiquetas])
        plt.yticks(range(n), [f"{e:g}" for e in etiquetas])
        plt.xlabel("Clase predicha")
        plt.ylabel("Clase real")
        plt.title("Matriz de confusion")

    plt.tight_layout()
    plt.show()


def pedir_red() -> Red:
    while True:
        print("1) Crear un nuevo perceptron multicapa")
        print("2) Cargarlo de un archivo")
        opcion = input("Elige 1 o 2: ").strip()
        if opcion == "1":
            u = pedir_entero("Neuronas de entrada: ")
            v = pedir_entero("Neuronas de salida: ")
            L = pedir_entero("Capas ocultas: ")
            b = pedir_entero("Neuronas por capa oculta: ")
            return crear_red(u, v, L, b)
        if opcion == "2":
            try:
                return cargar(resolver_ruta(input("Archivo de la red: ")))
            except (OSError, ValueError, IndexError):
                print("No se pudo leer el archivo de la red. Intenta de nuevo.")
        else:
            print("Opcion invalida. Intenta de nuevo.")


def ejecutar(red: Red, entrenando: bool) -> None:
    entradas, clases, etiquetas = pedir_datos(red)
    errores = None
    if entrenando:
        errores = entrenar(red, entradas, clases, pedir_entero("Numero de epocas: "))
    predicciones = predecir(red, entradas)
    aciertos = sum(1 for real, predicho in zip(clases, predicciones) if real == predicho)
    print(f"Aciertos: {aciertos}/{len(clases)}")
    imprimir_hiperparametros(red)
    graficar(entradas, clases, etiquetas, predicciones, errores)


if __name__ == "__main__":
    red = pedir_red()
    while True:
        print("\n1) Entrenar  2) Probar  3) Guardar  4) Salir")
        opcion = input("Elige una opcion: ").strip()
        if opcion == "1":
            ejecutar(red, entrenando=True)
        elif opcion == "2":
            ejecutar(red, entrenando=False)
        elif opcion == "3":
            guardar(red, resolver_ruta(input("Archivo donde guardar: ")))
            imprimir_hiperparametros(red)
        elif opcion == "4":
            break
        else:
            print("Opcion invalida. Intenta de nuevo.")
