# Nombre del integrante: Alexander Gabriel Ramírez Duarte
# Cédula del integrante: 29611077

"""Tarea 4 - Genetic Algorithm.

Difference-threshold interpretation
-----------------------------------
The assignment does not prescribe an exact use for ``difference_threshold``.
In this implementation it is a convergence criterion: after every generation
we calculate the fitness spread ``best - worst``. The population is considered
stable only when that spread is less than or equal to the threshold for
``CONVERGENCE_PATIENCE`` consecutive generations. Requiring several consecutive
checks avoids stopping because of one unusually uniform generation.

Representation
--------------
Individuals use a real-valued chromosome: one float per problem variable.
Therefore function A uses one gene and function B uses two genes. This keeps the
genetic operators generic while adapting naturally to both population shapes.

Only Python's standard library is used so the program can run on any computer
with Python 3 and the files in this folder.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import statistics
from typing import Any, Callable


DEFAULT_POPULATION_SIZE = 10
DEFAULT_DIFFERENCE_THRESHOLD = 0.05
DEFAULT_MAX_GENERATIONS = 500
DEFAULT_MUTATION_VARIABILITY = 1.0

# Genetic-operator choices. They are intentionally simple and visible because
# the purpose of the assignment is to understand the algorithm, not hide it in
# a specialized optimization library.
TOURNAMENT_SIZE = 3
ELITE_FRACTION = 0.20
MUTATION_PROBABILITY = 0.40
CONVERGENCE_PATIENCE = 10


def clamp(value: float, lower: float, upper: float) -> float:
    """Keep a gene inside the domain accepted by the fitness function."""
    return max(lower, min(upper, value))


def load_function_module(file_path: str):
    """Load a replaceable fitness-function file and validate its contract."""
    file_path = os.path.abspath(file_path)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Function file not found: {file_path}")

    module_name = "ga_user_function_" + str(abs(hash(file_path)))
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None or spec.loader is None:
        raise ValueError("The function file could not be imported.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "BOUNDS"):
        raise ValueError("The function file must define BOUNDS.")
    if not hasattr(module, "VARIABLE_COUNT"):
        raise ValueError("The function file must define VARIABLE_COUNT.")
    if not hasattr(module, "fitness") or not callable(module.fitness):
        raise ValueError("The function file must define a callable fitness(*variables).")

    bounds = module.BOUNDS
    variable_count = module.VARIABLE_COUNT

    if (
        not isinstance(bounds, (tuple, list))
        or len(bounds) != 2
        or not all(isinstance(value, (int, float)) for value in bounds)
        or bounds[0] >= bounds[1]
    ):
        raise ValueError("BOUNDS must be a pair (lower, upper) with lower < upper.")

    if not isinstance(variable_count, int) or variable_count < 1:
        raise ValueError("VARIABLE_COUNT must be a positive integer.")

    return module


def evaluate_genes(genes: list[float], fitness_function: Callable[..., float]) -> float:
    """Evaluate one chromosome using the selected replaceable function."""
    return float(fitness_function(*genes))


def make_member(genes: list[float], fitness_function: Callable[..., float]) -> dict[str, Any]:
    """Build one normalized population member."""
    clean_genes = [float(value) for value in genes]
    return {
        "genes": clean_genes,
        "variables": clean_genes.copy(),
        "fitness": evaluate_genes(clean_genes, fitness_function),
    }


def create_random_member(
    variable_count: int,
    bounds: tuple[float, float],
    fitness_function: Callable[..., float],
    rng: random.Random,
) -> dict[str, Any]:
    """Create one random real-valued chromosome inside the allowed bounds."""
    lower, upper = bounds
    genes = [rng.uniform(lower, upper) for _ in range(variable_count)]
    return make_member(genes, fitness_function)


def sort_population(population: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return members ordered from highest to lowest fitness."""
    return sorted(population, key=lambda member: member["fitness"], reverse=True)


def population_statistics(population: list[dict[str, Any]]) -> tuple[float, float, float]:
    """Return maximum, median and minimum fitness for the current population."""
    fitness_values = [member["fitness"] for member in population]
    return max(fitness_values), statistics.median(fitness_values), min(fitness_values)


def tournament_selection(
    population: list[dict[str, Any]], rng: random.Random
) -> dict[str, Any]:
    """Select a parent by randomly competing a small group of members."""
    group_size = min(TOURNAMENT_SIZE, len(population))
    competitors = rng.sample(population, group_size)
    return max(competitors, key=lambda member: member["fitness"])


def crossover(
    parent_a: dict[str, Any],
    parent_b: dict[str, Any],
    bounds: tuple[float, float],
    rng: random.Random,
) -> list[float]:
    """Create a child using real-valued arithmetic crossover.

    A different mixing factor is used for each gene. This matters for function B:
    a child can inherit a strong x coordinate from one parent and a strong y
    coordinate from another one.
    """
    lower, upper = bounds
    child = []

    for gene_a, gene_b in zip(parent_a["genes"], parent_b["genes"]):
        alpha = rng.random()
        value = alpha * gene_a + (1.0 - alpha) * gene_b
        child.append(clamp(value, lower, upper))

    return child


def mutate(
    genes: list[float],
    mutation_variability: float,
    bounds: tuple[float, float],
    rng: random.Random,
) -> list[float]:
    """Mutate genes with Gaussian noise whose sigma is the requested variability."""
    lower, upper = bounds
    mutated = genes.copy()

    for index in range(len(mutated)):
        if rng.random() < MUTATION_PROBABILITY:
            noise = rng.gauss(0.0, mutation_variability)
            mutated[index] = clamp(mutated[index] + noise, lower, upper)

    return mutated


def create_population(
    function_file: str,
    population_size: int = DEFAULT_POPULATION_SIZE,
    difference_threshold: float = DEFAULT_DIFFERENCE_THRESHOLD,
    max_generations: int = DEFAULT_MAX_GENERATIONS,
    mutation_variability: float = DEFAULT_MUTATION_VARIABILITY,
    seed: int | None = None,
) -> tuple[dict[str, Any], Any]:
    """Create a new population and the metadata required by the JSON contract."""
    module = load_function_module(function_file)
    rng = random.Random(seed)
    bounds = (float(module.BOUNDS[0]), float(module.BOUNDS[1]))

    if population_size < 2:
        raise ValueError("population_size must be at least 2.")
    if difference_threshold < 0:
        raise ValueError("difference_threshold cannot be negative.")
    if max_generations < 1:
        raise ValueError("max_generations must be at least 1.")
    if mutation_variability <= 0:
        raise ValueError("mutation_variability must be greater than 0.")

    population = [
        create_random_member(module.VARIABLE_COUNT, bounds, module.fitness, rng)
        for _ in range(population_size)
    ]

    data = {
        "function": os.path.basename(function_file),
        "parameters": {
            "population_size": int(population_size),
            "difference_threshold": float(difference_threshold),
            "max_generations": int(max_generations),
            "mutation_variability": float(mutation_variability),
            "generations_elapsed": 0,
        },
        "members": sort_population(population),
    }

    return data, module


def next_generation(
    population: list[dict[str, Any]],
    module: Any,
    mutation_variability: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Apply elitism, selection, crossover and mutation to create one generation."""
    population = sort_population(population)
    population_size = len(population)
    bounds = (float(module.BOUNDS[0]), float(module.BOUNDS[1]))

    # Elitism prevents the best solution found so far from being lost by chance.
    elite_count = max(1, int(round(population_size * ELITE_FRACTION)))
    elite_count = min(elite_count, population_size - 1)

    new_population = []
    for member in population[:elite_count]:
        new_population.append(make_member(member["genes"].copy(), module.fitness))

    # The rest of the generation comes from selected parents.
    while len(new_population) < population_size:
        parent_a = tournament_selection(population, rng)
        parent_b = tournament_selection(population, rng)
        child_genes = crossover(parent_a, parent_b, bounds, rng)
        child_genes = mutate(child_genes, mutation_variability, bounds, rng)
        new_population.append(make_member(child_genes, module.fitness))

    return sort_population(new_population)


def train_population(
    data: dict[str, Any],
    module: Any,
    *,
    verbose: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Train until max_generations or stable convergence is reached."""
    rng = random.Random(seed)
    parameters = data["parameters"]
    population = data["members"]

    max_generations = int(parameters["max_generations"])
    threshold = float(parameters["difference_threshold"])
    mutation_variability = float(parameters["mutation_variability"])
    generations_elapsed = int(parameters["generations_elapsed"])

    stable_generations = 0

    while generations_elapsed < max_generations:
        population = next_generation(population, module, mutation_variability, rng)
        generations_elapsed += 1

        best, median, worst = population_statistics(population)
        spread = best - worst

        if verbose:
            print(
                f"Generation {generations_elapsed:4d} | "
                f"max={best:.10f} | median={median:.10f} | min={worst:.10f}"
            )

        if spread <= threshold:
            stable_generations += 1
        else:
            stable_generations = 0

        if stable_generations >= CONVERGENCE_PATIENCE:
            if verbose:
                print(
                    "Convergence criterion reached: fitness spread stayed "
                    f"<= {threshold} for {CONVERGENCE_PATIENCE} generations."
                )
            break

    parameters["generations_elapsed"] = generations_elapsed
    data["members"] = sort_population(population)
    return data


def validate_population_data(data: dict[str, Any]) -> None:
    """Validate the repository JSON population contract before using a file."""
    if not isinstance(data, dict):
        raise ValueError("Population file must contain a JSON object.")

    for key in ("function", "parameters", "members"):
        if key not in data:
            raise ValueError(f"Population file is missing key: {key}")

    parameters = data["parameters"]
    required_parameters = (
        "population_size",
        "difference_threshold",
        "max_generations",
        "mutation_variability",
        "generations_elapsed",
    )

    for key in required_parameters:
        if key not in parameters:
            raise ValueError(f"Population parameters are missing: {key}")

    members = data["members"]
    if len(members) != int(parameters["population_size"]):
        raise ValueError("members length does not match population_size.")

    for member in members:
        if "variables" not in member or "fitness" not in member:
            raise ValueError("Each member must contain variables and fitness.")


def resolve_function_path(population_file: str, function_name: str) -> str:
    """Find the function file next to the JSON first, then next to this script."""
    population_directory = os.path.dirname(os.path.abspath(population_file))
    script_directory = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(population_directory, function_name),
        os.path.join(script_directory, function_name),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find {function_name} next to the population or program file."
    )


def normalize_loaded_members(data: dict[str, Any], module: Any) -> None:
    """Restore genes and recompute fitness so a loaded population can continue training."""
    variable_count = int(module.VARIABLE_COUNT)
    normalized = []

    for raw_member in data["members"]:
        genes = raw_member.get("genes", raw_member.get("variables"))

        if not isinstance(genes, list) or len(genes) != variable_count:
            raise ValueError(
                "A population member has an invalid number of genes for this function."
            )

        normalized.append(make_member([float(value) for value in genes], module.fitness))

    data["members"] = sort_population(normalized)


def load_population(file_path: str) -> tuple[dict[str, Any], Any]:
    """Load a JSON population and its matching replaceable fitness function."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    validate_population_data(data)
    function_path = resolve_function_path(file_path, data["function"])
    module = load_function_module(function_path)
    normalize_loaded_members(data, module)
    return data, module


def save_population(data: dict[str, Any], file_path: str) -> None:
    """Save the exact population contract used by the repository grader."""
    data["members"] = sort_population(data["members"])

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def print_population(data: dict[str, Any]) -> None:
    """Print all parameters and members from best to worst."""
    parameters = data["parameters"]
    members = sort_population(data["members"])

    print("\n" + "=" * 72)
    print("PARÁMETROS DE LA POBLACIÓN")
    print("=" * 72)
    print(f"Función: {data['function']}")
    print(f"Tamaño de población: {parameters['population_size']}")
    print(f"Umbral de diferencia: {parameters['difference_threshold']}")
    print(f"Máximo de generaciones: {parameters['max_generations']}")
    print(f"Generaciones transcurridas: {parameters['generations_elapsed']}")
    print(f"Variabilidad de mutación: {parameters['mutation_variability']}")

    print("\nMIEMBROS ORDENADOS DE MEJOR A PEOR")
    print("-" * 72)

    for position, member in enumerate(members, start=1):
        variables = ", ".join(f"{value:.10f}" for value in member["variables"])
        print(
            f"{position:>3}. variables=[{variables}]  "
            f"fitness={member['fitness']:.12f}"
        )

    best, median, worst = population_statistics(members)
    print("-" * 72)
    print(f"Máximo:  {best:.12f}")
    print(f"Mediana: {median:.12f}")
    print(f"Mínimo:  {worst:.12f}")
    print("=" * 72)


def read_int(prompt: str, default: int, minimum: int = 1) -> int:
    """Read an integer from the console while supporting a default value."""
    while True:
        text = input(f"{prompt} [{default}]: ").strip()
        if text == "":
            return default
        try:
            value = int(text)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"Ingrese un entero mayor o igual que {minimum}.")


def read_float(prompt: str, default: float, minimum_exclusive: float | None = None) -> float:
    """Read a float from the console while supporting a default value."""
    while True:
        text = input(f"{prompt} [{default}]: ").strip()
        if text == "":
            return default
        try:
            value = float(text)
            if minimum_exclusive is None or value > minimum_exclusive:
                return value
        except ValueError:
            pass
        print("Ingrese un número válido.")


def create_from_console() -> tuple[dict[str, Any], Any]:
    """Ask the user for all parameters required to create a population."""
    print("\n--- Crear nueva población ---")
    function_file = input("Archivo Python de la función: ").strip()

    population_size = read_int(
        "Tamaño de la población", DEFAULT_POPULATION_SIZE, minimum=2
    )
    difference_threshold = read_float(
        "Umbral de diferencia", DEFAULT_DIFFERENCE_THRESHOLD
    )
    max_generations = read_int(
        "Número máximo de generaciones", DEFAULT_MAX_GENERATIONS, minimum=1
    )
    mutation_variability = read_float(
        "Variabilidad de la mutación",
        DEFAULT_MUTATION_VARIABILITY,
        minimum_exclusive=0.0,
    )

    return create_population(
        function_file=function_file,
        population_size=population_size,
        difference_threshold=difference_threshold,
        max_generations=max_generations,
        mutation_variability=mutation_variability,
    )


def population_menu(data: dict[str, Any], module: Any) -> None:
    """Allow repeated training, saving and inspection of the active population."""
    while True:
        print("\n--- Menú de población ---")
        print("1. Entrenar / continuar entrenamiento")
        print("2. Guardar población")
        print("3. Mostrar población")
        print("4. Cambiar máximo de generaciones")
        print("5. Salir")

        option = input("Seleccione una opción: ").strip()

        if option == "1":
            elapsed = int(data["parameters"]["generations_elapsed"])
            maximum = int(data["parameters"]["max_generations"])

            if elapsed >= maximum:
                print(
                    "La población ya alcanzó su máximo configurado. "
                    "Use la opción 4 para aumentarlo antes de continuar."
                )
                continue

            train_population(data, module, verbose=True)
            print_population(data)

        elif option == "2":
            path = input("Nombre/ruta del archivo JSON: ").strip()
            if not path.lower().endswith(".json"):
                path += ".json"
            save_population(data, path)
            print(f"Población guardada en: {path}")

        elif option == "3":
            print_population(data)

        elif option == "4":
            current = int(data["parameters"]["max_generations"])
            elapsed = int(data["parameters"]["generations_elapsed"])
            new_max = read_int("Nuevo máximo de generaciones", current, minimum=elapsed + 1)
            data["parameters"]["max_generations"] = new_max
            print(f"Nuevo máximo configurado: {new_max}")

        elif option == "5":
            return

        else:
            print("Opción inválida.")


def main() -> None:
    """Run the interactive console application required by the assignment."""
    print("=" * 72)
    print("TAREA 4 - ALGORITMO GENÉTICO")
    print("=" * 72)
    print("1. Crear una nueva población")
    print("2. Cargar una población desde un archivo")

    while True:
        option = input("Seleccione una opción: ").strip()

        try:
            if option == "1":
                data, module = create_from_console()
                print("\nPoblación creada. Se iniciará el entrenamiento.")
                train_population(data, module, verbose=True)
                print_population(data)
                population_menu(data, module)
                return

            if option == "2":
                path = input("Archivo JSON de población: ").strip()
                data, module = load_population(path)
                print_population(data)
                population_menu(data, module)
                return

            print("Opción inválida. Escriba 1 o 2.")

        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
