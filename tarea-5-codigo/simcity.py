"""Text-mode cellular automaton inspired by the original SimCity.

Team: LUMON-2627-1
Members (full names and student ID numbers): Alexander Ramirez 29611077, Mario Suso 30751403, Diego Mendez 30830746
- TO COMPLETE BEFORE SUBMISSION (do not invent team members or ID numbers).

Implementation decisions for underspecified assignment requirements:
1. Distance <= 10 between zones is Manhattan distance between their nearest
   occupied cells (minimum distance between 3x3 rectangular footprints).
2. On a tick, each eligible condition provides one positive vote or two
   negative votes PER matching road cell / neighboring zone. Grow one level
   when positive > negative; shrink one level when negative > positive; on a
   tie do not change. Growth is capped to [0, 27]. All votes are evaluated
   from the previous tick state (synchronous updates).
3. Road traffic at a cell is the sum of sqrt(level) for distinct orthogonally
   adjacent zones; traffic from adjacent road cells propagates across the
   network by taking 80% of the maximum previous-tick neighbor traffic.
   We take the MAX between local traffic and propagated traffic, rather than
   repeatedly summing neighboring road flows, to avoid artificial runaway.
   This is an explicit choice because propagation is not mathematically
   specified by the assignment. Traffic remains a nonnegative float.
4. Traffic is formatted in 3 characters: vertical '|6|', horizontal '666',
   crossing ':6:'. A two- or three-digit number occupies the whole cell.
5. On tick zero, no zone or road was previously shrinking.
6. A road may cross a perpendicular road, but may not overlap a parallel
   road. All placements are checked before mutating the state.
7. 29 columns x 3 characters plus a 2-character row label occupy 89 printed
   characters. That is the smallest layout that displays all 29 complete
   cells and labels on the required 16-line grid; the PDF also mentions 88.

Only Python's standard library is required. The player-facing interface is
in Spanish; code, identifiers, and comments are in English.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

GRID_WIDTH = 29
GRID_HEIGHT = 15
ZONE_SIZE = 3
MAX_ZONE_LEVEL = 27
MAX_ZONE_DISTANCE = 10
HORIZONTAL = 1
VERTICAL = 2
TRAFFIC_PROPAGATION = 0.80
BASE_DIR = Path(__file__).resolve().parent
COORDINATE_PATTERN = re.compile(r"^([A-Za-z]{1,2})(\d{1,2})$")
COLUMN_PATTERN = re.compile(r"^[A-Za-z]{1,2}$")
ROW_PATTERN = re.compile(r"^\d{1,2}$")


class GameError(ValueError):
    """A rejected user operation. The state is left unchanged."""


def column_name(index: int) -> str:
    """Return a spreadsheet column name for a zero-based index."""
    value = index + 1
    name = ""
    while value:
        value, digit = divmod(value - 1, 26)
        name = chr(ord("A") + digit) + name
    return name


def column_index(label: str) -> int:
    if not COLUMN_PATTERN.fullmatch(label):
        raise GameError("columna inválida; utiliza A hasta AC")
    number = 0
    for letter in label.upper():
        number = number * 26 + ord(letter) - ord("A") + 1
    index = number - 1
    if not 0 <= index < GRID_WIDTH:
        raise GameError("columna fuera de la cuadrícula (A-AC)")
    return index


def parse_cell(parts: List[str]) -> Tuple[int, int]:
    """Read either ['C4'] or ['C', '4']; return zero-based (row,col)."""
    if len(parts) == 1:
        match = COORDINATE_PATTERN.fullmatch(parts[0])
        if not match:
            raise GameError("celda inválida; utiliza C4 o C 4")
        col_text, row_text = match.groups()
    elif len(parts) == 2 and COLUMN_PATTERN.fullmatch(parts[0]) and ROW_PATTERN.fullmatch(parts[1]):
        col_text, row_text = parts
    else:
        raise GameError("celda inválida; utiliza C4 o C 4")
    row = int(row_text) - 1
    if not 0 <= row < GRID_HEIGHT:
        raise GameError("fila fuera de la cuadrícula (1-15)")
    return row, column_index(col_text)


@dataclass
class Zone:
    kind: str
    row: int
    col: int
    level: int = 0
    last_change: int = 0  # -1 if it shrank in the last tick, 1 if it grew

    def contains(self, row: int, col: int) -> bool:
        return self.row <= row < self.row + ZONE_SIZE and self.col <= col < self.col + ZONE_SIZE

    def cells(self) -> Iterable[Tuple[int, int]]:
        for row in range(self.row, self.row + ZONE_SIZE):
            for col in range(self.col, self.col + ZONE_SIZE):
                yield row, col


@dataclass
class Road:
    orientation: int  # HORIZONTAL | VERTICAL (3 means intersection)
    traffic: float = 0.0
    decreased: bool = False  # traffic decreased in the last tick


class SimCity:
    """Owns grid state, voting logic, roads, file persistence, and rendering."""

    def __init__(self) -> None:
        self.zones: List[Zone] = []
        self.roads: Dict[Tuple[int, int], Road] = {}
        self.tick_count = 0

    def occupied_zone_cell(self, row: int, col: int) -> Optional[Zone]:
        return next((zone for zone in self.zones if zone.contains(row, col)), None)

    def add_zone(self, kind: str, row: int, col: int) -> None:
        if kind not in ("r", "c", "i"):
            raise GameError("tipo de zona inválido")
        if row < 0 or col < 0 or row + ZONE_SIZE > GRID_HEIGHT or col + ZONE_SIZE > GRID_WIDTH:
            raise GameError("la zona 3x3 queda fuera de la cuadrícula")
        candidate = Zone(kind, row, col)
        for r, c in candidate.cells():
            if (r, c) in self.roads or self.occupied_zone_cell(r, c) is not None:
                raise GameError("la zona se solapa con una celda ocupada")
        self.zones.append(candidate)

    def add_road(self, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        row1, col1 = start
        row2, col2 = end
        if start == end:
            raise GameError("la arteria necesita un inicio y fin distintos")
        if row1 == row2:
            orientation = HORIZONTAL
            cells = [(row1, col) for col in range(min(col1, col2), max(col1, col2) + 1)]
        elif col1 == col2:
            orientation = VERTICAL
            cells = [(row, col1) for row in range(min(row1, row2), max(row1, row2) + 1)]
        else:
            raise GameError("la arteria debe ser horizontal o vertical")
        # Validate the ENTIRE placement first: failed commands cannot mutate the city.
        for row, col in cells:
            if self.occupied_zone_cell(row, col) is not None:
                raise GameError("la arteria se solapa con una zona")
            prior = self.roads.get((row, col))
            if prior is not None and prior.orientation & orientation:
                raise GameError("la arteria se solapa con otra arteria paralela")
        for row, col in cells:
            road = self.roads.get((row, col))
            if road is None:
                self.roads[(row, col)] = Road(orientation)
            else:
                road.orientation |= orientation

    @staticmethod
    def neighbors(row: int, col: int) -> Iterable[Tuple[int, int]]:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                yield nr, nc

    @staticmethod
    def zone_distance(a: Zone, b: Zone) -> int:
        """Manhattan gap between the nearest cells of two 3x3 zones."""
        row_gap = max(a.row - b.row - (ZONE_SIZE - 1), b.row - a.row - (ZONE_SIZE - 1), 0)
        col_gap = max(a.col - b.col - (ZONE_SIZE - 1), b.col - a.col - (ZONE_SIZE - 1), 0)
        return row_gap + col_gap

    def adjacent_road_cells(self, zone: Zone) -> List[Tuple[int, int]]:
        cells = set()
        for row, col in zone.cells():
            for neighboring_cell in self.neighbors(row, col):
                if neighboring_cell in self.roads:
                    cells.add(neighboring_cell)
        return sorted(cells)

    def nearby_zones(self, zone: Zone, kind: str) -> List[Zone]:
        return [other for other in self.zones
                if other is not zone and other.kind == kind
                and self.zone_distance(zone, other) <= MAX_ZONE_DISTANCE]

    def zone_votes(self, zone: Zone) -> Tuple[int, int]:
        """Return (growth_votes, shrink_votes), already weighting shrink by 2."""
        grow, shrink = 0, 0
        road_cells = self.adjacent_road_cells(zone)
        if zone.kind in ("r", "i"):
            grow += len(road_cells)
        elif zone.kind == "c":
            grow += sum(self.roads[cell].traffic > 5 for cell in road_cells)

        if zone.kind == "r":
            shrink += 2 * sum(self.roads[cell].traffic > 5 for cell in road_cells)
            industrial = self.nearby_zones(zone, "i")
            grow += sum(other.level < 18 for other in industrial)
            shrink += 2 * max(0, len(industrial) - 10)
            for other in self.nearby_zones(zone, "c"):
                grow += other.level > math.sqrt(zone.level)
                shrink += 2 * (other.last_change < 0)
        elif zone.kind == "c":
            shrink += 2 * sum(self.roads[cell].decreased for cell in road_cells)
            for other in self.nearby_zones(zone, "r"):
                grow += 1
                shrink += 2 * (other.last_change < 0)
            for other in self.nearby_zones(zone, "i"):
                grow += other.level > math.sqrt(zone.level)
                shrink += 2 * (other.last_change < 0)
        else:  # Industrial
            for other in self.nearby_zones(zone, "r"):
                grow += 1
                shrink += 2 * (other.last_change < 0)
            for other in self.nearby_zones(zone, "c"):
                grow += other.level > math.sqrt(zone.level)
                shrink += 2 * (other.last_change < 0)
            for other in self.nearby_zones(zone, "i"):
                shrink += 2 * (other.last_change < 0)
        return grow, shrink

    def local_road_traffic(self, row: int, col: int) -> float:
        adjacent_zones = set()
        for nr, nc in self.neighbors(row, col):
            for index, zone in enumerate(self.zones):
                if zone.contains(nr, nc):
                    adjacent_zones.add(index)
        return sum(math.sqrt(self.zones[index].level) for index in adjacent_zones)

    def tick(self) -> None:
        # Evaluate each zone using ONLY the previous state's zone and road data.
        moves = []
        for zone in self.zones:
            growth, shrink = self.zone_votes(zone)
            direction = (growth > shrink) - (growth < shrink)
            updated = max(0, min(MAX_ZONE_LEVEL, zone.level + direction))
            moves.append((zone, updated, updated - zone.level))
        for zone, level, change in moves:
            zone.level, zone.last_change = level, change

        # Use a snapshot of the previous road traffic to avoid update-order bias.
        previous_traffic = {cell: road.traffic for cell, road in self.roads.items()}
        updated_traffic = {}
        for (row, col), road in self.roads.items():
            local = self.local_road_traffic(row, col)
            neighbors = [previous_traffic[p] for p in self.neighbors(row, col)
                         if p in previous_traffic]
            propagated = TRAFFIC_PROPAGATION * max(neighbors, default=0.0)
            updated_traffic[(row, col)] = max(local, propagated)
        for cell, traffic in updated_traffic.items():
            road = self.roads[cell]
            road.decreased = traffic < previous_traffic[cell] - 1e-9
            road.traffic = traffic
        self.tick_count += 1

    @staticmethod
    def render_zone_cell(zone: Zone, row: int, col: int) -> str:
        start = ((row - zone.row) * ZONE_SIZE + (col - zone.col)) * 3
        uppercase = max(0, min(3, zone.level - start))
        return zone.kind.upper() * uppercase + zone.kind.lower() * (3 - uppercase)

    @staticmethod
    def render_road_cell(road: Road) -> str:
        value = max(2, min(999, int(round(road.traffic)))) if road.traffic > 1 else 0
        if road.traffic <= 1.0:
            return ":+:" if road.orientation == 3 else "===" if road.orientation == HORIZONTAL else "| |"
        if value >= 10:
            return f"{value:>3}"  # Retain exact cell width even at >=10.
        if road.orientation == 3:
            return f":{value}:"
        if road.orientation == VERTICAL:
            return f"|{value}|"
        return str(value) * 3

    def render(self) -> str:
        # One header and 15 data rows; every cell is exactly 3 characters wide.
        header = "  " + "".join(f"{column_name(col):>3}" for col in range(GRID_WIDTH))
        lines = [header]
        for row in range(GRID_HEIGHT):
            cells = []
            for col in range(GRID_WIDTH):
                zone = self.occupied_zone_cell(row, col)
                if zone is not None:
                    cells.append(self.render_zone_cell(zone, row, col))
                elif (row, col) in self.roads:
                    cells.append(self.render_road_cell(self.roads[(row, col)]))
                else:
                    cells.append("   ")
            lines.append(f"{row + 1:>2}" + "".join(cells))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "format": "simcity-lumon-1", "width": GRID_WIDTH, "height": GRID_HEIGHT,
            "ticks": self.tick_count,
            "zones": [dict(kind=z.kind, row=z.row, col=z.col,
                           level=z.level, last_change=z.last_change)
                      for z in self.zones],
            "roads": [dict(row=r, col=c, orientation=road.orientation,
                           traffic=round(road.traffic, 9), decreased=road.decreased)
                      for (r, c), road in sorted(self.roads.items())],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimCity":
        if not isinstance(data, dict) or data.get("format") != "simcity-lumon-1":
            raise GameError("archivo con formato desconocido")
        if data.get("width") != GRID_WIDTH or data.get("height") != GRID_HEIGHT:
            raise GameError("dimensiones incompatibles de la ciudad guardada")
        ticks = data.get("ticks")
        if type(ticks) is not int or ticks < 0:
            raise GameError("contador de ticks inválido")
        if not isinstance(data.get("zones"), list) or not isinstance(data.get("roads"), list):
            raise GameError("lista de zonas o arterias inválida")
        city = cls()
        for raw in data["zones"]:
            if not isinstance(raw, dict):
                raise GameError("registro de zona inválido")
            kind, row, col = raw.get("kind"), raw.get("row"), raw.get("col")
            if type(row) is not int or type(col) is not int or kind not in ("r", "c", "i"):
                raise GameError("coordenadas o tipo de zona inválidos")
            city.add_zone(kind, row, col)
            zone = city.zones[-1]
            level, last_change = raw.get("level"), raw.get("last_change")
            if type(level) is not int or not 0 <= level <= MAX_ZONE_LEVEL:
                raise GameError("nivel de zona inválido")
            if type(last_change) is not int or last_change not in (-1, 0, 1):
                raise GameError("historial de zona inválido")
            zone.level, zone.last_change = level, last_change
        for raw in data["roads"]:
            if not isinstance(raw, dict):
                raise GameError("registro de arteria inválido")
            row, col, orientation = raw.get("row"), raw.get("col"), raw.get("orientation")
            if (type(row) is not int or type(col) is not int or
                    not (0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH) or
                    type(orientation) is not int or orientation not in (1, 2, 3)):
                raise GameError("coordenadas u orientación de arteria inválidas")
            if (row, col) in city.roads or city.occupied_zone_cell(row, col):
                raise GameError("la ciudad guardada tiene celdas superpuestas")
            traffic, decreased = raw.get("traffic"), raw.get("decreased")
            if (type(traffic) not in (int, float) or not math.isfinite(traffic) or
                    traffic < 0 or type(decreased) is not bool):
                raise GameError("historial de tráfico inválido")
            city.roads[(row, col)] = Road(orientation, float(traffic), decreased)
        city.tick_count = ticks
        return city

    @staticmethod
    def file_path(name: str) -> Path:
        # Game files must remain inside the assignment folder.
        if not name:
            raise GameError("indica la ruta del archivo")
        target = Path(name)
        if target.is_absolute():
            raise GameError("utiliza una ruta relativa a tarea-5-codigo")
        full_path = (BASE_DIR / target).resolve()
        if full_path != BASE_DIR and BASE_DIR not in full_path.parents:
            raise GameError("la ruta debe permanecer en tarea-5-codigo")
        return full_path

    def save(self, name: str) -> None:
        path = self.file_path(name)
        try:
            # Avoid partially replacing a working save if serialization fails.
            path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise GameError(f"no se pudo guardar el archivo: {exc}") from exc

    def load(self, name: str) -> None:
        path = self.file_path(name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            next_city = self.from_dict(data)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise GameError(f"no se pudo cargar la ciudad: {exc}") from exc
        # Atomic replacement: preserve existing state on ANY load failure.
        self.zones, self.roads, self.tick_count = (
            next_city.zones, next_city.roads, next_city.tick_count
        )


def parse_road_args(parts: List[str]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Accept 'a C7 P', 'a C 7 P', 'a F4 9' or complete end cells."""
    if len(parts) in (2, 3) and COORDINATE_PATTERN.fullmatch(parts[0]):
        start = parse_cell([parts[0]])
        remaining = parts[1:]
    elif len(parts) in (3, 4) and COLUMN_PATTERN.fullmatch(parts[0]) and ROW_PATTERN.fullmatch(parts[1]):
        start = parse_cell(parts[:2])
        remaining = parts[2:]
    else:
        raise GameError("usa a C7 P, a F4 9, a C 7 P o a F 4 9")
    if len(remaining) == 1:
        finish = remaining[0]
        if COLUMN_PATTERN.fullmatch(finish):
            end = (start[0], column_index(finish))
        elif ROW_PATTERN.fullmatch(finish):
            row = int(finish) - 1
            if not 0 <= row < GRID_HEIGHT:
                raise GameError("fila final fuera de la cuadrícula")
            end = (row, start[1])
        else:
            end = parse_cell([finish])
    elif len(remaining) == 2:
        end = parse_cell(remaining)
    else:
        raise GameError("extremo final de arteria inválido")
    if start[0] != end[0] and start[1] != end[1]:
        raise GameError("la arteria debe ser horizontal o vertical")
    return start, end


def handle_command(city: SimCity, raw: str) -> bool:
    """Process one command; return False only when user wants to exit."""
    if not raw.strip():
        city.tick()
        print(city.render())
        return True
    try:
        args = shlex.split(raw)
        if not args:
            city.tick()
            print(city.render())
            return True
        command, params = args[0].lower(), args[1:]
        if command == "salir":
            if params:
                raise GameError("el comando salir no recibe argumentos")
            print("Juego finalizado.")
            return False
        if command in ("r", "c", "i"):
            row, col = parse_cell(params)
            city.add_zone(command, row, col)
            print(city.render())
        elif command == "a":
            city.add_road(*parse_road_args(params))
            print(city.render())
        elif command in ("guardar", "cargar"):
            if len(params) != 1:
                raise GameError(f"uso: {command} <ruta_archivo>")
            if command == "guardar":
                city.save(params[0])
                print(f"Ciudad guardada en {params[0]}.")
            else:
                city.load(params[0])
                print(city.render())
        else:
            raise GameError("comando desconocido (r, c, i, a, guardar, cargar, salir)")
    except (GameError, ValueError) as exc:
        print(f"Error: {exc}")
    return True


def main() -> int:
    city = SimCity()
    print("SimCity - Computación Emergente | Enter avanza un tick; 'salir' termina.")
    print(city.render())
    while True:
        try:
            entered = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nJuego finalizado.")
            return 0
        if not handle_command(city, entered):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
