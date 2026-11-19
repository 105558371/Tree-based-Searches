import re

NODE_PATTERN = re.compile(r"(\d+)\s*:\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
EDGE_PATTERN = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*(-?\d+(?:\.\d+)?)")
SECTION_HEADERS = {
    "nodes:": "nodes",
    "edges:": "edges",
    "origin:": "origin",
    "destinations:": "destinations",
}


def parse_graph(filename):
    """
    Reads a problem file. Sections may appear in any order and may repeat.
    Returns:
      nodes:        dict[int, tuple[int, int]]   node id -> (x, y)
      edges:        dict[int, dict[int, float]]  source -> {target: cost}
      origin:       int
      destinations: set[int]
    """
    nodes = {}
    edges = {}
    origin = None
    destinations = set()    # unique
    current_section = None

    # utf-8-sig removes the BOM that Windows Notepad can add
    with open(filename, "r", encoding="utf-8-sig") as file:
        for line_num, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:    # Skip empty lines
                continue

            # Value may follow the header, e.g. "Origin: 2"
            header = next((h for h in SECTION_HEADERS if line.lower().startswith(h)), None)
            if header:
                current_section = SECTION_HEADERS[header]
                line = line[len(header):].strip()
                if not line:
                    continue

            try:
                if current_section == "nodes":
                    add_node(nodes, edges, line)
                elif current_section == "edges":
                    add_edge(edges, line)
                elif current_section == "origin":
                    origin = read_origin(origin, line)
                elif current_section == "destinations":
                    add_destinations(destinations, line)
                else:
                    raise ValueError(f"'{line}' is not inside a section. The file should start with 'Nodes:'")
            except ValueError as error:
                raise ValueError(f"{filename}, line {line_num}: {error}") from error

    validate_graph(filename, nodes, edges, origin, destinations)
    return nodes, edges, origin, destinations


def to_number(text):
    value = float(text)
    return int(value) if value.is_integer() else value


def add_node(nodes, edges, line):
    """Parse a node line. An exact duplicate is ignored; different coordinates conflict."""
    match = NODE_PATTERN.fullmatch(line)
    if not match:
        raise ValueError(f"Invalid node '{line}'. Expected a node like: 1: (4,1)")

    node_id, x, y = int(match[1]), to_number(match[2]), to_number(match[3])
    if node_id in nodes and nodes[node_id] != (x, y):
        raise ValueError(f"Node {node_id} already defined as {nodes[node_id]}")

    nodes[node_id] = (x, y)
    edges.setdefault(node_id, {})


def add_edge(edges, line):
    """Parse an edge line. An exact duplicate is ignored; a different cost conflicts."""
    match = EDGE_PATTERN.fullmatch(line)
    if not match:
        raise ValueError(f"Invalid edge '{line}'. Expected an edge like: (2,1): 4")

    source, target, cost = int(match[1]), int(match[2]), float(match[3])
    if cost < 0:
        raise ValueError(f"Edge ({source},{target}) has a negative cost; costs must be 0 or more")

    old_cost = edges.get(source, {}).get(target)
    if old_cost is not None and old_cost != cost:
        raise ValueError(f"Edge ({source}, {target}) already has cost {old_cost}")

    edges.setdefault(source, {})[target] = cost
    edges.setdefault(target, {})


def read_origin(origin, line):
    """Parse an origin line and return it. The same value again is ignored; a different one conflicts."""
    if not re.fullmatch(r"\d+", line):
        raise ValueError(f"Invalid origin '{line}'. Expected one node number, e.g. 2")

    new_origin = int(line)
    if origin is not None and origin != new_origin:
        raise ValueError(f"Origin already defined as {origin}")

    return new_origin


def add_destinations(destinations, line):
    """Parse a destinations line like '5; 4' and merge it into the set."""
    for part in line.split(";"):
        part = part.strip()
        if not part:    # Allow a trailing ";"
            continue
        if not re.fullmatch(r"\d+", part):
            raise ValueError(f"Invalid destination '{part}'. Expected node numbers separated by ';', e.g. 5; 4")

        destinations.add(int(part))


def validate_graph(filename, nodes, edges, origin, destinations):
    """Check the completed graph. Sections can be in any order, so this runs last."""
    if not nodes:
        raise ValueError(f"{filename}: No nodes found")
    if origin is None:
        raise ValueError(f"{filename}: No origin found")
    if not destinations:
        raise ValueError(f"{filename}: No destinations found")

    if origin not in nodes:
        raise ValueError(f"{filename}: The origin {origin} is not listed under 'Nodes:'")
    for d in destinations:
        if d not in nodes:
            raise ValueError(f"{filename}: Destination {d} is not listed under 'Nodes:'")
    for source, adjacent in edges.items():
        for target in adjacent:
            for end in (source, target):
                if end not in nodes:
                    raise ValueError(f"{filename}: Edge ({source},{target}) uses node {end}, "
                                     f"which is not listed under 'Nodes:'")
