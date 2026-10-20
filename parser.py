import re

NODE_PATTERN = re.compile(r"(\d+)\s*:\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
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

            if line.lower() in SECTION_HEADERS:
                current_section = SECTION_HEADERS[line.lower()]
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
                    raise ValueError("Data found outside a section")
            except ValueError as error:
                raise ValueError(f"{filename}, line {line_num}: {error}") from error

    validate_graph(filename, nodes, edges, origin, destinations)
    return nodes, edges, origin, destinations


def add_node(nodes, edges, line):
    """Parse a node line. An exact duplicate is ignored; different coordinates conflict."""
    match = NODE_PATTERN.fullmatch(line)
    if not match:
        raise ValueError("Invalid node format")

    node_id, x, y = map(int, match.groups())
    if node_id in nodes and nodes[node_id] != (x, y):
        raise ValueError(f"Node {node_id} already defined as {nodes[node_id]}")

    nodes[node_id] = (x, y)
    edges.setdefault(node_id, {})


def add_edge(edges, line):
    """Parse an edge line. An exact duplicate is ignored; a different cost conflicts."""
    match = EDGE_PATTERN.fullmatch(line)
    if not match:
        raise ValueError("Invalid edge format")

    source, target, cost = int(match[1]), int(match[2]), float(match[3])
    if cost < 0:
        raise ValueError("Edge costs cannot be negative")

    old_cost = edges.get(source, {}).get(target)
    if old_cost is not None and old_cost != cost:
        raise ValueError(f"Edge ({source}, {target}) already has cost {old_cost}")

    edges.setdefault(source, {})[target] = cost
    edges.setdefault(target, {})


def read_origin(origin, line):
    """Parse an origin line and return it. The same value again is ignored; a different one conflicts."""
    if not re.fullmatch(r"\d+", line):
        raise ValueError("Invalid origin format")

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
            raise ValueError("Invalid destination format")

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
        raise ValueError(f"{filename}: Origin node {origin} does not exist")
    for d in destinations:
        if d not in nodes:
            raise ValueError(f"{filename}: Destination node {d} does not exist")
    for source, adjacent in edges.items():
        if source not in nodes:
            raise ValueError(f"{filename}: Source node {source} does not exist")
        for target in adjacent:
            if target not in nodes:
                raise ValueError(f"{filename}: Target node {target} does not exist")
