from dataclasses import dataclass

@dataclass
class Stream:
    id: int
    name: str
    pcp: int
    size: int
    period: object  # int or None for best-effort
    deadline: object  # int or None for best-effort
    source: str
    destination: str


@dataclass
class Link:
    source: str
    destination: str
    delay: float
    bandwidth_mbps: float

@dataclass
class Route:
    flow_id: int
    path: list


@dataclass
class QueueState:
    credit: float = 0.0
