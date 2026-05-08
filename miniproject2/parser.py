import json

from models import Stream, Link, Route


# Parse topology.json
def parse_topology(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    topology = data['topology']
    bandwidth = topology['default_bandwidth_mbps']

    links = []

    for l in topology['links']:
        links.append(Link(
            source=l['source'],
            destination=l['destination'],
            delay=l['delay_ms'],
            bandwidth_mbps=l.get('bandwidth_mbps', bandwidth)
        ))
    return links



# Parse streams.json
def parse_streams(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    streams = []
    for s in data['streams']:
        streams.append(Stream(
            id=s['id'],
            name=s['name'],
            pcp=s['pcp'],
            size=s['size_bytes'],
            period=s['period_ms'],
            deadline=s['deadline_ms'],
            source=s['source'],
            destination=s['destination']
        ))
    return streams


# Parse routes.json
def parse_routes(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    routes = []
    for r in data['routes']:
        routes.append(Route(
            flow_id=r['flow_id'],
            path=r['path']
        ))
    return routes