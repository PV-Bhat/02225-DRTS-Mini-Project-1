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
            delay=l['delay'],
            bandwidth_mbps=l.get('bandwidth_mbps', bandwidth)
        ))
    return links, bandwidth



# Parse streams.json
def parse_streams(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    streams = []
    for s in data['streams']:
        streams.append(Stream(
            id=s['id'],
            name=s['name'],
            pcp=s['PCP'],
            size=s['size'],
            period=s['period'],
            deadline=s['destinations'][0]['deadline'],
            source=s['source'],
            destination=s['destinations'][0]['id']
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
            path=[hop['node'] for hop in r['paths'][0]]
        ))
    return routes
