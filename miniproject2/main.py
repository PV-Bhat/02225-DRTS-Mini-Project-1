import sys
from parser import parse_topology, parse_streams, parse_routes
from simulator import simulate
from analysis import calculate_stream_wcd, calculate_sp_delay


# Load files
TEST_CASE = sys.argv[1]

links, bandwidth = parse_topology(f'{TEST_CASE}/topology.json')
streams = parse_streams(f'{TEST_CASE}/streams.json')
routes = parse_routes(f'{TEST_CASE}/routes.json')

print(f"Topology bandwidth: {bandwidth} Mbps")
print(f"Number of streams: {len(streams)}")
print(f"Number of links: {len(links)}")
print()

# Analytical CBS WCDs
print("CBS Analytical WCDs:")
for stream in streams:
    route = next((r for r in routes if r.flow_id == stream.id), None)
    if route is None:
        print(f"Stream {stream.name} (ID {stream.id}): no route found, skipping")
        continue
    cbs_wcd = calculate_stream_wcd(stream, route, links, streams, routes, bandwidth)
    print(f"Stream {stream.name} (ID {stream.id}): {cbs_wcd:.2f} us")

# SP Analytical WCDs
print("\nSP Analytical WCDs:")
for stream in streams:
    sp_wcd = calculate_sp_delay(stream, streams, bandwidth)
    print(f"Stream {stream.name} (ID {stream.id}): {sp_wcd:.2f} us")

# Simulation (only periodic streams)
periodic = []
for s in streams:
    if s.period is not None and s.period > 0:
        periodic.append(s)

print(f"\nSimulated Results ({len(periodic)} periodic streams):")

results = simulate(periodic, bandwidth)

for stream_id, delays in sorted(results.items()):
    max_delay = max(delays)
    avg_delay = sum(delays) / len(delays)
    print(f"Stream ID {stream_id}: Max Delay = {max_delay:.2f} us, Avg Delay = {avg_delay:.2f} us")
