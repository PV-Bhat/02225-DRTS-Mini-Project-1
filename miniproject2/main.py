
from parser import parse_topology, parse_streams, parse_routes
from simulator import simulate
from analysis import calculate_stream_wcd, calculate_sp_delay





#Load files

TEST_CASE = 'test_cases/examples/test_case_1'

links = parse_topology(f'{TEST_CASE}/topology.json')
streams = parse_streams(f'{TEST_CASE}/streams.json')
routes = parse_routes(f'{TEST_CASE}/routes.json')


#Analyical WCDs

print("CBS Analytical WCDs:")

for stream in streams:
    route = next(r for r in routes if r.flow_id == stream.id)

    cbs_wcd = calculate_stream_wcd(stream, route, links, streams)
    print(f"Stream {stream.name} (ID {stream.id}): {cbs_wcd:.2f} us")

# SP Analytical WCDs
print("\nSP Analytical WCDs:")
for stream in streams:
    sp_wcd = calculate_sp_delay(stream, streams)
    print(f"Stream {stream.name} (ID {stream.id}): {sp_wcd:.2f} us")



# Simulation 
print("\nSimulated Results:")

results = simulate(streams)

for stream_id, delays in results.items():
    max_delay = max(delays)
    avg_delay = sum(delays) / len(delays)
    print(f"Stream ID {stream_id}: Max Delay = {max_delay:.2f} us, Avg Delay = {avg_delay:.2f} us")
