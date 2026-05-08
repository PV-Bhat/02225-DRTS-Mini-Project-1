

from config import *


# Transmission time in microseconds
def transmission_delay_us(size_bytes, bandwith_mbps = 100):
    return (size_bytes *8) / bandwith_mbps



#Queue mapping
# PCP 2 -> AVB_A
# PCP 1 -> AVB_B
# PCP 0 -> BE

def queue_mapping(pcp):
    if pcp == 2:
        return 'AVB_A'
    elif pcp == 1:
        return 'AVB_B'
    else:
        return 'BE'
    
# Simple CBS delay calculation
def calculate_stream_wcd(stream, route, links, streams):
    total_delay = 0
    tx_delay = transmission_delay_us(stream.size)
    hop_count = len(route.path) - 1

    #base transmission
    for _ in range(hop_count):
        total_delay += tx_delay

    # add propagation delay
    for link in links:
        total_delay += link.delay
    

    # delays from higher or equal priority streams
    interference = 0

    for s in streams:
        if s.id == stream.id:
            continue
        if s.pcp >= stream.pcp:
            interference += transmission_delay_us(s.size)
    
    total_delay += interference

    # CBS credit recovery time
    cbs_recovery_time = interference * IDLE_SLOPE
    
    total_delay += cbs_recovery_time

    return total_delay


# SP comparison
def calculate_sp_delay(stream, streams):
    delay = transmission_delay_us(stream.size)

    for s in streams:
        if s.id == stream.id:
            continue
        if s.pcp > stream.pcp:
            delay += transmission_delay_us(s.size)

    return delay
    