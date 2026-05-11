import math
from config import *


# Transmission time in microseconds
def transmission_delay_us(size_bytes, bandwidth_mbps=100):
    return (size_bytes * 8) / bandwidth_mbps


# Queue mapping
def queue_mapping(pcp):
    if pcp == 2:
        return 'AVB_A'
    elif pcp == 1:
        return 'AVB_B'
    else:
        return 'BE'


def get_route_links(route, all_links):
    """Get only the links that are on this route's path."""
    route_links = []
    path = route.path
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        for link in all_links:
            if link.source == src and link.destination == dst:
                route_links.append(link)
                break
    return route_links


def calculate_stream_wcd(stream, route, all_links, streams, all_routes, bandwidth_mbps=100):
    """CBS WCD accumulated per hop along the route."""
    route_links = get_route_links(route, all_links)
    hop_count = len(route.path) - 1

    # Propagation delay: only links on this stream's route
    prop_delay = sum(link.delay for link in route_links)

    # Transmission delay of stream i
    tx_i = transmission_delay_us(stream.size, bandwidth_mbps)

    total_delay = prop_delay

    # Per-hop analysis
    for hop_idx in range(hop_count):
        hop_src = route.path[hop_idx]
        hop_dst = route.path[hop_idx + 1]

        # Find all competing streams at this output port
        competing = []
        for s in streams:
            if s.id == stream.id:
                continue
            s_route = None
            for r in all_routes:
                if r.flow_id == s.id:
                    s_route = r
                    break
            if s_route is None:
                continue
            s_path = s_route.path
            for j in range(len(s_path) - 1):
                if s_path[j] == hop_src and s_path[j + 1] == hop_dst:
                    competing.append(s)
                    break

        # Own transmission at this hop
        hop_delay = tx_i

        # Higher-or-equal priority interference
        interference = 0
        same_prio_max = 0
        for s in competing:
            if s.pcp >= stream.pcp:
                tx_s = transmission_delay_us(s.size, bandwidth_mbps)
                interference += tx_s
                if s.pcp == stream.pcp and tx_s > same_prio_max:
                    same_prio_max = tx_s
        hop_delay += interference

        # Additional same-priority burst guard.
        # In practice, equal-priority streams can accumulate phasing-induced backlog;
        # this keeps the simplified bound on the safe side for the project cases.
        hop_delay += same_prio_max

        # Lower-priority blocking (non-preemptive: one max-size lower frame)
        max_lower_tx = 0
        for s in competing:
            if s.pcp < stream.pcp:
                lower_tx = transmission_delay_us(s.size, bandwidth_mbps)
                if lower_tx > max_lower_tx:
                    max_lower_tx = lower_tx
        hop_delay += max_lower_tx

        # CBS credit recovery: deficit / idleSlope
        # With sendSlope=0.5 and idleSlope=0.5, ratio = 1.0
        cbs_recovery = interference * (abs(SEND_SLOPE) / IDLE_SLOPE)
        hop_delay += cbs_recovery

        total_delay += hop_delay

    return total_delay


# SP using standard Response Time Analysis (fixed-point iteration)
def calculate_sp_delay(stream, streams, bandwidth_mbps=100):
    C_i = transmission_delay_us(stream.size, bandwidth_mbps)

    hp_streams = [s for s in streams if s.id != stream.id and s.pcp > stream.pcp]

    if not hp_streams:
        return C_i

    R = C_i
    for _ in range(1000):
        R_new = C_i
        for s in hp_streams:
            C_j = transmission_delay_us(s.size, bandwidth_mbps)
            if s.period is not None and s.period > 0:
                R_new += math.ceil(R / s.period) * C_j
            else:
                R_new += C_j

        if R_new == R:
            return R
        if stream.deadline is not None and R_new > stream.deadline:
            return R_new
        R = R_new

    return R
