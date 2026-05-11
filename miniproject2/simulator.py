from collections import defaultdict
from config import *
from analysis import transmission_delay_us


class Packet:
    def __init__(self, stream, release_time, bandwidth_mbps=100):
        self.stream = stream
        self.release_time = release_time
        self.tx_time = transmission_delay_us(stream.size, bandwidth_mbps)


# Event-driven CBS simulator (single output port)
def simulate(streams, bandwidth_mbps=100, duration_us=1000000):

    queues = {
        'AVB_A': [],
        'AVB_B': [],
        'BE': []
    }

    credits = {
        'AVB_A': 0.0,
        'AVB_B': 0.0
    }

    results = defaultdict(list)

    # Generate packets for each periodic stream
    packets = []
    for stream in streams:
        if stream.period is None or stream.period <= 0:
            continue
        t = 0
        while t < duration_us:
            packets.append(Packet(stream, t, bandwidth_mbps))
            t += stream.period

    packets.sort(key=lambda p: p.release_time)

    packet_index = 0
    time = 0.0

    while time < duration_us:
        # Release packets that have arrived by current time
        while packet_index < len(packets) and packets[packet_index].release_time <= time:
            p = packets[packet_index]
            if p.stream.pcp == 2:
                queues['AVB_A'].append(p)
            elif p.stream.pcp == 1:
                queues['AVB_B'].append(p)
            else:
                queues['BE'].append(p)
            packet_index += 1

        selected = None
        selected_class = None

        # CBS scheduling: pick highest-priority queue with credit >= 0
        if queues['AVB_A'] and credits['AVB_A'] >= 0:
            selected = queues['AVB_A'].pop(0)
            selected_class = 'AVB_A'
        elif queues['AVB_B'] and credits['AVB_B'] >= 0:
            selected = queues['AVB_B'].pop(0)
            selected_class = 'AVB_B'
        elif queues['BE']:
            selected = queues['BE'].pop(0)
            selected_class = 'BE'

        if selected:
            tx_time = selected.tx_time
            finish_time = time + tx_time

            # Update credits for the transmitting class (send slope)
            if selected_class in ('AVB_A', 'AVB_B'):
                credits[selected_class] += SEND_SLOPE * tx_time

            # Update credits for non-selected AVB classes (idle slope)
            # Only accumulate if that queue has waiting packets
            for cls in ('AVB_A', 'AVB_B'):
                if cls != selected_class and queues[cls]:
                    credits[cls] += IDLE_SLOPE * tx_time

            # Record response time
            response_time = finish_time - selected.release_time
            results[selected.stream.id].append(response_time)

            time = finish_time
        else:
            # No packet ready - advance time
            next_time = duration_us

            # Check if any AVB queue needs credit recovery
            for cls in ('AVB_A', 'AVB_B'):
                if queues[cls] and credits[cls] < 0:
                    recovery_time = abs(credits[cls]) / IDLE_SLOPE
                    candidate = time + recovery_time
                    if candidate < next_time:
                        next_time = candidate

            # Check next packet arrival
            if packet_index < len(packets):
                if packets[packet_index].release_time < next_time:
                    next_time = packets[packet_index].release_time

            if next_time <= time:
                next_time = time + 0.001

            # Accumulate idle slope credit during idle time
            elapsed = next_time - time
            for cls in ('AVB_A', 'AVB_B'):
                if queues[cls]:
                    credits[cls] += IDLE_SLOPE * elapsed

            # Reset credit to 0 if queue is empty (CBS spec)
            for cls in ('AVB_A', 'AVB_B'):
                if not queues[cls] and credits[cls] > 0:
                    credits[cls] = 0.0

            time = next_time

    return results
