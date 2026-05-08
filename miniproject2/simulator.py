import heapq
from collections import defaultdict
from config import *

from analysis import transmission_delay_us

class Packet: 
    def __init__(self, stream, release_time):
        self.stream = stream
        self.release_time = release_time
        self.remaining = transmission_delay_us(stream.size)



# Event driven CBS simulator

def simulate(streams, duration_us=1000000):
    
    
    queues = {
        'AVB_A': [],
        'AVB_B': [],
        'BE': []
    }

    credits = {
        'AVB_A': 0,
        'AVB_B': 0
    }

    results = defaultdict(list)
    

    # generate packets for each stream
    packets = []
    for stream in streams:
        t= 0 
        while t < duration_us:
            packets.append(Packet(stream, t))
            t += stream.period
    
    packets.sort(key=lambda p: p.release_time)

    packet_index = 0
    time = 0


    while time < duration_us:
        #release packets
        while (packet_index < len(packets) and packets[packet_index].release_time <= time):
            p = packets[packet_index]
            
            if p.stream.pcp == 2:
                queues['AVB_A'].append(p)
            elif p.stream.pcp == 1:
                queues['AVB_B'].append(p)
            else:
                queues['BE'].append(p)
            
            packet_index += 1

        selected = None
        
    

        # CBS scheduling
        if queues['AVB_A'] and credits['AVB_A'] >= 0:
            selected = queues['AVB_A'].pop(0)
            credits['AVB_A'] += SEND_SLOPE * selected.remaining


        elif queues['AVB_B'] and credits['AVB_B'] >= 0:
            selected = queues['AVB_B'].pop(0)
            credits['AVB_B'] += SEND_SLOPE * selected.remaining
            
        
        elif queues['BE']:
            selected = queues['BE'].pop(0)
            
        #recovery
        credits['AVB_A'] += IDLE_SLOPE
        credits['AVB_B'] += IDLE_SLOPE


        #if packet selected
        if selected:

            fisnish_time = time + selected.remaining

            response_time = fisnish_time - selected.release_time

            results[selected.stream.id].append(response_time)

            time = fisnish_time
        else:
            time += 1
        
    return results
            


