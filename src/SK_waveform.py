import os
from waveloc.filters import sw_kurtosis2

def read_waveform(filename):
    """
    Read waveform using obspy routines
    """
    from obspy.core import read
    st=read(filename)
    return st

def process_stream_kurtosis(st, kwin):
    """
    Take sliding-window kurtosis of stream st with window length kwin.
    """
    # use first trace to get dt
    tr = st[0]

    # retrieve dt
    dt=tr.stats.delta

    # get length of kwin in points
    nwin = int(kwin/dt)

    # process all traces in stream
    for tr in st:
        xs = sw_kurtosis2(tr.data,nwin)
        tr.stats.starttime = tr.stats.starttime + (nwin-1)*dt
        tr.data=xs
