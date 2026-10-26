#!/usr/bin/env python3
"""Astra Pro metric depth -> stdout stream (run by rema.astra_depth.DepthStream, not by hand).

Runs in its own process so the exact-device USB descriptor workaround (LD_PRELOAD astra_usb_compat.so)
and OpenNI stay out of the live compositor. Each frame: 32-byte header '<4sIQQf4x'
(b'RDEP', width*height, seq, host CLOCK_MONOTONIC ns, metres per unit) + uint16 depth (640x480).
Stops when stdout closes or on any error (message on stderr).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import ctypes as C
import struct
import sys
import time

import numpy as np

HEADER = struct.Struct('<4sIQQf4x')


def main():
    lib = C.CDLL(str(_P.WORK / 'openni_live_bridge.so'))
    lib.rema_open.argtypes = [C.c_char_p, C.c_int]
    lib.rema_open.restype = C.c_void_p
    lib.rema_read.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64), C.POINTER(C.c_float), C.c_char_p, C.c_int]
    lib.rema_read.restype = C.c_int
    lib.rema_close.argtypes = [C.c_void_p]
    err = C.create_string_buffer(1024)
    ptr = lib.rema_open(err, len(err))
    if not ptr:
        sys.exit('astra depth open failed: ' + err.value.decode())
    out = sys.stdout.buffer
    raw = np.empty((480, 640), np.uint16)
    seq, stamp, scale = C.c_uint64(), C.c_uint64(), C.c_float()
    misses = 0
    try:
        while True:
            r = lib.rema_read(ptr, raw.ctypes.data, C.byref(seq), C.byref(stamp), C.byref(scale), err, len(err))
            now = time.monotonic_ns()
            if r < 0:
                sys.exit('astra depth read failed: ' + err.value.decode())
            if r == 0:
                misses += 1
                if misses >= 3:
                    sys.exit('astra depth stream timeout')
                continue
            misses = 0
            out.write(HEADER.pack(b'RDEP', raw.size, seq.value, now, scale.value / 1000.0) + raw.tobytes())
            out.flush()
    except BrokenPipeError:
        pass
    finally:
        lib.rema_close(ptr)


if __name__ == '__main__':
    main()
