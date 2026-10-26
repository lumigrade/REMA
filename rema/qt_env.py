"""Make PySide6's X11 (xcb) platform plugin loadable without system changes.

Qt >= 6.5 needs libxcb-cursor.so.0, which this host lacks. A copy unpacked from the official Ubuntu
package (apt download libxcb-cursor0) lives under $REMA_DATA/deps/libxcb-cursor0. Loading it with
RTLD_GLOBAL before QApplication lets the xcb plugin resolve it by soname. Call before importing Qt widgets.
"""
import ctypes
import ctypes.util

from rema import paths as _P

LOCAL = _P.DATA / 'deps/libxcb-cursor0/root/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0'


def ensure_xcb_cursor():
    if ctypes.util.find_library('xcb-cursor'):
        return 'system'
    if LOCAL.exists():
        ctypes.CDLL(str(LOCAL), mode=ctypes.RTLD_GLOBAL)
        return str(LOCAL)
    return None
