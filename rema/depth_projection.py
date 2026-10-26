"""Optional native CPU registration, with the NumPy implementation as fallback."""
from rema import paths as _P
import ctypes
from pathlib import Path
import numpy as np

_PATH=Path(str(_P.DATA)+'/work/depth_projection.so')
_library=None
if _PATH.exists():
    _library=ctypes.CDLL(str(_PATH))
    _library.rema_project_depth.argtypes=[ctypes.c_void_p]*7
    _library.rema_project_depth.restype=None


def align_native(projection, depth):
    if _library is None or projection.dist.size not in (4,5):
        return None
    if projection.R.shape!=(3,3) or projection.t.shape!=(3,) or projection.K.shape!=(3,3) or projection.rays.shape!=(480*640,3):
        raise ValueError('Invalid registration matrix dimensions')
    if np.shape(depth)!=(480,640):
        raise ValueError('Native registration requires 640x480 depth')
    arrays=[np.ascontiguousarray(depth,dtype=np.float32),
            np.ascontiguousarray(projection.rays,dtype=np.float32),
            np.ascontiguousarray(projection.R,dtype=np.float64),
            np.ascontiguousarray(projection.t,dtype=np.float64),
            np.ascontiguousarray(projection.K,dtype=np.float64),
            np.ascontiguousarray(np.pad(projection.dist.ravel(),(0,5-projection.dist.size)),dtype=np.float64),
            np.empty((480,640),np.float32)]
    _library.rema_project_depth(*[ctypes.c_void_p(a.ctypes.data) for a in arrays])
    return arrays[-1]
