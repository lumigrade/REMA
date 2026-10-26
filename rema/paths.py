"""Host-independent project/data locations.

Override with environment variables when the checkout or data root moves:
  REMA_ROOT     source checkout (default: this file's parent directory)
  REMA_DATA     ext4 data root for envs, recordings, logs, models (default: ~/rema-data)
  REMA_OUTPUTS  user-facing videos/reports (default: $REMA_DATA/outputs)
  REMA_SHARED_LOCKS  optional other-project per-GPU lock directory; respected only if it exists

Machine-specific facts (GPU UUIDs, live GPU choice) live in $REMA_DATA/config/host.json,
never in source.
"""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('REMA_ROOT', Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get('REMA_DATA', Path.home() / 'rema-data'))
OUTPUTS = Path(os.environ.get('REMA_OUTPUTS', DATA / 'outputs'))
SHARED_LOCKS = Path(os.environ.get('REMA_SHARED_LOCKS', Path.home() / 'vto-data' / 'locks'))
WORK = DATA / 'work'
REPORTS = DATA / 'reports'
LOCKS = DATA / 'locks'
CONFIG = DATA / 'config'
HOST_CONFIG = CONFIG / 'host.json'
RESERVE_VOLUMES = (DATA, ROOT)   # 25 GiB free-space reserve applies to both
RESERVE_BYTES = 25 * 1024 ** 3


def host_config():
    """Inventoried machine facts. Missing file means no GPU has been inventoried."""
    try:
        return json.loads(HOST_CONFIG.read_text())
    except FileNotFoundError:
        return {}


def disk_reserve_ok():
    import shutil
    return all(shutil.disk_usage(v).free >= RESERVE_BYTES for v in RESERVE_VOLUMES if Path(v).exists())
