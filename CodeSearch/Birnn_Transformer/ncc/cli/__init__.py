import os
from ncc import __DEFAULT_DIR__

DEMO_MODEL_DIR = os.path.join(__DEFAULT_DIR__, 'demo')
os.makedirs(DEMO_MODEL_DIR, exist_ok=True)

__all__ = (
    DEMO_MODEL_DIR,
)
