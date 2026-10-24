"""Make the backend modules importable from the tests regardless of launch directory.

Adds app/backend/ to sys.path so `import lane_status`, `import stats`, `import data_loader`
work whether pytest is run from the repo root or from app/backend.
"""

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
