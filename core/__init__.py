import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/')
LOGS_DIR = os.path.join(BASE_DIR, 'logs/')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints/')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')