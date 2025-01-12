import sys
import os

def add_src_to_path():
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )