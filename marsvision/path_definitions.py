import os
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config.yml"
)


parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
PDSC_TABLE_PATH = os.path.join(
    parent_dir,
    "pdsc_tables"
)