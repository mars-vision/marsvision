import os
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))

CONFIG_PATH = os.path.join(
    parent_dir,
    "data",
    "config.yml"
)

PDSC_TABLE_PATH = os.path.join(
    parent_dir,
    "data",
    "pdsc_tables"
)