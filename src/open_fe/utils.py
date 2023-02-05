import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

Config = dict[str, Any]

__all__ = [
    "parse_args",
    "setup_logging",
    "get_extension",
    "load_cfg",
    "encode_category_features",
]


def parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command line parameters.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
        metavar="FILE",
        default="",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel: int) -> None:
    """Setup basic logging."""
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def get_extension(file_path: str) -> str:
    """Get extension from given file path.

    Parameters
    ----------
    file_path : str
        file path

    Returns
    -------
    str
        extension string
    """
    return Path(file_path).suffix


def load_cfg(config_path: str) -> Config:
    """Load yaml file into python dictionary.

    Parameters
    ----------
    config_path : str
        yaml file path

    Returns
    -------
    Config
        python dictionary
    """
    with open(config_path) as f:
        parsed_yaml: Config = yaml.safe_load(f)
    return parsed_yaml


def encode_category_features(features: pd.DataFrame) -> pd.DataFrame:
    for column in features.columns:
        if features[column].dtype in ["category", "object"]:
            features[column] = features[column].astype("category").cat.codes
    return features
