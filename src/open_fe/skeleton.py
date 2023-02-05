"""This is a skeleton file that can serve as a starting point for a Python console script. To run this script
uncomment the following lines in the ``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = open_fe.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys

from open_fe.core import OpenFE
from open_fe.data_center import DataCenter
from open_fe.model_center import ModelCenter
from open_fe.trainer_center import TrainerCenter
from open_fe.utils import load_cfg, parse_args, setup_logging

_logger = logging.getLogger(__name__)


def main(args_: list[str]) -> None:
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion.

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args: argparse.Namespace = parse_args(args_)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")

    # read cfg
    cfg = load_cfg(args.config)

    # read data
    datasets = DataCenter.parse(cfg).build()
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]
    submit_dataset = datasets["submit"]

    # build meta model
    model = ModelCenter.parse(cfg).build()

    # build trainer
    trainer = TrainerCenter.parse(cfg).build().attach(model)

    # without openfe
    predictions = trainer.fit_then_predict(train_dataset, test_dataset)
    submit_dataset["predictions"] = predictions
    submit_dataset.save("submission_without_openfe.csv")

    # with openfe
    openfe = OpenFE.parse(cfg).build()
    openfe.fit(train_dataset)
    train_dataset = openfe.transform(train_dataset)
    test_dataset = openfe.transform(test_dataset)
    predictions = trainer.fit_then_predict(train_dataset, test_dataset)
    submit_dataset["predictions"] = predictions
    submit_dataset.save("submission_with_openfe.csv")

    _logger.info("Script ends here")


def run() -> None:
    """Calls :func:`main` passing the CLI arguments extracted from
    :obj:`sys.argv`

    This function can be used as entry point to create console scripts
    with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m open_fe.skeleton --config configs/config.yaml
    #
    run()
