import logging
from pathlib import Path
from typing import OrderedDict

class LogBuilder:
    
    @staticmethod
    def create_sublogger(
        project_dir: Path,
        component_name: str, 
        env_vars: OrderedDict
    ) -> logging.Logger:
        """
        The purpose of this is to create a new logger for each 
        meaningful component of the simulation process. This means
        using a unified configuration supplied in the .env file 
        and assigning a unique name for each component.

        Args:
            - `project_dir` [req]: The `Path` object containing the 
            address for the directory that contains the logs directory,
            where logs will be saved. 
            - `component_name` [req]: The name of the subcomponent.
            This name is appended before the `RUN_NAME` .env variable
            when each log entry is recorded.
            - `env_vars` [req]: The OrderedDict returned from a call
            to `dotenv_values
        """
        project_dir = Path(__file__).parent.parent.parent
        logger = logging.getLogger(
            component_name + "|" + env_vars["RUN_NAME"]+"> "
        )
        logger.setLevel(env_vars["LOG_LEVEL"])
        log_formatter = logging.Formatter(env_vars["LOG_FORMAT"])
        log_formatter.default_msec_format = "" # remove milliseconds
        log_dir = Path(project_dir, "logs")

        _fh = logging.FileHandler(Path(log_dir, env_vars["LOG_FILE_NAME"]))
        _fh.setFormatter(log_formatter)
        logger.addHandler(_fh)
        if env_vars["LOG_TO_TERMINAL"].lower() == "true":
            _ch = logging.StreamHandler()
            _ch.setFormatter(log_formatter)
            logger.addHandler(_ch)
        return logger

