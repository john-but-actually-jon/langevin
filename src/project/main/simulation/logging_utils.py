import logging
from pathlib import Path
from typing import OrderedDict

from openmm import app

class LogBuilder:

    def _configure_output_path(env_vars: OrderedDict) -> Path:
        path = Path(
            Path(__file__).parent.parent.parent, 
            "data", 
            env_vars["CONFIGURATION_NAME"], 
            env_vars["RUN_NAME"]
        )
        if not path.exists():
            if not path.parent.exists():
                print(f"Performing first time setup for {path.parent}")
                path.parent.mkdir()
            path.mkdir()
        return path
    
    @staticmethod
    def create_sublogger(
        component_name: str, 
        env_vars: OrderedDict
    ) -> logging.Logger:
        """
        The purpose of this is to create a new logger for each 
        meaningful component of the simulation process. This means
        using a unified configuration supplied in the .env file 
        and assigning a unique name for each component.

        Args:
            - `component_name` [req]: The name of the subcomponent.
            This name is appended before the `RUN_NAME` .env variable
            when each log entry is recorded.
            - `env_vars` [req]: The OrderedDict returned from a call
            to `dotenv_values
        """
        data_dir = LogBuilder._configure_output_path(env_vars)
        logger = logging.getLogger(
            component_name + "|" + env_vars["RUN_NAME"]+"> "
        )
        logger.setLevel(env_vars["LOG_LEVEL"])
        log_formatter = logging.Formatter(env_vars["LOG_FORMAT"])
        log_formatter.default_msec_format = "" # remove milliseconds

        _fh = logging.FileHandler(Path(data_dir, env_vars["LOG_FILE_NAME"]))
        _fh.setFormatter(log_formatter)
        logger.addHandler(_fh)
        if env_vars["LOG_TO_TERMINAL"].lower() == "true":
            _ch = logging.StreamHandler()
            _ch.setFormatter(log_formatter)
            logger.addHandler(_ch)
        return logger

    @staticmethod
    def create_simulation_loggers(
        env_vars: OrderedDict,
        ) -> list:
        """ 
        Use this to generate the standard simulation logging objects in 
        a list to loop over.

        Returns:
            [PDBReporter, DCDReporter, StateDataReporter]
        """
        data_dir = LogBuilder._configure_output_path(env_vars)

        pdb_logger = app.pdbreporter.PDBReporter(
            f"{data_dir}/{env_vars['RUN_NAME']}.pdb",
            int(env_vars["PDB_LOG_FREQUENCY"])
        )
        dcd_logger = app.dcdreporter.DCDReporter(
            f"{data_dir}/{env_vars['RUN_NAME']}.dcd",
            int(env_vars["DCD_LOG_FREQUENCY"])
        )
        state_reporter = app.statedatareporter.StateDataReporter(
            f"{data_dir}/{env_vars['LOG_FILE_NAME']}",
            int(env_vars["STATE_DATA_LOG_FREQUENCY"]),
            time=True,
            temperature=True,
            # progress=True,
            # remainingTime=True
        )
        return [pdb_logger, dcd_logger, state_reporter]
    

if __name__ == "__main__":
    from dotenv import dotenv_values
    _path = Path(Path(__file__).parent.parent.parent.absolute(), ".env")
    vars = dotenv_values(_path)
    LogBuilder.create_simulation_loggers(vars)

