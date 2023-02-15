from dotenv import load_dotenv, dotenv_values
from abc import ABC
from pathlib import Path
from .logging_utils import LogBuilder

from typing import Union, NoReturn
import mdtraj as md
import openmm as mm
from openmm import app
import openmm.unit as U
import pdbfixer

class Configuration(ABC):
    def __init__(self):
        raise NotImplementedError


class StdConfiguration(Configuration):
    """
    A container for all settings applied to individual simulations.

    Recommended to manipulate only the settings containe in the .env
    file. Which allows each individual "run" to be given a name. The 
    logging configuration of the simulation can also be controlled.

    Start-stop simulations are enabled through use of the 
    `import_existing_data` method, which loads a .pdb file 
    from the data directory with the filename supplied.
    """
    def __init__(self) -> None:
        self.project_dir = Path(__file__).parent.parent.parent 

        dotenv_path = Path(self.project_dir, ".env")
        assert dotenv_path.exists(), f"No .env file exists in the expected directory! [{dotenv_path}]"

        self.env_vars = dotenv_values(dotenv_path)
        self.logger = LogBuilder.create_sublogger(
            self.project_dir,
            "Configuration", 
            self.env_vars
        )


    def import_existing_data(self, filename: str) -> md.Trajectory:
        """
        Load an existing trajectory from a <filename>.pdb 
        file stored in the data directory. Useful for when 
        a simulation is interrupted and needs to be continued
        """
        return md.load(Path(self.project_dir, "data", filename))


if __name__ == "__main__":
    a = StdConfiguration()
    # print(a.logger.__dict__)
    a.logger.error("Some interesting message here")