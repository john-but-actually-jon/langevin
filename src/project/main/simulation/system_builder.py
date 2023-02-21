from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

import openmm as mm
from openmm import app
from openmm import unit as U
from pdbfixer import PDBFixer
import mdtraj as md

from OpenSMOG import SBM

from .logging_utils import LogBuilder
from .configuration import Configuration, StdConfiguration

class SystemHandler(ABC):
    @abstractmethod
    def __init__(self):
        self.configuration: Configuration 

    @abstractmethod
    def build_simulation(self):
        raise NotImplementedError

class SMOGSystem(SystemHandler):
    """
    Generates a simulation from GROMACS files generated through SMOG. 
    """

    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.configuration = StdConfiguration()
        self.vars = self.configuration.env_vars
        self.logger = LogBuilder.create_sublogger(
            "SystemBuilder", 
            self.vars
        )
        # File structure should be handled by the logger initialization
        self.home_dir = self.data_dir / self.vars["CONFIGURATION_NAME"] / self.vars["RUN_NAME"]
        assert self.home_dir.exists(), "Run directory does not exist, system initialization failed!"


    def build_simulation(self, gromacs_dir: str = 'base') -> mm.System:
        """
        Build the system from the GROMACS files found in `gromacs_dir`. 
        This value defaults to the `data/base` directory  
        
        Args:
            - `gromacs_dir` (req): The directory containing the GROMACS-format
            files downloaded from SMOG. 
        """
        gromacs_directory = self.data_dir / gromacs_dir
        assert gromacs_directory.exists(), f"No directory named {gromacs_dir} in {self.data_dir}!"

        gro = app.GromacsGroFile(str(gromacs_directory / f"{gromacs_dir}.gro"))
        top = app.GromacsTopFile(str(gromacs_directory / f"{gromacs_dir}.top"))

        system=top.createSystem(
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=0.65*U.nanometer,
            removeCMMotion=True,
            ewaldErrorTolerance=0.0005
        )

        system.setDefaultPeriodicBoxVectors(
            *[vec * 2 for vec in gro.getPeriodicBoxVectors()]
        )

        # Cache the starting system
        with open(self.home_dir / "initial_system.xml", "w") as  f:
            _xml = mm.XmlSerializer.serialize(system)
            f.write(_xml)

        integrator = mm.LangevinIntegrator(
            float(self.vars["LANGEVIN_TEMPERATURE"])*mm.unit.kelvin,
            float(self.vars["LANGEVIN_FRICTION_COEFF"])/mm.unit.picosecond, 
            float(self.vars["LANGEVIN_TIMESTEP"])*mm.unit.picoseconds
        )

        integrator.setRandomNumberSeed(int(self.vars["RANDOM_NUMBER_SEED"]))

        simulation = app.Simulation(top.topology, system, integrator)
        simulation.context.setPositions(gro.positions)

        for reporter in LogBuilder.create_simulation_loggers(self.vars):
            simulation.reporters.append(reporter)

        return simulation

class ExternalSourceHandler(SystemHandler):
    """ 
    Used for when the data being imported doesn't 
    reside in the data directory, but rather 
    an external source
    """
    def __init__(self, path_to_data: Path):
        self.data_dir = path_to_data
        self.configuration = StdConfiguration()
        self.vars = self.configuration.env_vars
        self.logger = LogBuilder.create_sublogger(
            "SystemBuilder", 
            self.vars
        )
        # File structure should be handled by the logger initialization
        self.home_dir = self.data_dir / self.vars["CONFIGURATION_NAME"] / self.vars["RUN_NAME"]
        assert self.home_dir.exists(), "Run directory does not exist, system initialization failed!"

    def load_trajectory(self):
        pass

    def build_simulation(self):
        return super().build_simulation()


def SomeHandler(SystemHandler):
    """Not sure where to put these methods right now"""

    def build_simulation(self, system, struc): 

        integrator = mm.openmm.LangevinIntegrator(
            300*U.kelvin, 
            1/U.picosecond, 
            float(self.configuration.env_vars["LANGEVIN_TIMESTEP"]) * U.picoseconds
        )

        platform = mm.Platform.getPlatformByName(self.configuration.env_vars["SIMULATION_PLATFORM"])
        simulation = app.simulation.Simulation(
            struc.topology,
            system,
            integrator,
            platform
        )
        simulation.context.setPositions(struc.positions)

        simulation.minimizeEnergy()

        return simulation
    def fix_pdb(self, pdb_filename: Path, output_filename: Optional[str] = None) -> None:
        struc = PDBFixer(
            str(Path(self.configuration.project_dir, "data", pdb_filename))
        )
        self.logger.info(
            f"Topology loaded from {pdb_filename}: {struc.topology.__repr__().strip('Topology; <>')}"
        )
        struc.addMissingHydrogens()
        struc.findMissingResidues()
        struc.findMissingAtoms()
        struc.findNonstandardResidues()
        self.logger.info(f"Missing residues: {struc.missingResidues}")
        self.logger.info(f"Missing atoms: {struc.missingAtoms}")
        self.logger.info(f"Nonstandard residues: {struc.nonstandardResidues}")
        self.logger.info(f"Updated Topology: {struc.topology.__repr__().strip('Topology; <>')}")
        with open(output_filename, 'w') as f:
            pass





if __name__ == "__main__":
    # b = StdConfiguration()
    pass
    # a.logger.info("System build info message")
    # b.logger.info("Configuration info message")
