from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

import openmm as mm
from openmm import app
from openmm import unit as U
from pdbfixer import PDBFixer
import mdtraj as md

from .logging_utils import LogBuilder
from .configuration import Configuration, StdConfiguration

class SystemHandler(ABC):
    @abstractmethod
    def __init__(self):
        self.configuration: Configuration 

    @abstractmethod
    def build_system(self) -> mm.System:
        raise NotImplementedError

class NVTSystem(SystemHandler):
    """
    System handler for the NVT system
    """

    def __init__(self):
        self.configuration = StdConfiguration()
        self.logger = LogBuilder.create_sublogger(
            self.configuration.project_dir,
            "SystemBuild", 
            self.configuration.env_vars
        )

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

    
    def build_system(self, base_structure_filename: Path, ) -> mm.System:
        """"""
        struc = md.load_pdb(base_structure_filename)
        forcefield = app.ForceField('amber99sb.xml')
        system = forcefield.createSystem(
            struc.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0*U.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005
        )
        return system

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








if __name__ == "__main__":
    a = NVTSystem()
    a.build("2efv.pdb")
    # b = StdConfiguration()

    # a.logger.info("System build info message")
    # b.logger.info("Configuration info message")
