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

        system=top.createSystem()

        non_bonded_force = mm.openmm.CustomNonbondedForce("(sigma/r)^6")
        non_bonded_force.addGlobalParameter('sigma', 0.2)
        non_bonded_force.setCutoffDistance(0.65*U.nanometer)
        system.addForce(non_bonded_force)
        
        system.addForce(mm.openmm.CMMotionRemover())

        # Cache the starting system
        with open(self.home_dir / "initial_system.xml", "w") as  f:
            _xml = mm.XmlSerializer.serialize(system)
            f.write(_xml)

        integrator = mm.LangevinIntegrator(
            float(self.vars["LANGEVIN_TEMPERATURE"])*mm.unit.kelvin,
            float(self.vars["LANGEVIN_FRICTION_COEFF"])/mm.unit.picosecond, 
            float(self.vars["LANGEVIN_TIMESTEP"])*mm.unit.picoseconds
        )

        simulation = app.Simulation(top.topology, system, integrator)
        simulation.context.setPositions(pos.positions)

        return simulation


    # def SbmFuckery():
    #     sbm = SBM(
    #         name=self.vars["RUN_NAME"], 
    #         time_step=self.vars["TIME_STEP"], 
    #         collision_rate=self.vars["COLLISION_RATE"], 
    #         r_cutoff=self.vars["CUTOFF_RADIUS"], 
    #         temperature=0.5
    #     )

    #     sbm.setup_openmm(
    #         platform=self.vars["PLATFORM"]
    #     )

    #     sbm.saveFolder(f'{self.vars["RUN_NAME"]}-output')

    #     sbm_grofile = Path(self.configuration.project_dir, 'data', self.vars["RUN_NAME"], f'{self.vars["BASE_NAME"]}.gro')
    #     sbm_topfile = Path(self.configuration.project_dir, 'data', self.vars["RUN_NAME"], f'{self.vars["BASE_NAME"]}.top')
    #     sbm_xmlfile = Path(self.configuration.project_dir, 'data', self.vars["RUN_NAME"], f'{self.vars["BASE_NAME"]}.xml')

    #     sbm.loadSystem(Grofile=sbm_grofile, Topfile=sbm_topfile, Xmlfile=sbm_xmlfile)



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
