## Data Directory Structure

The (data directory)[./data] is organised to maximise the ease with which new runs can be started and old runs can be continued. The structure of the directory is as follows,

```
|-- data/
    |-- base_structure/
        |-- 2efv_fixed.pdb
        |-- base.gro
        |-- base.top
        |-- ...
    |-- configuration-name-1/
        |-- base-structure/
        |-- run_name_1/
            |-- run-name.pdb
            |-- run-name.log
            |-- run-name-pre.xml
        |-- run_name_2/
        |-- .../
    |-- configuration-name-2/
    |-- .../

```

Each simulation is therefore characterised by a `configuration-name` which corresponds to a specific `base-structure`, then each run is specified by a `run-name` which corresponds to a particular RNG seed.

Each of these names should be specified in the `.env` file.

The top-level `base_structure` directory contains the files downloaded from the SMOG server (GROMACS format) when the `2efv_fixed.pdb` file was uploaded and the course grained model was generated.