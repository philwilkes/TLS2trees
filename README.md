# TLS point cloud semantic classification and individual tree segmentation

This repository describes methods to extract individual trees from TLS point clouds. This is done using a 3-step process

### 0. Setup

From the repository's root directory, run `pip install -e .` to install the project as a local, editable module.

### 1. rxp-pipeline

This step preprocesses data captured with RIEGL VZ TLS data. Code and details can be found [here](https://github.com/philwilkes/rxp-pipeline). 

### 2. FSCT _lite_

This is forked from [@SKrisanski](https://github.com/SKrisanski/FSCT) and is a _lite_ version that only runs the semantic segmentation (ground, wood, leaf, cwd). Typical usage is:

`python run.py -p <point_cloud> --tile-index <path_to_index> --buffer <buffer> --verbose`

```
optional arguments:
  -h, --help            show this help message and exit
  --point-cloud POINT_CLOUD, -p POINT_CLOUD
                        path to point cloud
  --params PARAMS       path to pickled parameter file
  --odir ODIR           output directory
  --step STEP           which process to run to
  --redo REDO           which process to run to
  --tile-index TILE_INDEX
                        path to tile index in space delimited format "TILE X Y"
  --buffer BUFFER       included data from neighbouring tiles
  --batch_size BATCH_SIZE
                        If you get CUDA errors, try lowering this.
  --num_procs NUM_PROCS
                        Number of CPU cores you want to use. If you run out of RAM, lower this.
  --keep-npy            Keeps .npy files used for segmentation after inference is finished.
  --output_fmt OUTPUT_FMT
                        file type of output
  --verbose             print stuff
  ```

### 3. Instance segmentation to extract individual trees

Following classification an instance segmenation can be run to seperate individual trees using:

`python points2trees.py -t 001.downsample.segmented.ply --tindex ../tile_index.dat -o ../tmp/ --n-tiles 5 --slice-thickness .5 --find-stems-height 2 --find-stems-thickness .5 --pandarallel --verbose --add-leaves --add-leaves-voxel-length .5 --graph-maximum-cumulative-gap 3 --save-diameter-class --ignore-missing-tiles`

```
optional arguments:
  -h, --help            show this help message and exit
  --tile TILE, -t TILE  fsct directory
  --odir ODIR, -o ODIR  output directory
  --tindex TINDEX       path to tile index
  --n-tiles N_TILES     enlarges the number of tiles i.e. 3x3 or tiles or 5 x 5 tiles
  --overlap OVERLAP     buffer to crop adjacent tiles
  --slice-thickness SLICE_THICKNESS
                        slice thickness for constructing graph
  --find-stems-height FIND_STEMS_HEIGHT
                        height for identifying stems
  --find-stems-thickness FIND_STEMS_THICKNESS
                        thickness of slice used for identifying stems
  --find-stems-min-radius FIND_STEMS_MIN_RADIUS
                        minimum radius of found stems
  --find-stems-min-points FIND_STEMS_MIN_POINTS
                        minimum number of points for found stems
  --graph-edge-length GRAPH_EDGE_LENGTH
                        maximum distance used to connect points in graph
  --graph-maximum-cumulative-gap GRAPH_MAXIMUM_CUMULATIVE_GAP
                        maximum cumulative distance between a base and a cluster
  --min-points-per-tree MIN_POINTS_PER_TREE
                        minimum number of points for a identified tree
  --add-leaves          add leaf points
  --add-leaves-voxel-length ADD_LEAVES_VOXEL_LENGTH
                        voxel sixe when add leaves
  --add-leaves-edge-length ADD_LEAVES_EDGE_LENGTH
                        maximum distance used to connect points in leaf graph
  --save-diameter-class
                        save into dimeter class directories
  --ignore-missing-tiles
                        ignore missing neighbouring tiles
  --pandarallel         use pandarallel
  --verbose             print something
```

## Docker

To build a Docker container with all the libraries installed use:
```
docker build -t tls2trees:latest .
```
Then to run FSCT and the instance segmentation use:
```
docker run -it -v /path/to/data/outsidecontainer:/path/to/data/incontainer fsct:latest run.py
docker run -it -v /path/to/data/outsidecontainer:/path/to/data/incontainer fsct:latest points2trees.py
```

For HPC systems, where you don't have permission to run Docker, you can build the container on your local machine and convert to a singularity file using:

```
sudo singularity build tls2trees_latest.sif docker-daemon://tls2trees:latest
```
Copy this to the HPC system and run this using
```
singularity exec tls2trees_latest.sif run.py
singularity exec tls2trees_latest.sif points2trees.py
```
