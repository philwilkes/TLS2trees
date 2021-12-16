# FSCT _lite_

This repo is forked from @SKrisanski and is a _lite_ version that only runs the semantic segmentation (ground, wood, leaf, cwd). Typical usage is:

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
  
Following classification an instance segmenation can be run to seperate individual trees with:

`python points2trees.py -i . --odir ../clouds/ --verbose --n-prcs 10 --add-leaves`

