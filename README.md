# Forest Structural Complexity Tool
### Created by Sean Krisanski
## Purpose of this tool

This tool was written for the purpose of allowing plot based measurements to be extracted automatically from most
high-resolution forest point clouds from a variety of sensor sources. Such sensor types it works on include
Terrestrial Laser Scanning (TLS), Mobile Laser Scanning (MLS), Terrestrial Photogrammetry, Above and below-canopy
UAS Photogrammetry or similar. 

## References

The deep learning component uses Pytorch https://pytorch.org/ and Pytorch-Geometric 
https://pytorch-geometric.readthedocs.io/en/latest/#

The first step is semantic segmentation of the forest point cloud. This is performed using a modified version of
Pointnet++ https://github.com/charlesq34/pointnet2 using the implementation in Pytorch-Geometric as a starting point
provided here: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py


## Citation
### If you find this tool helpful or use this tool in your research, please cite these two papers:

**The semantic segmentation tool is described here:**
\
Krisanski, S.; Taskhiri, M.S.; Gonzalez Aracil, S.; Herries, D.; Turner, P. Sensor Agnostic Semantic Segmentation of
Structurally Diverse and Complex Forest Point Clouds Using Deep Learning. Remote Sens. 2021, 13, 1413.
https://doi.org/10.3390/rs13081413

\
**The measurement tool is described here:**
\
Krisanski, S.; Taskhiri, M.S.; Gonzalez Aracil, S.; Herries, D.; Turner, P. Forest Structural Complexity Tool - An Open
Source, Fully-Automated Tool for Measuring Forest Point Clouds. Remote Sens. 2021, XX, XXXX. 
https://doi.org/XX.XXXX/rsXXXXXXXX

## Acknowledgements
This research was funded by the Australian Research Council - Training Centre for Forest Value 
(University of Tasmania, Australia).

Thanks to my supervisory team Assoc. Prof Paul Turner and Dr. Mohammad Sadegh Taskhiri from the eLogistics Research
Group and Dr. James Montgomery from the University of Tasmania.

Thanks to Susana Gonzalez Aracil and David Herries from Interpine Group Ltd. (New Zealand), who provided a number of the raw point
clouds and plot measurements used during the development and validation of this tool.


## Contributing
Interested in contributing to this code? Get in touch! This code is likely far from optimal, so if you find errors or 
have ideas/suggestions on improvements, they would be very welcome!

## How to use
It is strongly recommended to have a CUDA compatible GPU (Nvidia) for running this tool. This can be run on CPU
only, but expect inference to take a long time.

### Recommended PC Specifications
**Warning: FSCT is computationally expensive in its current form.** Fortunately, it is still considerably faster than a human 
at what it does.

It should be able to be run on most modern gaming desktop PCs (or particularly powerful laptops), however, it will
take a while if you are running it on a lesser setup than below.

I use the following setup and the computational times are bearable:
- CPU: Intel i9-10900K Overclocked to 4.99GHz all cores.
- GPU: Nvidia Titan RTX (24 GB vRAM)
- RAM: 128 GB DDR4 at 3200 MHz (NOTE: this is often not enough on large point clouds, so I have a 500 GB page file to assist if I run out of RAM).
- SSD: M.2 NVMe 2 TB, 3500 MB/s read,  3000 MB/s write.

Hopefully in time, I'll be able to make this more efficient and less resource hungry.

### User Parameters
#### batch_size
The number of samples in a batch used for the deep learning inference. This number depends on the amount of GPU RAM you
have. If you set this too high, you will run out of GPU RAM. As a rough guide, I can fit 18-20 on an Nvidia Titan RTX GPU with 24 GB GPU
RAM.

#### num_procs
The number of CPU cores you have/wish to use.

#### max_diameter
Fairly self-explanatory. Trims away any diameter measurements larger than this value. This is rarely needed, so 
generally just leave it much larger than the largest tree in your plot. I generally don't change it.




#### slice_thickness



    """
    If you want to select individual files, leave directory_mode set to 0 or False.
    If you want to process ALL '.las' files within a directory and its sub-directories, set directory_mode to 1 or True.
    "Directory mode" will ignore FSCT_output '.las' files. """
    directory_mode = 0

    root = tk.Tk()
    if directory_mode:
        point_clouds_to_process = []
        directory = fd.askdirectory(parent=root, title='Choose directory')
        unfiltered_point_clouds_to_process = glob.glob(directory + '/**/*.las', recursive=True)
        for i in unfiltered_point_clouds_to_process:
            if 'FSCT_output' not in i:
                point_clouds_to_process.append(i)
    else:
        point_clouds_to_process = fd.askopenfilenames(parent=root, title='Choose files', filetypes=[("LAS", "*.las"), ("LAZ", "*.laz"), ("CSV", "*.csv")])
    root.destroy()

    for point_cloud in point_clouds_to_process:
        print(point_cloud)

        parameters = dict(input_point_cloud=point_cloud,
                          batch_size=18,  # If you get CUDA errors, lower this. This is suitable for 24 GB of vRAM.
                          num_procs=20,  # Number of CPU cores you want to use.
                          max_diameter=5,  # Maximum diameter setting. Any measurements greater than this are considered erroneous and are ignored.
                          slice_thickness=0.2,  # default = 0.2
                          slice_increment=0.05,  # default = 0.05
                          slice_clustering_distance=0.1,  # default = 0.1
                          cleaned_measurement_radius=0.18,
                          minimum_CCI=0.3,  # Minimum valid Circuferential Completeness Index (CCI) for non-interpolated circle/cylinder fitting. Any measurements with CCI below this are deleted.
                          min_tree_volume=0.005,  # Measurements from trees with volume (m3) less than this are ignored in the outputs.
                          ground_veg_cutoff_height=3,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=5,
                          Site='not_specified',  # Enter the site name if you wish. Only used for report generation.
                          PlotID='not_specified',  # Enter the plot name/ID if you wish. Only used for report generation.
                          plot_centre=None,  # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is the median XY coords of the point cloud.
                          plot_radius=0,  # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=0,  # See README.md  This is used for "Intelligent Plot Cropping Mode".
                          UTM_zone_number=50,  # Self explanatory.
                          UTM_zone_letter=None,  # Self explanatory.
                          UTM_is_north=False,   # If in the northern hemisphere, set this to True.
                          filter_noise=0,
                          low_resolution_point_cloud_hack_mode=0)  # See README.md for details. Dodgy hack that can be useful on low resolution point clouds. Approximately multiplies the number of points in the point cloud by this number.


### Other Parameters (Changing these parameters is not recommended)



### Intelligent Plot Cropping
The purpose of this mode is to trim a point cloud to a specified plot radius, without removing points from trees that
are considered to be in the plot.

We first trim the point cloud to a radius where the initial trim radius = plot_radius + plot_radius_buffer.
Next, all main steps are run on this point cloud. Finally, we check which detected trees are within the plot,
keep the vegetation and stem points associated with those trees inside the plot, while trimming all ground vegetation
and terrain points to the specified plot_radius.



### Low resolution hack mode...
This model was trained on relatively high resolution point clouds, so if a stem is of sufficiently low resolution,
it will likely be classified as vegetation instead. Eventually, I will train this model on a larger training dataset
with more examples of sparse point clouds, however, in the meantime, I came up with an abomination of a hack that
sometimes helps a little when working with datasets of low resolution which this tool was not really designed for.

Low resolution hack mode will copy the input point cloud, jitter the points in random directions by 1 cm, then join this
copied point cloud to the original point cloud. This gives the model more points to work with, which it tends to be
happier about...
Once inference is complete, the original point cloud is returned.