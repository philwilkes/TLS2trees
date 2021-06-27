# Forest Structural Complexity Tool
## Created by Sean Krisanski
This tool was written for the purpose of processing high-resolution forest point clouds from a variety of sensor
sources including Terrestrial Laser Scanning (TLS), Mobile Laser Scanning (MLS), Terrestrial Photogrammetry, Above 
or below-canopy UAS Photogrammetry or similar. It is built using Pytorch https://pytorch.org/ and Pytorch-Geometric 
https://pytorch-geometric.readthedocs.io/en/latest/#

## General Concept
The first step is semantic segmentation of the forest point cloud. This is performed using a modified version of
Pointnet++ https://github.com/charlesq34/pointnet2 using the implementation in Pytorch-Geometric as a starting point
provided here: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py


##Purpose of this tool



## Citation
###If you find this tool helpful or use this tool in your research, please cite these two papers:

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


##Contributing##
Interested in contributing to this code? Get in touch! This code is likely far from optimal, so if you find errors or 
have ideas/suggestions on improvements, they would be very welcome!

##How to use
It is strongly recommended to have a CUDA compatible GPU (Nvidia) for running this tool. This can be run on CPU
only, but expect inference to take a long time.

###Recommended PC Specifications
**Warning: FSCT is computationally expensive in its current form.** Fortunately, it is still considerably faster than a human 
at what it does.

It should be able to be run on most modern gaming desktop PCs (or particularly powerful laptops), however, it will
take a while if you are running it on a lesser setup than below.

I use the following setup and the computational times are bearable:
- CPU: Intel i9-10900K Overclocked to 4.99GHz all cores.
- GPU: Nvidia Titan RTX (24GB vRAM)
- RAM: 128 GB DDR4 at 3200 MHz (NOTE: this is often not enough on large point clouds, so I have a 500 GB swap file to assist if I run out of RAM).

Hopefully in time, I'll be able to make this more efficient and less resource hungry.

###User Parameters
####batch_size
The number of samples in a batch used for the deep learning inference. This number depends on the amount of GPU RAM you
have. If you set this too high, you will run out of GPU RAM. As a rough guide, I can fit 18-20 on an Nvidia Titan RTX GPU with 24 GB GPU
RAM.

####num_procs
The number of CPU cores you have/wish to use.

####max_diameter
Fairly self-explanatory. Trims away any diameter measurements larger than this value. This is rarely needed, so 
generally just leave it much larger than the largest tree in your plot. I generally don't change it.




####slice_thickness



slice_increment=0.05,#default = 0.05
slice_clustering_distance=0.2, #default = 0.1
cleaned_measurement_radius=0.18,
minimum_CCI=0.3,
min_tree_volume=0.005,
ground_veg_cutoff_height=3,
canopy_mode='continuous',
Site='not_specified',
PlotID='not_specified',
plot_centre=None,
plot_radius=5,
intelligent_plot_cropping=1,
plot_radius_buffer=3,
UTM_zone_number=50,
UTM_zone_letter=None,
UTM_is_north=False,
filter_noise=0,
low_resolution_point_cloud_hack_mode=0) #TODO could add this mode to measure.

###Other Parameters (Changing these parameters is not recommended)



###Intelligent Plot Cropping
The purpose of this mode is to trim a point cloud to a specified plot radius, without removing points from trees that
are considered to be in the plot.

We first trim the point cloud to a radius where the initial trim radius = plot_radius + plot_radius_buffer.
Next, all main steps are run on this point cloud. Finally, we check which detected trees are within the plot,
keep the vegetation and stem points associated with those trees inside the plot, while trimming all ground vegetation
and terrain points to the specified plot_radius.



###Low resolution hack mode...
This model was trained on relatively high resolution point clouds, so if a stem is of sufficiently low resolution,
it will likely be classified as vegetation instead. Eventually, I will train this model on a larger training dataset
with more examples of sparse point clouds, however, in the meantime, I came up with an abomination of a hack that
sometimes helps a little when working with datasets of low resolution which this tool was not really designed for.

Low resolution hack mode will copy the input point cloud, jitter the points in random directions by 1 cm, then join this
copied point cloud to the original point cloud. This gives the model more points to work with, which it tends to be
happier about...
Once inference is complete, the original point cloud is returned.