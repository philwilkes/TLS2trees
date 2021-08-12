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
### If you find this tool helpful or use this tool in your research, please cite:

**The semantic segmentation tool is described here:**
\
Krisanski, S.; Taskhiri, M.S.; Gonzalez Aracil, S.; Herries, D.; Turner, P. Sensor Agnostic Semantic Segmentation of
Structurally Diverse and Complex Forest Point Clouds Using Deep Learning. Remote Sens. 2021, 13, 1413.
https://doi.org/10.3390/rs13081413

\
**The measurement tool is described here:**
\
Krisanski, S.; Taskhiri, M.S.; Gonzalez Aracil, S.; Herries, D.; Montgomery, J.; Turner, P. Forest Structural Complexity Tool - An Open
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