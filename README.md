# Forest Structural Complexity Tool
## Created by Sean Krisanski
This tool was written for the purpose of processing high-resolution forest point clouds from a variety of sensor sources including Terrestrial Laser Scanning (TLS), Mobile Laser Scanning (MLS), Terrestrial Photogrammetry, Above or below-canopy UAS Photogrammetry or similar. It is built using Pytorch https://pytorch.org/ and Pytorch-Geometric https://pytorch-geometric.readthedocs.io/en/latest/#

## General Concept
The first step is semantic segmentation of the forest point cloud. This is performed using a modified version of Pointnet++ https://github.com/charlesq34/pointnet2 using the implementation in Pytorch-Geometric as a starting point provided here: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py




## Citation
###If you find this tool helpful or use this tool in your research, please cite these two papers:

**The semantic segmentation tool is described here:**
\
Krisanski, S.; Taskhiri, M.S.; Gonzalez Aracil, S.; Herries, D.; Turner, P. Sensor Agnostic Semantic Segmentation of Structurally Diverse and Complex Forest Point Clouds Using Deep Learning. Remote Sens. 2021, 13, 1413. https://doi.org/10.3390/rs13081413

\
**The measurement tool is described here:**
\
Krisanski, S.; Taskhiri, M.S.; Gonzalez Aracil, S.; Herries, D.; Turner, P. Forest Structural Complexity Tool - An Open Source, Fully-Automated Tool for Measuring Forest Point Clouds. Remote Sens. 2021, XX, XXXX. https://doi.org/XX.XXXX/rsXXXXXXXX


