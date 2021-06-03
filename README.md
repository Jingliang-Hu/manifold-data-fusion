# Manifold data fusion
![Classification maps](https://github.com/Jingliang-Hu/manifold-data-fusion/blob/main/pic/berlin_classification_map.JPG)
Caption: Classification maps of a scene in Berlin produced by 16 algorithms.

## Introduction
This repo publishes codes implemented for the following two papers.
> Jingliang Hu, Danfeng Hong, Yuanyuan Wang, Xiao Zhu (2019). A Comparative Review of Manifold Learning Techniques for Hyperspectral and Polarimetric SAR Image Fusion. Remote Sensing, 11(6), pp. 681.[paper](https://www.mdpi.com/2072-4292/11/6/681)
> 
> Jingliang Hu, Danfeng Hong, Xiao Xiang Zhu (2019). MIMA: MAPPER-Induced Manifold Alignment for Semi-Supervised Fusion of Optical Image and Polarimetric SAR Data. IEEE Transactions on Geoscience and Remote Sensing, 57(11), pp. 9025–9040.[paper](https://ieeexplore.ieee.org/abstract/document/8802291)


If you think the papers and codes are helpful, please cite them.

```
@article{hu2019comparative,
  title={A comparative review of manifold learning techniques for hyperspectral and polarimetric sar image fusion},
  author={Hu, Jingliang and Hong, Danfeng and Wang, Yuanyuan and Zhu, Xiao Xiang},
  journal={Remote Sensing},
  volume={11},
  number={6},
  pages={681},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
and,
```
@article{hu2019mima,
  title={MIMA: MAPPER-induced manifold alignment for semi-supervised fusion of optical image and polarimetric SAR data},
  author={Hu, Jingliang and Hong, Danfeng and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={57},
  number={11},
  pages={9025--9040},
  year={2019},
  publisher={IEEE}
}
```

The 16 scripts are the algorithms compared in the first paper. The algorithm 13 is the one introduced in the second paper, and 14 to 16 are variants of 13.

## Data
Please download examplary data here: 
> ftp://ftp.lrz.de/transfer/temporary_data_storage/

Place the "data" directory at the same level of "mani".

The referred data is the Berlin land cover land use data set used in these two papers, including an hyperspectral image and a Sentinel-1 dual-Pol data.

## Environment
The codes are tested with Matlab R2020b.


