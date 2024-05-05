<div align="center">

<h1>PLA & RegionPLC</h1>
<p>This repo contains the official implementation of <a href="https://dingry.github.io/projects/PLA">PLA (CVPR2023)</a> and <a href="https://jihanyang.github.io/projects/RegionPLC">RegionPLC (CVPR 2024)</a></p>

<hr style="color: #333; height: 2px; width: 85%">

<h4>PLA: Language-Driven Open-Vocabulary 3D Scene Understanding</h4>

<div>
    <a href="https://dingry.github.io/" target="_blank">Runyu Ding</a><sup>*</sup>,</span>
    <a href="https://jihanyang.github.io/" target="_blank">Jihan Yang</a><sup>*</sup>,</span>
    <a href="https://scholar.google.com/citations?user=KJU5YRYAAAAJ&hl=en" target="_blank">Chuhui Xue</a><sup></sup>,</span>
    <a href="https://github.com/HannibalAPE" target="_blank">Wenqing Zhang</a><sup></sup>,</span>
    <a href="https://songbai.site/" target="_blank">Song Bai</a><sup>&#8224</sup>,</span>
    <a href="https://xjqi.github.io/" target="_blank">Xiaojuan Qi</a><sup>&#8224</sup>,</span>  
</div>

<p><em>CVPR 2023</em></p>

[project page](https://dingry.github.io/projects/PLA) | [arXiv](https://arxiv.org/abs/2211.16312)

<hr style="color: #333; height: 2px; width: 85%">

<h4>RegionPLC: Regional Point-Language Contrastive Learning for Open-World 3D Scene Understanding</h4>

<div>
    <a href="https://jihanyang.github.io/" target="_blank">Jihan Yang</a><sup>*</sup>,</span>
    <a href="https://dingry.github.io/" target="_blank">Runyu Ding</a><sup>*</sup>,</span>
    <a href="https://github.com/VincentDENGP" target="_blank">Weipeng Deng</a>,</span>
    <a href="https://wang-zhe.me/" target="_blank">Zhe Wang</a>,</span>
    <a href="https://xjqi.github.io/" target="_blank">Xiaojuan Qi</a>,</span>  
</div>
<p><em>CVPR 2024</em></p>

<p><a href="https://jihanyang.github.io/projects/RegionPLC">project page</a> | <a href="https://arxiv.org/pdf/2304.00962">arXiv</a></p>

</div>

##### Highlights:
- Official PLA implementation is contained in the `main` branch
- Official RegionPLC implementation is contained in the `regionplc` branch

### Release
- [2024-05-05] Releasing **RegionPLC** implementation. Please checkout `regionplc` branch to try it!

### Getting Started

#### Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for the installation.

#### Dataset Preparation
Please refer to [DATASET.md](docs/DATASET.md) for dataset preparation.

#### Training & Inference

Please refer to [MODEL.md](docs/MODEL.md) for training and inference scripts and pretrained models.


### Citation
If you find this project useful in your research, please consider cite:
```bibtex
@inproceedings{ding2022language,
    title={PLA: Language-Driven Open-Vocabulary 3D Scene Understanding},
    author={Ding, Runyu and Yang, Jihan and Xue, Chuhui and Zhang, Wenqing and Bai, Song and Qi, Xiaojuan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023}
}
```

```bibtex
@inproceedings{yang2024regionplc,
    title={RegionPLC: Regional point-language contrastive learning for open-world 3d scene understanding},
    author={Yang, Jihan and Ding, Runyu and Deng, Weipeng and Wang, Zhe and Qi, Xiaojuan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```

### Acknowledgement
Code is partly borrowed from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [PointGroup](https://github.com/dvlab-research/PointGroup) and [SoftGroup](https://github.com/thangvubk/SoftGroup).