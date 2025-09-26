# Methane Alert and Response System Matched Filters retrievals for EMIT, PRISMA and EnMAP

This repository provides an implementation with [georeader](https://github.com/spaceml-org/georeader) of the matched filters methods of [Roger et al. 2024](https://doi.org/10.5194/amt-17-1333-2024) for EMIT, PRISMA and EnMAP. It also includes an adaptation of [mag1c](https://github.com/markusfoote/mag1c) of [Foote et al. 2020](https://doi.org/10.1109/TGRS.2020.2976888) for EMIT.


# Installation

```
pip install .
```

## Examples

* EMIT example 👉 [emit_example.ipynb](./notebooks/emit_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/mars_mf/blob/main/notebooks/emit_example.ipynb)
* PRISMA example 👉 [prisma_example.ipynb](./notebooks/prisma_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/mars_mf/blob/main/notebooks/prisma_example.ipynb)
* EnMAP example 👉 [enmap_example.ipynb](./notebooks/enmap_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/mars_mf/blob/main/notebooks/enmap_example.ipynb)

## Citation

If you use this repo please cite:

```
@Article{roger_2024,
    AUTHOR = {Roger, J. and Guanter, L. and Gorroño, J. and Irakulis-Loitxate, I.},
    TITLE = {Exploiting the entire near-infrared spectral range to improve the detection of methane plumes with high-resolution imaging spectrometers},
    JOURNAL = {Atmospheric Measurement Techniques},
    VOLUME = {17},
    YEAR = {2024},
    NUMBER = {4},
    PAGES = {1333--1346},
    URL = {https://amt.copernicus.org/articles/17/1333/2024/},
    DOI = {10.5194/amt-17-1333-2024}
}

@article{ruzicka_starcop_2023,
	title = {Semantic segmentation of methane plumes with hyperspectral machine learning models},
	volume = {13},
	issn = {2045-2322},
	url = {https://www.nature.com/articles/s41598-023-44918-6},
	doi = {10.1038/s41598-023-44918-6},
	number = {1},
	journal = {Scientific Reports},
	author = {Růžička, Vít and Mateo-Garcia, Gonzalo and Gómez-Chova, Luis and Vaughan, Anna, and Guanter, Luis and Markham, Andrew},
	month = nov,
	year = {2023},
	pages = {19999}
}

@ARTICLE{foote_2020,
    author={M. D. {Foote} and P. E. {Dennison} and A. K. {Thorpe} and D. R. {Thompson} and S. {Jongaramrungruang} and C. {Frankenberg} and S. C. {Joshi}},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    title={Fast and Accurate Retrieval of Methane Concentration From Imaging Spectrometer Data Using Sparsity Prior},
    year={2020},
    volume={},
    number={},
    pages={1-13},
    keywords={Airborne Visible InfraRed Imaging Spectrometer-Next Generation (AVIRIS-NG);greenhouse gas emissions;methane mapping;plume detection.},
    doi={10.1109/TGRS.2020.2976888},
    ISSN={1558-0644},
    month={},
}
```
