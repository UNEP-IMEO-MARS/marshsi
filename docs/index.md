
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://UNEP-IMEO-MARS.github.io/marshsi/)
[![arXiv:2511.07719](https://img.shields.io/badge/arXiv-2511.07719-b31b1b.svg)](https://doi.org/10.48550/arXiv.2511.07719)
[![DOI:10.5194/amt-17-1333-2024](https://img.shields.io/badge/DOI-10.5194%2Famt--17--1333--2024-blue)](https://doi.org/10.5194/amt-17-1333-2024)
[![PyPI](https://img.shields.io/pypi/v/marshsi)](https://pypi.org/project/marshsi/)
[![PyPI - License](https://img.shields.io/pypi/l/marshsi)](https://github.com/UNEP-IMEO-MARS/marshsi/blob/main/LICENSE)

# Methane Alert and Response System Matched Filters retrievals for EMIT, PRISMA and EnMAP

This repository provides an implementation with [georeader](https://github.com/spaceml-org/georeader) of the matched filters methods of [Roger et al. 2024](https://doi.org/10.5194/amt-17-1333-2024) for EMIT, PRISMA and EnMAP. It also includes an adaptation of [mag1c](https://github.com/markusfoote/mag1c) of [Foote et al. 2020](https://doi.org/10.1109/TGRS.2020.2976888) for EMIT.


# Installation

```
pip install marshsi
```

## Examples


* EMIT example 👉 [emit_example.ipynb](./docs/emit_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marshsi/blob/main/docs/emit_example.ipynb)
* PRISMA example 👉 [prisma_example.ipynb](./docs/prisma_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marshsi/blob/main/docs/prisma_example.ipynb)
* EnMAP example 👉 [enmap_example.ipynb](./docs/enmap_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marshsi/blob/main/docs/enmap_example.ipynb)

## Citation

If you use this repo please cite:

```
@article{ruzicka_2025,
    author = {Růžička, V. and Mateo-García, G. and Irakulis-Loitxate, I. and Johnson, J. E. and Montesino San Martín, M. and Allen, A. and Guanter, L. and Thompson, D. R.},
    title = {Operational machine learning for remote spectroscopic detection of CH₄ point sources},
    journal = {arXiv},
    year = {2025},
    doi = {10.48550/arXiv.2511.07719},
    url = {https://doi.org/10.48550/arXiv.2511.07719}
}

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
