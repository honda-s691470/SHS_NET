# SHS_NET
Suguru Honda, Koichiro Yano, Eiichi Tanaka, Katsunori Ikari, Masayoshi Harigai

This project predicts the Sharp/van der Heijde score (SHS) for hand radiographs through three phases, orientation, joint detection, damage prediction. The results of this analysis have been posted on Medrxiv and can be found [here](https://doi.org/10.1101/2022.06.08.22276135). The overview of this study as follows.


![overview](https://user-images.githubusercontent.com/80377824/170998690-8b7bc102-bbdc-4930-a900-6bd187c53457.png)

# Recommended Requirements
This code was tested primarily on Python 3.8.12 using jupyter notebook.
The following environment is recommended.

- pytorch　>= 1.7.1
- Numpy >= 1.21.4
- Pandas >= 1.13
- Matplotlib >= 3.3.1
- Seaborn >= 0.11.0
- Sklearn >= 0.23.2
- timm >= 0.4.12
- albumentation >= 1.03 (only joint detection phase)
