# MarioNet

https://arxiv.org/pdf/1911.08139.pdf

Чтобы прогнать тесты, нужно выполнить команду `python -m pytest` в корне проекта (не `pytest`, потому что у него проблемы с относительными импортами).

Реализовать следующие элементы пайплайна:

* VoxCeleb1, CelebV. Download.
* 3D Landmark Extractor. Download opensource version.
* 3D -> 2D Landmark converter
* Landmark Transformer
* Driver Encoder
* Target Encoder
* Blender -> Attention Block
* Positional encoding
* Decoder
* Discriminator
* Connect all parts of model together
* Loss GAN
* Perceptual losses -> Lp Lpf
* Feature matching loss Lfm
* Overal Generator loss
* Overall Discriminator loss
* Training Pipeline
* Inference Pipeline
