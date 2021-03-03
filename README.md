# MarioNet

https://arxiv.org/pdf/1911.08139.pdf

Чтобы начать работать с проектом, следует сначала прогнать `pip install .`. Эта команда установит необходимые пакеты, а также закинет папку `marionet` в `site-packages` энвайронмента. Если в результате работы содержимое папки меняется, то стоит снова прогнать `pip install .`.

Чтобы прогнать тесты, нужно выполнить команду `pytest .` в корне проекта.

Чтобы начать использовать pre-commit hooks (обязательно) нужно сделать `pip install pre-commit && pre-commit install`. После этого перед коммитом будут прогоняться `black` и `flake8`.


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
