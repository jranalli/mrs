# Models for Remote Sensing
![Model comparison](./demo/results_cmp.png)

## Installation
### Dependencies
- PyTorch >= 0.4
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [TorchSummary](https://github.com/sksq96/pytorch-summary)
- [Albumentation](https://github.com/albu/albumentations)

## Pretrained Models
| Encoder Name | Decoder Name | Dataset | Label    | Score (IoU) | Size    | Model |
|:------------:|:------------:|:-------:|:--------:|:-----------:|:--------|:-----:|
| VGG16        | UNet         | Inria   | Building | 78.56       | 207.3MB | [Box](https://duke.box.com/s/0y8bcxmsrqe5e3xjlaakytmvrwt7m5f7) |
| VGG19        | UNet         | Inria   | Building | 78.17       | 247.8MB | [Box](https://duke.box.com/s/ph3amubdy5vtl2mrzjrdc98qioks7z3y) |
| ResNet34     | UNet         | Inria   | Building | 77.06       | 204.2MB | [Box](https://duke.box.com/s/bceeabdfg31cl9uadiir8fdyrfk0aa2l) |
| ResNet50     | UNet         | Inria   | Building | 78.78       | 666.9MB | [Box](https://duke.box.com/s/nhvkbb6nqezjz40g19j9s2zfhjku8jjz) |
| ResNet101    | UNet         | Inria   | Building | 79.09       | 812.1MB | [Box](https://duke.box.com/s/d88bnmnkbmlhgpqfxws0w12xypijyk7t) |
| VGG16        | PSPNet       | Inria   | Building | 76.23       | 171.1MB | [Box](https://duke.box.com/s/4rhkj8ce4f90t967wh371bh1r66hos7k) |
| VGG19        | PSPNet       | Inria   | Building | 75.94       | 211.6MB | [Box](https://duke.box.com/s/fqevw4n6t8orszwh94smxiwwvp5jgfdd) |
| ResNet34     | PSPNet       | Inria   | Building | 76.11       | 221.2MB | [Box](https://duke.box.com/s/eu49tfvllgefxf8ergh1b8mv7y4vjifz) |
| ResNet50     | PSPNet       | Inria   | Building | 77.46       | 418.3MB | [Box](https://duke.box.com/s/kxm9r269csgxfosrui5jnqqir54ttd59) |
| ResNet101    | PSPNet       | Inria   | Building | 78.55       | 563.5MB | [Box](https://duke.box.com/s/zx2yfyrekvi0dk84l0qpo6xy5le1qsex) |
| ResNet50     | DLinkNet     | Inria   | Building | 77.08       | 1.4GB   | [Box](https://duke.box.com/s/1tn7zcuvfknkxfdb9aa0lye0pyn8n056) |

## TODOs
- [ ] Encoder Structures:
    - [X] [VGG](./network/backbones/vggnet.py)
    - [X] [ResNet](./network/backbones/resnet.py)
    - [ ] DenseNet
    - [ ] SqueezeNet
    - [ ] InceptionNet
- [ ] Decoder Structures:
    - [X] [UNet](./network/unet.py)
    - [X] [DLinkNet](./network/dlinknet.py)
    - [X] [PSPNet](./network/pspnet.py)
    - [ ] DeepLabV3
- [ ] Different Losses:
    - [X] Xent
    - [X] Jaccard Approximation
    - [ ] Focal Loss
    - [ ] Lovasz softmax (https://github.com/bermanmaxim/LovaszSoftmax/tree/master/pytorch)
    - [ ] Weighted combination of arbitrary supported losses
- [X] Multi GPU Training
- [ ] Evaluation
    - [X] Dataset Evaluator
    - [X] Evaluate Report & Prediction Map
- [X] Toy Dataset
- [X] Config as json file
- [X] Check flex loading function
- [ ] Results visualization
- [ ] Class weights on criterions
## Known Bugs
- [ ] Unable to do model-wise data parallel