# TrackNet
Pytorch implementation based on [TrackNetv2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2).

<br>
<sup>Supported logging options:</sup>

<a href="https://wandb.ai"><img align=center src="img/wandb_logo.png" width="150" height="auto" /></a>
<a href="https://www.tensorflow.org/tensorboard"><img align=center src="img/tensorboard-logo.png" width="150" height="auto" /> </a>

## Installation
```
git clone https://github.com/mareksubocz/TrackNet
cd /TrackNet
pip install -r requirements.txt
```

## Training
```
python train.py --dataset PATH_TO_DATASET --device cuda
```

## Prediction
```
python predict.py PATH_TO_VIDEO --weights PATH_TO_TRAINED_WEIGHTS --device cuda
```

## Dataset Labelling

Keybindings:
- <kbd>l</kbd> / <kbd>→</kbd>  : next frame
- <kbd>h</kbd> / <kbd>←</kbd>  : previous frame
- <kbd>v</kbd>    : annotate well-visible ball
- <kbd>o</kbd>    : annotate occluded ball
- <kbd>m</kbd>    : annotate ball in motion (blurred)
- <kbd>f</kbd>    : fast-forward/pause video
- <kbd>n</kbd>    : go to next annotated frame
- <kbd>x</kbd>    : remove annotation
- <kbd>=</kbd> / <kbd>+</kbd>  : enlarge the annotation mark size
- <kbd>-</kbd>    : reduce the annotation mark size
- <kbd>q</kbd>    : finish annotating and save results

```
python labellingTool.py video.mp4
```

<p align="center">
  <img src="img/labelling_tool_demo.gif" alt="animated" />
</p>
<p align="center">
  <em>Labelling tool in use. Fast-forward function is distorted due to gif compression.</em>
</p>

## train.py Parameters cheatsheet
| Argument name      | Type  | Default value | Description |
|--------------------|-------|---------------|-------------|
|weights                |str    |None           |Path to initial weights the model should be loaded with. If not specified, the model will be initialized with random weights.|
|checkpoint             |str    |None           |Path to a checkpoint, chekpoint differs from weights by to including information about current loss, epoch and optimizer state.|
|batch_size             |int    |2              |Batch size of the training dataset.|
|val_batch_size         |int    |1              |Batch size of the validation dataset.|
|shuffle                |bool   |True           |Should the dataset be shuffled before training?|
|epochs                 |int    |10             |Number of epochs.|
|train_size             |float  |0.8            |Training dataset size.|
|lr                     |float  |0.01           |Learning rate.|
|momentum               |float  |0.9            |Momentum.|
|dropout                |float  |0.0            |Dropout rate. If equals to 0.0, no dropout is used.|
|dataset                |str    |'dataset/'     |Path to dataset.|
|device                 |str    |'cpu'          |Device to use (cpu, cuda, mps).|
|type                   |str    |'auto'         |Type of dataset to create (auto, image, video). If auto, the dataset type will be inferred from the dataset directory, defaulting to image.|
|save_period            |int    |10             |Save checkpoint every x epochs (disabled if <1).|
|save_weights_only      |bool   |False          |Save only weights, not the whole checkpoint|
|save_path              |str    |'weights/'     |Path to save checkpoints at.|
|no_shuffle             | -     | -             |Don't shuffle the training dataset.|
|tensorboard            | -     | -             |Use tensorboard to log training progress.')|
|one_output_frame       | -     | -             |Demand only one output frame instead of three.')|
|no_save_output_examples| -     | -             |Don't save output examples to results folder.|
|grayscale              | -     | -             |Use grayscale images instead of RGB.')|
|single_batch_overfit   | -     | -             |Overfit the model on a single batch.')|

Arguments without type or default value are used without an additional value, e.x.
``` bash
python train.py --dataset dataset/ --grayscale --one_output_frame
```
