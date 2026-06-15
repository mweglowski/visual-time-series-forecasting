
## Experiment 1

> Initial VisualAE architecture

### Params
* **model_class**: `VisualAE`
* **batch**: 32
* **latent**: 128
* **epochs_target**: 50
* **epochs_completed**: 17
* **device**: cuda

### Model Architecture
#### `EncoderPlainConv`:
* `Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `Conv2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=1600, out_features=128, bias=True)`

#### `DecoderPlainConv`:
* `Linear(in_features=128, out_features=1600, bias=True)`
* `ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ConvTranspose2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 14.37s
* **Best Val JSD Loss**: 0.657835
* **Final Train JSD Loss**: nan
* **Final Val MASE**: 6.329186
* **Final Val 1D IoU**: 0.0000

### Training Charts
![train_history_1](../visualizations/training/train_history_1.jpg)

---

## Experiment 2

> Increase latent: 128 -> 512

### Params
* **model_class**: `VisualAE`
* **batch**: 32
* **latent**: 512
* **epochs_target**: 50
* **epochs_completed**: 16
* **device**: cuda

### Model Architecture
#### `EncoderPlainConv`:
* `Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `Conv2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=1600, out_features=512, bias=True)`

#### `DecoderPlainConv`:
* `Linear(in_features=512, out_features=1600, bias=True)`
* `ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ConvTranspose2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 13.69s
* **Best Val JSD Loss**: 0.658248
* **Final Train JSD Loss**: nan
* **Final Val MASE**: 6.329186
* **Final Val 1D IoU**: 0.0000

### Training Charts
![train_history_2](../visualizations/training/train_history_2.jpg)

---

## Experiment 3

> Added batch normalization on initial architecture

### Params
* **model_class**: `VisualAE`
* **batch**: 32
* **latent**: 128
* **epochs_target**: 50
* **epochs_completed**: 22
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=1600, out_features=128, bias=True)`

#### `Decoder`:
* `Linear(in_features=128, out_features=1600, bias=True)`
* `ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 21.59s
* **Best Val JSD Loss**: 0.585579
* **Final Train JSD Loss**: 0.394044
* **Final Val MASE**: 2.964341
* **Final Val 1D IoU**: 0.0317

### Training Charts
![train_history_3](../visualizations/training/train_history_3.jpg)

---

## Experiment 4

> Increase number of filters

### Params
* **model_class**: `VisualAE`
* **batch**: 32
* **latent**: 128
* **epochs_target**: 50
* **epochs_completed**: 32
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=1600, out_features=128, bias=True)`

#### `Decoder`:
* `Linear(in_features=128, out_features=1600, bias=True)`
* `ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(32, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 36.69s
* **Best Val JSD Loss**: 0.588057
* **Final Train JSD Loss**: 0.387571
* **Final Val MASE**: 2.978562
* **Final Val 1D IoU**: 0.0307

### Training Charts
![train_history_4](../visualizations/training/train_history_4.jpg)

---

## Experiment 5

> Increase latent dim and target number of epochs, just for the future

### Params
* **model_class**: `VisualAE`
* **batch**: 32
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 56
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=1600, out_features=512, bias=True)`

#### `Decoder`:
* `Linear(in_features=512, out_features=1600, bias=True)`
* `ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(32, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 1m 0.30s
* **Best Val JSD Loss**: 0.500561
* **Final Train JSD Loss**: nan
* **Final Val MASE**: 6.329186
* **Final Val 1D IoU**: 0.0000

### Training Charts
![train_history_5](../visualizations/training/train_history_5.jpg)

---

## Experiment 6

> Increase batch size

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 31
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=1600, out_features=512, bias=True)`

#### `Decoder`:
* `Linear(in_features=512, out_features=1600, bias=True)`
* `ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(32, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 28.16s
* **Best Val JSD Loss**: 0.571638
* **Final Train JSD Loss**: 0.336512
* **Final Val MASE**: 2.861675
* **Final Val 1D IoU**: 0.0543

### Training Charts
![train_history_6](../visualizations/training/train_history_6.jpg)

---

## Experiment 7

> Increase number of channels

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 30
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(256, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=1600, out_features=512, bias=True)`

#### `Decoder`:
* `Linear(in_features=512, out_features=1600, bias=True)`
* `ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(32, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(256, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 49.92s
* **Best Val JSD Loss**: 0.577563
* **Final Train JSD Loss**: 0.362917
* **Final Val MASE**: 2.971868
* **Final Val 1D IoU**: 0.0547

### Training Charts
![train_history_7](../visualizations/training/train_history_7.jpg)

---

## Experiment 8

> Changing number of channels

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 34
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=1600, out_features=512, bias=True)`

#### `Decoder`:
* `Linear(in_features=512, out_features=1600, bias=True)`
* `ConvTranspose2d(16, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 44.69s
* **Best Val JSD Loss**: 0.570212
* **Final Train JSD Loss**: 0.321830
* **Final Val MASE**: 2.848406
* **Final Val 1D IoU**: 0.0547

### Training Charts
![train_history_8](../visualizations/training/train_history_8.jpg)

---

## Experiment 9

> Increasing number of channels again

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 46
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=6400, out_features=512, bias=True)`

#### `Decoder`:
* `Linear(in_features=512, out_features=6400, bias=True)`
* `ConvTranspose2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 1m 1.83s
* **Best Val JSD Loss**: 0.582398
* **Final Train JSD Loss**: 0.253745
* **Final Val MASE**: 3.069262
* **Final Val 1D IoU**: 0.0339

### Training Charts
![train_history_9](../visualizations/training/train_history_9.jpg)

---

## Experiment 10

> Changing channels to be like /\, from lower to higher and so on, like hiking a mountain and going back

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 40
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=25600, out_features=512, bias=True)`

#### `Decoder`:
* `Linear(in_features=512, out_features=25600, bias=True)`
* `ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 57.54s
* **Best Val JSD Loss**: 0.587154
* **Final Train JSD Loss**: 0.345221
* **Final Val MASE**: 2.810701
* **Final Val 1D IoU**: 0.0418

### Training Charts
![train_history_10](../visualizations/training/train_history_10.jpg)

---

## Experiment 11

> Trying architecture channel values from paper (not kernel sizes and padding yet)

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 42
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=51200, out_features=512, bias=True)`

#### `Decoder`:
* `Linear(in_features=512, out_features=51200, bias=True)`
* `ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 2m 24.54s
* **Best Val JSD Loss**: 0.594411
* **Final Train JSD Loss**: 0.359475
* **Final Val MASE**: 2.956074
* **Final Val 1D IoU**: 0.0404

### Training Charts
![train_history_11](../visualizations/training/train_history_11.jpg)

---

## Experiment 12

> Going back to architecture with smaller number of parameters

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 40
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=25600, out_features=512, bias=True)`

#### `Decoder`:
* `Linear(in_features=512, out_features=25600, bias=True)`
* `ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 52.81s
* **Best Val JSD Loss**: 0.592283
* **Final Train JSD Loss**: 0.331036
* **Final Val MASE**: 2.885121
* **Final Val 1D IoU**: 0.0398

### Training Charts
![train_history_12](../visualizations/training/train_history_12.jpg)

---

## Experiment 13

> Decrease number of parameters

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 42
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=12800, out_features=512, bias=True)`

#### `Decoder`:
* `Linear(in_features=512, out_features=12800, bias=True)`
* `ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 27.65s
* **Best Val JSD Loss**: 0.581988
* **Final Train JSD Loss**: 0.279005
* **Final Val MASE**: 2.817555
* **Final Val 1D IoU**: 0.0508

### Training Charts
![train_history_13](../visualizations/training/train_history_13.jpg)

---

## Experiment 14

> Change parameters: kernel=5, padding=2

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 38
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=12800, out_features=512, bias=True)`

#### `Decoder`:
* `Linear(in_features=512, out_features=12800, bias=True)`
* `ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics (Best/Final Convergence)
* **Execution Time**: 42.04s
* **Best Val JSD Loss**: 0.598772
* **Final Train JSD Loss**: 0.419757
* **Final Val MASE**: 2.811742
* **Final Val 1D IoU**: 0.0374

### Training Charts
![train_history_14](../visualizations/training/train_history_14.jpg)

---

## Experiment 15

> Change parameters: kernel=7, padding=3

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 37
* **device**: cuda

### Model Architecture
#### `Encoder`:
* `Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=12800, out_features=512, bias=True)`

#### `Decoder`:
* `Linear(in_features=512, out_features=12800, bias=True)`
* `ConvTranspose2d(128, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1))`
* `BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(32, 1, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics
| Metric | Best | Final |
| :--- | :--- | :--- |
| **Train JSD** | 0.4717 | 0.4717 |
| **Val JSD** | 0.6068 | 0.6083 |
| **Val MASE** | 2.8155 | 2.8288 |
| **Val IoU** | 0.0460 | 0.0280 |

**Execution time**: 51.97s

### Training Charts
![train_history_15](../visualizations/training/train_history_15.jpg)

---

## Experiment 16

> VisualAE from paper

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 50
* **device**: cuda

### Model Architecture
#### `EncoderPaper`:
* `Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=25600, out_features=512, bias=True)`

#### `DecoderPaper`:
* `Linear(in_features=512, out_features=25600, bias=True)`
* `ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(64, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics
| Metric | Best | Final |
| :--- | :--- | :--- |
| **Train JSD** | 0.3926 | 0.3926 |
| **Val JSD** | 0.5993 | 0.6006 |
| **Val MASE** | 2.8135 | 2.9745 |
| **Val IoU** | 0.0495 | 0.0339 |

**Execution time**: 2m 10.17s

### Training Charts
![train_history_16](../visualizations/training/train_history_16.jpg)

---

## Experiment 17

> Checking VisualAE on a new dataset -> GSPC

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 100
* **epochs_completed**: 31
* **device**: cuda

### Model Architecture
#### `EncoderPaper`:
* `Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=25600, out_features=512, bias=True)`

#### `DecoderPaper`:
* `Linear(in_features=512, out_features=25600, bias=True)`
* `ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(64, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics
| Metric | Best | Final |
| :--- | :--- | :--- |
| **Train JSD** | 0.3250 | nan |
| **Val JSD** | 0.3257 | nan |
| **Val MASE** | 2.4708 | 5.1518 |
| **Val IoU** | 0.0608 | 0.0000 |

**Execution time**: 8m 37.03s

### Training Charts
![train_history_17](../visualizations/training/train_history_17.jpg)

---

## Experiment 18

> Another training but on number of epochs before previous crash

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 15
* **epochs_completed**: 15
* **device**: cuda

### Model Architecture
#### `EncoderPaper`:
* `Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=25600, out_features=512, bias=True)`

#### `DecoderPaper`:
* `Linear(in_features=512, out_features=25600, bias=True)`
* `ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(64, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics
| Metric | Best | Final |
| :--- | :--- | :--- |
| **Train JSD** | 0.3320 | 0.3320 |
| **Val JSD** | 0.3170 | 0.3170 |
| **Val MASE** | 2.4217 | 8.7097 |
| **Val IoU** | 0.0662 | 0.0199 |

**Execution time**: 4m 15.86s

### Training Charts
![train_history_18](../visualizations/training/train_history_18.jpg)

---

## Experiment 19

> Shorter training to prevent from crash

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 10
* **epochs_completed**: 10
* **device**: cuda

### Model Architecture
#### `EncoderPaper`:
* `Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=25600, out_features=512, bias=True)`

#### `DecoderPaper`:
* `Linear(in_features=512, out_features=25600, bias=True)`
* `ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(64, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics
| Metric | Best | Final |
| :--- | :--- | :--- |
| **Train JSD** | 0.3600 | 0.3600 |
| **Val JSD** | 0.3465 | 0.3465 |
| **Val MASE** | 2.5377 | 9.3767 |
| **Val IoU** | 0.0640 | 0.0000 |

**Execution time**: 2m 48.55s

### Training Charts
![train_history_19](../visualizations/training/train_history_19.jpg)

---

## Experiment 20

> Even shorter training to prevent from crash

### Params
* **model_class**: `VisualAE`
* **batch**: 128
* **latent**: 512
* **epochs_target**: 8
* **epochs_completed**: 8
* **device**: cuda

### Model Architecture
#### `EncoderPaper`:
* `Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))`
* `BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ReLU()`
* `Flatten(start_dim=1, end_dim=-1)`
* `Linear(in_features=25600, out_features=512, bias=True)`

#### `DecoderPaper`:
* `Linear(in_features=512, out_features=25600, bias=True)`
* `ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
* `ConvTranspose2d(64, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))`
* `ReLU()`
* `Sigmoid()`

### Metrics
| Metric | Best | Final |
| :--- | :--- | :--- |
| **Train JSD** | 0.5231 | 0.5231 |
| **Val JSD** | 0.5398 | 0.5398 |
| **Val MASE** | 2.5600 | 2.6151 |
| **Val IoU** | 0.0597 | 0.0210 |

**Execution time**: 2m 16.79s

### Training Charts
![train_history_20](../visualizations/training/train_history_20.jpg)

---
