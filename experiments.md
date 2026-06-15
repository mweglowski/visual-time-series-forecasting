## Experiment 1
### Model
#### Encoder
* `Conv2d(1, 64, 3, 2, 1)` + `ReLU`
* `Conv2d(64, 32, 3, 2, 1)` + `ReLU`
* `Conv2d(32, 16, 3, 2, 1)` + `ReLU`
* `Flatten`
* `Linear`
#### Decoder
* `Conv2d(16, 32, 3, 2, 1, 1)` + `ReLU`
* `Conv2d(32, 64, 3, 2, 1, 1)` + `ReLU`
* `Conv2d(64, 1, 3, 2, 1, 1)` + `ReLU`
* `Flatten`
* `Linear`