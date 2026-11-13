# Akida Pico — Model Training & Fine-Tuning Guide

BrainChip Holdings Ltd. · November 2026

## Purpose and scope

This guide shows how to get trained weights for an Akida Pico model: train a recurrent TENN from scratch, or fine-tune a zoo model on your own data. It stops where the *Model Development Guide* starts, at the stateful conversion, quantization and mapping steps, which it only summarizes.

**Choose the shortest path that meets your accuracy target**

| Situation | Do this |
| --- | --- |
| Your task is the zoo task (12-class keyword spotting, 4-label bearing faults) and the pretrained accuracy is enough | Use the pretrained model and go straight to quantization |
| Same task, your own data, similar signal type and sample rate | Fine-tune from the zoo weights |
| Same signal type, different classes or keywords | Fine-tune with a new classification head |
| Different signal, sample rate or output type | Train from scratch with the same architecture family |

**What this guide assumes**

- You have a labelled dataset of raw 1-D samples that can be stored as int16.
- You have a CUDA GPU for training. Pico itself is not used for training.
- You can run the Pico steps afterwards on the Akida cloud FPGA, or on the software model of Pico.

**How the fine-tuning recipe was built.** The public training scripts expose `train`, `eval`, `convert` and `extract`. They do not include a dedicated tuning mode, so the fine-tuning recipe here is a workflow built on those scripts and on general practice. BrainChip has not published validated fine-tuning results for Pico models, so treat its learning rates and epoch counts as starting points to tune on your data.

## How training works for Pico

You train the kernelized model in floating point on a GPU, then convert, quantize and map it afterwards. The stateful model that runs on Pico cannot be trained: the training scripts raise *Training of stateful model is not supported*.

1. **Create** the kernelized model with a fixed architecture (`akida_models create`).
2. **Train** in float on whole windows, with the loss computed on logits.
3. **Evaluate** the float model, then repeat the evaluation on the stateful form and the Akida form (Evaluation and reproducibility).
4. **Convert** to stateful, **extract** calibration samples, **quantize**, and **map** (After training).

**What stays fixed**

- **Input.** int16 samples, divided by 2^15 by a `Rescaling` layer inside the model, so the network sees values in [−1, 1).
- **Output.** Raw logits. The head has no batch norm and no activation; softmax or sigmoid is applied in the loss and at inference.
- **Quantization is post-training.** The scripts do not do quantization-aware training. The float model is quantized once, with calibration samples, and the measured cost is about 0.3 to 1 point.
- **Block structure.** Kernelized layer, subsampling, batch norm, ReLU, repeated, as the Model Development Guide describes.

**Window length.** Train on a window whose length is a power of 2 (16,384 for SC12, 8,192 for UORED): the SC12 preprocessing pads or crops to a power of 2 because the kernelized layer convolves with an FFT. The window also sets the initial time constants of each layer's state-space kernel, so changing it changes what the model starts out able to remember. For signals with slow structure, such as anomaly detection, longer windows exploit the state-space memory.

**Batches are whole windows.** At train time the data pipeline feeds `(batch, window, 1)`. Only the stateful evaluation path splits windows into 256-sample chunks.

## Environment and prerequisites

Install the MetaTF packages in a clean environment and confirm that Python imports the version you expect. A stale `akida_models` earlier on the path is a common cause of unexpected behavior.

```bash
pip install "akida~=2.19.2" "cnn2snn~=2.19.2" "akida_models~=1.14.1" quantizeml \
            tensorflow-datasets h5py scikit-learn tqdm
conda install -c conda-forge ffmpeg        # SC12 only: TFDS uses it to prepare audio
python -c "import akida_models; print(akida_models.__file__)"   # check which copy is loaded
```

The versions follow the `requirements.txt` in the `bc_cloud_pico` repository. The training scripts use `tf_keras` and TensorFlow; train on a CUDA GPU and select it with `CUDA_VISIBLE_DEVICES`.

**Command-line tools.** `pip` installs three entry points.

| Command | Purpose |
| --- | --- |
| `akida_models create -s <file>.h5 <model>` | Build a fresh kernelized model, `tenn_recurrent_sc12` or `tenn_recurrent_uored`, and save it |
| `tenn_sc12_train <action> ...` | Speech Commands keyword spotting |
| `tenn_uored_train <action> ...` | UORED bearing fault classification |

Both training tools take the same actions and common options.

| Action | What it does |
| --- | --- |
| `train` | Trains the model in `-m` for `-e` epochs; `-s` names the saved file |
| `eval` | Scores a model; add `-ak` to evaluate the Akida-converted model |
| `convert` | Converts a kernelized model to stateful; `-ts` sets the chunk size, default 256 |
| `extract` | Saves int16 calibration samples to the file named by `-s` (default `samples.npz`); `-ns` sets how many |

Common options: `-m` model file, `-b` batch size (default 128 for SC12, 32 for UORED), `-d` data path. UORED adds `--split` to choose one of the stored bearing-wise splits (0 to 104, default 0). There is no separate `tune` action.

**Data and time**

- SC12 downloads Speech Commands through TensorFlow Datasets (about 2 GB on first use, cached afterwards).
- UORED needs `uored_full.h5` (about 96 MB) and the split JSON files in the directory you pass with `-d`. The file is not bundled with the examples.
- The SC12 reference recipe uses 200 epochs at batch size 128. Training time depends on your GPU; measure it on a short run first.

## Preparing your data

The model wants fixed-length int16 windows of shape `(N, window, channels)`, a label per window, and a split that keeps every recording source on one side only. Most accuracy lost in practice is lost here, not in the model.

**What the two reference pipelines do**

|  | SC12 keyword spotting | UORED bearing faults |
| --- | --- | --- |
| Source | Speech Commands v0.02, TFDS `train` and `test` | `uored_full.h5`, 420,000 samples per recording at 42 kHz |
| Channel used | Mono audio | Accelerometer channel 0 only |
| Window | 16,384 samples; shorter clips zero-padded, longer cropped | Non-overlapping windows of the model length (8,192) |
| Scaling | Waveform kept in int16 range | Each recording divided by its own peak, then multiplied by 2^15 − 1 |
| Labels | Integer 0 to 11 | Four 0/1 labels per recording, repeated for every window |
| Split | Dataset's own train and test sets | Bearing-wise: `split_000.json` lists train and test bearing IDs; 20 bearings, 8 held out (2 per fault type) |
| Balancing | Equal samples per class per epoch (3,100) | None |
| Augmentation | None in the reference recipe | Random gain, normal with mean 1.0 and standard deviation 0.7, clipped to int16 |

**Preparing your own signal**

```python
import numpy as np

GLOBAL_PEAK = 4.0          # one number for the whole dataset, e.g. the sensor's full-scale value
x = np.clip(signal / GLOBAL_PEAK, -1.0, 1.0)
X = (x * (2**15 - 1)).astype(np.int16)[..., np.newaxis]   # (N, window, 1)
```

**Rules that matter**

1. **Use one global scale, not a per-recording one.** Per-recording peak scaling is what the UORED script does, but a deployed sensor never knows a recording's peak. Pick one scale and offset from the sensor's range and keep it identical in training, calibration and deployment.
2. **Split by source.** Hold out whole bearings, speakers, machines or sessions, never random windows. Windows cut from one recording are near-duplicates; a random split inflates the score.
3. **Keep a validation set that is not the test set.** The reference scripts pass the test set as validation data, and the SC12 script uses it to pick the best epoch. For your own data use three sets: train, validation for epoch selection, and a test set you touch once.
4. **Balance the classes.** The SC12 sampler draws 3,100 examples per class each epoch, downsampling the large `unknown` class and upsampling `silence` with replacement. Removing it cost about 2 points in one run, which is about the size of run-to-run noise, so confirm the effect on your data. Use `build_balanced_indices` or an equivalent.
5. **Augment cautiously.** Gain augmentation is cheap and fits sensors with variable amplitude. The SC12 reference recipe uses none because run-to-run noise hid any gain, though augmentation may help robustness on a different test set. Test it on several seeds.
6. **Match the sample rate.** The chunk size is fixed in samples, so the decision rate follows your sample rate. Resample to the rate you will deploy.

**Calibration samples.** Quantization needs a `.npz` file with a `data` array of int16 segments drawn from the training distribution: 100 segments of 1,024 samples for SC12 (cut from the middle of the word) and 256 segments of 8,192 samples for UORED. Create it with the `extract` action so it passes through the same preprocessing as training, and make sure it covers every class and operating condition you expect.

## Training from scratch

Create the kernelized model, train it with `AdamW` and a cosine schedule with warm-up, and keep the best weights. The commands below reproduce the zoo recipes; the Python version shows how to adapt them to your own data.

**Command line**

```bash
# Keyword spotting, 12 classes
akida_models create -s tenn_recurrent_sc12.h5 tenn_recurrent_sc12
CUDA_VISIBLE_DEVICES=0 tenn_sc12_train train -d /data/sc12 \
    -m tenn_recurrent_sc12.h5 -s tenn_recurrent_sc12_trained.h5 -e 200 -b 128

# Bearing fault classification, 4 labels
akida_models create -s tenn_recurrent_uored.h5 tenn_recurrent_uored
tenn_uored_train train -d /data/uored --split 0 \
    -m tenn_recurrent_uored.h5 -s tenn_recurrent_uored_trained.h5 -e <epochs> -b 32
```

The SC12 epoch count of 200 is the reference recipe. Choose the UORED epoch count from the validation curve.

**Recipes**

| Setting | SC12 | UORED |
| --- | --- | --- |
| Optimizer | AdamW, weight decay 0.05 | AdamW, weight decay 0.05 |
| Peak learning rate | 0.01 | 0.005 |
| Schedule | Cosine decay, linear warm-up over 2.5% of steps | Cosine decay, linear warm-up over 10% of steps |
| Loss | Sparse categorical cross-entropy on logits | Binary cross-entropy on logits |
| Metrics | Accuracy, macro F1, weighted F1 | Binary accuracy (threshold 0 on logits), multi-label AUC |
| Batch size | 128 | 32 |
| Regularization | Spatial dropout 0.1 after every block | Spatial dropout 0.1 after every block |
| Model selection | Best epoch by validation accuracy | Final epoch |
| Seeding | Not fixed | Seed 42 for NumPy and TensorFlow; model creation also seeded |

**Python, for your own data and classes**

```python
import akida_models, tf_keras as keras
from akida_models.param_scheduler import CosineDecayWithLinearWarmup
from akida_models.training import RestoreBest

model = akida_models.tenn_recurrent_sc12(input_shape=(16384, 1), num_classes=8)
model.save("my_model.h5", include_optimizer=False)          # keep the untrained model too

total_steps = steps_per_epoch * epochs
scheduler = CosineDecayWithLinearWarmup(0.01, 0.025, total_steps)
model.compile(
    optimizer=keras.optimizers.experimental.AdamW(learning_rate=scheduler, weight_decay=0.05),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()])

model.fit(train_ds, validation_data=val_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
          callbacks=[RestoreBest(model, monitor="val_sparse_categorical_accuracy")])
model.save("my_model_trained.h5", include_optimizer=False)
```

`train_ds` yields `(int16 windows, labels)` as prepared in the previous section. For a multi-label task use `tenn_recurrent_uored(num_classes=K)`, `BinaryCrossentropy(from_logits=True)` and an AUC metric.

**Rules**

- Keep the loss on logits. The model outputs raw logits and the scripts rely on that.
- Do not train the stateful form; it is rejected.
- Change `input_shape` only to a power-of-2 window, and keep total subsampling at the chunk size you intend to stream with.
- Run `convert_to_stateful`, quantization and `map()` on the untrained model once, before training, to catch a model that will not map.

## Fine-tuning on your data

Fine-tune by building the same architecture, loading the zoo weights into it and training briefly at a lower learning rate. Keeping the architecture unchanged is the main advantage: the model still maps to Pico in one sequence, and chunk size, latency and power stay as measured.

> **Starting points, not validated results.** Loading kernelized weights into the same architecture and running the `train` action are documented mechanics (the notebooks do the former). The learning rates, epoch counts and freezing schedule below are suggested starting points based on general practice. Replace them with values you measure on your own data.

**Pick a case**

| Case | Model | Layers trained | Starting learning rate | Epochs |
| --- | --- | --- | --- | --- |
| A. Same task and classes, new data | Zoo architecture unchanged | All | About 10× below the from-scratch peak: 1e-3 for SC12-type, 5e-4 for UORED-type | 10 to 30% of the from-scratch count |
| B. Same signal, different classes | Same blocks, new `num_classes` | Head only, then all | Head 1e-3; then 1e-4 for all layers | Head 5 to 10; all 10 to 30 |
| C. Different sample rate | Resample to the zoo rate if the information survives; otherwise train from scratch | As A | As A | As A |
| D. Different signal type or output | Train from scratch | All | Recipe in the previous section | Full |

**Case B in code**

```python
import akida_models, tf_keras as keras

pre = akida_models.tenn_recurrent_sc12()                     # 12 classes
pre.load_weights("tenn_recurrent_sc12.h5")                   # zoo kernelized weights

model = akida_models.tenn_recurrent_sc12(num_classes=8)      # your classes
for layer in model.layers:                                   # copy every block except the head
    if layer.name != "ssm_layer_head":
        layer.set_weights(pre.get_layer(layer.name).get_weights())

# Phase 1: train the new head only
for layer in model.layers:
    layer.trainable = layer.name == "ssm_layer_head"
model.compile(...)                                           # compile again after changing trainable
model.fit(train_ds, validation_data=val_ds, epochs=8, steps_per_epoch=steps)

# Phase 2: all layers, lower learning rate
for layer in model.layers:
    layer.trainable = True
model.compile(...)
model.fit(train_ds, validation_data=val_ds, epochs=20, steps_per_epoch=steps)

model.save("my_model_finetuned.h5", include_optimizer=False)
```

The head's output matrix has a new shape, so it keeps its fresh initialization. Case A is the same code with the same `num_classes`, the head copied too, and phase 2 only. The saved full model is what `tenn_sc12_train` or `tenn_uored_train` takes with `-m`.

**Practical points**

- **Data volume.** With few labelled examples, stay in phase 1 longer and watch the gap between training and validation scores. Fine-tuned small models overfit quickly.
- **Batch norm.** Frozen blocks keep batch norm in inference mode; after phase 2 begins it updates again, so use a batch size large enough for stable statistics.
- **Baselines.** Evaluate the untouched zoo model on your test set first (a zero-shot baseline), and compare the fine-tuned model with a from-scratch run on the same split. If fine-tuning does not beat both, train from scratch.
- **Recalibrate.** The zoo calibration pack does not represent your data. Extract a new one from your training set before quantizing.
- **Off-device only.** Pico does not train. After any retraining the whole model is quantized again and reloaded onto the device.
- **Your data decides the result.** The accuracy figures published for the zoo models do not transfer to other data.

## Evaluation and reproducibility

Two training runs with identical settings can differ by more than many design changes do, so a single run cannot show that one design beats another. Evaluate at every stage, and compare designs on several seeds.

**How much runs vary.** These are example training runs for the 12-class SC12 model, scored per 1-second clip.

| Run | Setting | Float | Stateful float | Quantized stateful |
| --- | --- | --- | --- | --- |
| 1 | Augmentation, class balancing | 94.98% | 95.01% | 94.01% |
| 2 | Same as run 1 | 96.22% | 96.24% | 95.93% |
| 3 | No augmentation, balancing | 95.21% | 95.24% | 93.74% |
| 4 | No balancing (augmentation setting not stated) | 93.32% | 93.29% | 91.48% |
| 5 | No augmentation, balancing (the reference recipe) | 95.17% | 95.21% | 93.27% |

Runs 1 and 2 use identical settings and moved by 1.2 points float and 1.9 points quantized. Comparing run 2 with run 3 alone would suggest that augmentation is worth 2 points; run 1 shows that gap is inside the noise. Removing balancing (run 4) cost about 2 points, the same size as the spread between runs 1 and 2, so confirm its effect on your data across seeds.

**Evaluate the four stages with the same tool**

```bash
tenn_sc12_train eval    -m trained.h5                -d <data>        # float, kernelized
tenn_sc12_train convert -m trained.h5 -ts 256                         # writes the stateful model
tenn_sc12_train eval    -m trained_stateful.h5       -d <data>        # float, stateful
# quantize (next section), then:
tenn_sc12_train eval    -m <quantized stateful>.h5   -d <data>        # quantized
tenn_sc12_train eval    -m <quantized stateful>.h5   -d <data> -ak    # Akida
```

Akida evaluation runs at batch size 1 on the CPU and is slow; the stateful float evaluation uses large batches. UORED works the same way with `tenn_uored_train` and a `--split`.

**What the scripts report.** SC12: accuracy, macro F1 and weighted F1, with one prediction per clip from the mean of the 64 per-chunk logits. UORED: binary accuracy and macro AUROC per segment, averaging logits over the chunks, then a sigmoid. State is reset between segments in both. `print_history_stats` prints the minimum, maximum, mean and standard deviation of each training metric.

**A protocol for honest comparisons**

1. Fix the split and the evaluation path first. Compare stateful or Akida scores, not float kernelized ones, because that is what ships.
2. Train each design on at least three seeds; five is better. Report mean and standard deviation, and the worst run.
3. Pick the checkpoint with a validation set, never the test set (see Preparing your data).
4. If you ship a single best run, record its seed, code commit, command line and split ID with the model file, and say it is the best of N.
5. Name the metric exactly: "top-1 accuracy per 1-second clip", "macro AUROC per 1-second segment".
6. Put the four stage scores side by side so a quantization loss is visible, not averaged away.

**Seeding.** Set the NumPy and TensorFlow seeds, and create the model under a fixed seed. Even then, GPU kernels and data ordering can differ between machines, so treat seeds as a way to reduce variance, not to eliminate it.

## After training: stateful, calibrate, quantize, map

Four steps turn the trained kernelized file into something Pico runs. The Model Development Guide explains each; this is the short path with the command that produces each file.

**1. Convert to stateful.** The chunk size must equal the total subsampling.

```bash
tenn_sc12_train convert -m trained.h5 -ts 256        # saves the stateful model next to trained.h5
```

**2. Extract calibration samples** from the training set, so they pass through the same preprocessing as training. SC12 cuts 1,024 samples from the middle of each of `-b` clips; UORED takes `-ns` whole windows.

```bash
tenn_sc12_train   extract -m trained.h5 -d <data> -s calib_sc12.npz  -ns 1024 -b 100
tenn_uored_train  extract -m trained.h5 -d <data> -s calib_uored.npz -ns 256
```

**3. Quantize** to 8-bit weights and activations with int16 input.

```python
import numpy as np, quantizeml
from tf_keras.models import load_model
from quantizeml.models import QuantizationParams

stateful = load_model("trained_stateful.h5")
samples = np.load("calib_sc12.npz")["data"]                      # int16
quantized = quantizeml.models.quantize(
    stateful,
    qparams=QuantizationParams(activation_bits=8, weight_bits=8, input_dtype="int16"),
    batch_size=100, epochs=1, samples=samples)
quantized.save("trained_stateful_i8_w8_a8.h5")
```

The command-line form is `quantizeml quantize -w 8 -a 8 -id int16`; run it with `--help` for the model and calibration-file options.

**4. Convert, map and evaluate on Pico.**

```python
import akida, cnn2snn
model_akida = cnn2snn.convert(quantized)
model_akida.map(akida.devices()[0], hw_only=True)    # or akida.PicoIP() without the board
```

```bash
tenn_sc12_train eval -m trained_stateful_i8_w8_a8.h5 -d <data> -ak
```

**Keep these together** for every model you may ship, so any result can be reproduced and any file traced.

| Artifact | Why |
| --- | --- |
| Trained kernelized `.h5` | The training result; the source of every later file |
| Stateful `.h5` and chunk size | What was converted, and at which chunk size |
| Calibration `.npz` | Quantization is only as good as this file |
| Quantized stateful `.h5` | What `cnn2snn.convert` consumes |
| Seed, commit, command line, split ID | Reproducibility |
| The four stage scores | Shows where accuracy was lost |

**Before you call it done**

- [ ] The four stage scores are in one table and each step is within the expected drop.
- [ ] `map(..., hw_only=True)` succeeds and the summary shows one sequence on one `TNP_R`.
- [ ] Calibration data comes from your data, not from the zoo.
- [ ] Latency and chunk rate meet the real-time budget (see the Model Development Guide).

## Changing the architecture

You can build your own stack from `kernelized_block`, the same building block the zoo models use, as long as it stays a plain chain that converts to stateful and maps to Pico. Check each change against the table before you train.

```python
import tensorflow as tf
from tf_keras.layers import Input, Rescaling, SpatialDropout1D, GlobalAveragePooling1D
from tf_keras.models import Model
from akida_models.layer_blocks import kernelized_block

channels, subsampling = [8, 16, 32, 64], [4, 4, 4, 4]      # total ×256 = chunk size
inputs = Input(shape=(8192, 1), dtype=tf.int16, name="input")
x = Rescaling(1. / 2**15, 0, name="rescaling")(inputs)
for i, (ch, sub) in enumerate(zip(channels, subsampling)):
    x = kernelized_block(x, 32, ch, subsampling=sub, add_batchnorm=True,
                         relu_activation="ReLU", name=f"ssm_layer_{i}")
    x = SpatialDropout1D(0.1)(x)
x = kernelized_block(x, 32, num_outputs, subsampling=False, add_batchnorm=False,
                     relu_activation=False, name="ssm_layer_head")
x = GlobalAveragePooling1D(name="gap")(x)
model = Model(inputs, x)
```

| Change | What it does | Limit or caution |
| --- | --- | --- |
| More or fewer blocks | Depth and capacity | At most 8 layers in the hardware design |
| More or fewer channels | Capacity, size in memory | Powers of 2 up to 256 are the verified range |
| Subsampling per block | Where time is reduced | At most 4 in any one layer; the product sets the chunk size |
| Chunk size | Decision rate and latency | Power of 2; stream with exactly this size |
| `num_coeffs` and `repeat` | States per layer, `num_coeffs × repeat` | 64 states is validated; 256 is listed as possible |
| Input window | Training window and initial time constants | Power of 2 |
| Input channels | Multi-channel sensors | Packed first-layer row must fill a power-of-2 number of 32-bit words, at most 128 |
| Head | Classes, labels or a regression output | Raw logits; no activation on the last layer |
| Branches or skip connections | Not supported | The stateful conversion accepts only single-input kernelized layers |
| Layers other than the ones above | Untested | Conversion handles kernelized layers, stride, batch norm, ReLU, dropout and pooling only |

**Anomaly detection and per-step outputs.** For an autoencoder or any per-timestep output, leave out `GlobalAveragePooling1D` and train on the reconstruction. In `convert_to_stateful`, passing `threshold` adds a post-processing layer that compares input and output.

**Check the size, not just the shape.** The mapped summary reports the hardware program size in bytes. Read it after `map()` and compare it with the memory available in your Pico configuration.

**Order of work for a new design**

1. Sketch channels and subsampling so the product equals your chunk size.
2. Build the model, convert to stateful, quantize with random data, map with `hw_only=True`.
3. If it maps, train. If it does not, the error names the layer and the limit; change the design, not the limit.
4. Compare against the zoo architecture on the same split and several seeds before you keep the change.

## Anomaly detection (autoencoder)

Anomaly detection is trained differently from classification: the model learns to reconstruct normal signals, and a large reconstruction error marks an anomaly. No packaged anomaly detection model is included in `akida_models` or the public examples, so this section is design guidance, not a command to run.

**Principles that matter**

- **Avoid time-frequency front ends.** A spectral decomposition in front of the network removes the reason to use a state-space model. Start from the raw time series.
- **Train one general model on normal data only.** Keep labelled anomalies out of the training windows and check unlabelled data for hidden ones. An autoencoder trained on data that contains anomalies learns to reconstruct them.
- **Use long windows and a temporal bottleneck.** Long input windows (for example 16,384 time points) let the state-space memory work. Reducing time by 4 per encoder layer, and reversing it in the decoder, gives a bottleneck.
- **Avoid per-file models and heavily overlapping short windows.** These suit other model types and waste the state-space memory.

**Training and evaluation notes**

1. **Watch the input scale.** MSE depends on the absolute values, so scaling the input up to the int16 range changes the loss magnitude and the right learning rate. Re-tune the learning rate after any scale change.
2. **Average the error over a span shorter than the input window,** for example 128 points, and stream in chunks of that size (128 or 256) to see how detection behaves in deployment.
3. **Reset state on the training window.** When evaluating the Akida model, performance can collapse unless the state is reset every training window (for example every 16,384 points).
4. **Expect slow quantized evaluation.** The quantized stateful model is much slower to evaluate than the float model; use a subset while iterating.
5. **Set the threshold on normal validation data,** for example a high percentile of reconstruction error, and keep it adjustable. Changing the threshold means reloading the whole model onto Pico.

**Pico specifics.** `convert_to_stateful(model, 256, threshold=...)` appends a post-processing layer that compares the buffered input with the reconstruction. Check the current Pico documentation for supported distance measures, the maximum input vector size, and whether the decoder's upsampling is supported, before committing to a design that depends on them.

## Troubleshooting

Common causes are a stale package, a model that was already converted, a calibration file that does not match the data, and plain run-to-run variance.

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Training behaves differently from the documented recipe, or a script argument is missing | An older `akida_models` earlier on the path | Use a clean environment; print `akida_models.__file__`; reinstall |
| `Training of stateful model is not supported` | `-m` points at a stateful file | Train the kernelized `.h5`, then run `convert` |
| Quantized accuracy falls more than about 1.5 points below stateful float | Calibration data is random, from another dataset, or too narrow | Extract a calibration file from your training set with `extract`; cover all classes |
| `eval -ak` fails on a float model | The Akida path needs the quantized stateful model | Quantize first, then evaluate with `-ak` |
| Accuracy well below the zoo figure on the same dataset | No class balancing, wrong input scale or dtype, wrong window length | Check int16 input, the `Rescaling` layer and the balanced sampler against the reference recipe |
| Two runs with the same settings differ by 1 to 2 points | Normal run-to-run variance | Run several seeds and compare means (Evaluation and reproducibility) |
| Validation score looks excellent, deployment score does not | Validation is the test set, or the split leaks recordings | Use a separate validation set and a source-level split |
| An assertion that the timestep must divide the signal length | Window length is not a multiple of the chunk size | Choose the window as a multiple of the chunk size, both powers of 2 |
| Mapping fails after an architecture change | A layer subsamples by more than 4, or the first-layer input row is too large | Change the design; see Changing the architecture |
| GPU out of memory | The kernelized layer convolves a whole window per sample | Reduce batch size or window length |
| `uored_full.h5` or split files not found | The raw recordings are not bundled with the examples | Obtain UORED-VAFCLS and point `-d` at the directory with the split JSON |
| SC12 preparation fails or is very slow | `ffmpeg` missing, or the first 2 GB download | Install `ffmpeg`; later runs use the cache |
| Akida evaluation is very slow | It runs at batch size 1 on the CPU | Expected. Evaluate a subset while iterating |
| Autoencoder output flat-lines after quantization, or Akida accuracy collapses on long streams | State not reset on the training window, or calibration scale or version mismatch | Reset state every training window; recheck versions and calibration scale |

The GPU memory row is general reasoning about the layer, not a documented failure.

## Notes and limits

- **Fine-tuning.** The fine-tuning learning rates, epoch counts and freezing schedule are starting points, not validated results.
- **Accuracy.** Published accuracy figures apply to the datasets named above and do not transfer to other data. Training results vary from run to run.
- **Quantization.** The scripts use post-training quantization only. Expect a cost of about 0.3 to 1 point on the reference models.
- **Versions.** Commands and options were written for `akida` 2.19.x and `akida_models` 1.14.x.

## References

- [MetaTF documentation](https://doc.brainchipinc.com/index.html): API reference and hardware constraints.
- [bc_cloud_pico](https://github.com/Brainchip-Inc/bc_cloud_pico): runnable notebooks and platform specification for the Akida Pico FPGA cloud platform.
- [`akida_models` on PyPI](https://pypi.org/project/akida-models/): model definitions, stateful conversion and the `tenn_sc12_train` and `tenn_uored_train` scripts.
- Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition* (Speech Commands v0.02, CC BY 4.0).
- UORED-VAFCLS: University of Ottawa rolling-element bearing dataset. Check the dataset's licence and citation requirements before use.

© 2026 BrainChip Holdings Ltd. All rights reserved.
