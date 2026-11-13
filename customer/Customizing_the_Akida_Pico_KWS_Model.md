# Customizing the Akida Pico Keyword Spotting Model

BrainChip Holdings Ltd. · November 2026

This guide shows how to adapt the Akida Pico keyword spotting (KWS) model to your own keywords: choose the keyword set, prepare data, fine-tune, quantize, map to Pico and check the result. It is a short path through the *Model Training & Fine-Tuning Guide* and the *Model Development Guide*, which have the full detail.

## What you can change, and what stays fixed

| You choose | Fixed by the Pico model |
| --- | --- |
| The keywords, and so the number of classes | Raw 16 kHz mono audio, as int16 samples; no FFT, MFCC or spectrogram front end |
| The training data, including your noise and speakers | A 1-second window of 16,384 samples for training |
| How the output is turned into a trigger (smoothing, thresholds) | A 256-sample streaming chunk, so one decision step every 16 ms |
| Whether to fine-tune the zoo model or train from scratch | 8-bit weights and activations with int16 input, mapped as one sequence on one `TNP_R` |

Keeping the architecture and changing only the output head means the customized model uses the same hardware structure as the zoo model. Measure its cycles per chunk after mapping rather than assuming they are identical.

## The path at a glance

1. Define the keyword set.
2. Prepare the data.
3. Fine-tune the zoo model, or train from scratch.
4. Convert to stateful, calibrate and quantize.
5. Map to Pico and check accuracy at every stage.
6. Build the trigger logic and check false accepts on your own audio.

## 1. Define the keyword set

The zoo model has 12 classes: 10 keywords, `silence` and `unknown`. Keep that structure for your own model: *N* keywords, plus `silence` for background noise, plus `unknown` for everything else a user might say.

- The `unknown` class is what stops ordinary speech from triggering a keyword. Give it a wide variety of words and speakers.
- Every extra keyword makes the task harder. Start with the fewest keywords that meet your product need.
- Choose keywords that sound different from each other and from common words.

## 2. Prepare the data

The model reads fixed windows of 16,384 int16 samples at 16 kHz, with shape `(N, 16384, 1)`.

- **Clips.** Pad shorter clips with zeros at the end and crop longer ones to 16,384 samples.
- **Scale.** Keep samples in the int16 range and use one scale for the whole dataset. Do not normalize each clip to its own peak, because a live microphone has no per-clip peak to use.
- **Coverage.** Record each keyword from many speakers, at different distances from the microphone, and in the noise conditions of your product.
- **Silence and noise.** Build the `silence` class from real background audio from your target environment.
- **Unknown.** Include many other words and speakers. The public Speech Commands v0.02 dataset can supply `unknown` and `silence` examples alongside your own keywords.
- **Split by speaker,** never by random clips. Clips from one speaker are near-duplicates; a random split inflates the score. Keep a validation set for choosing the best epoch and a separate test set you use once.
- **Balance the classes.** The reference recipe draws an equal number of examples per class each epoch, downsampling `unknown` and upsampling `silence`. Removing this cost about 2 points in one run, so check its effect on your data.

```python
import numpy as np

# clip: float waveform at 16 kHz, already on the dataset's single global scale (int16 range)
x = np.zeros(16384, dtype=np.float32)
n = min(len(clip), 16384)
x[:n] = clip[:n]                       # zero-pad or crop at the end
X = np.clip(x, -32768, 32767).astype(np.int16)[:, None]    # (16384, 1)
```

## 3. Fine-tune, or train from scratch

Fine-tuning starts from the zoo weights and needs less data and time than training from scratch. Train from scratch if your audio differs a lot from speech at 16 kHz.

> **Starting points, not validated results.** The learning rates and epoch counts below are suggested starting points based on general practice. BrainChip has not published validated fine-tuning results for Pico models. Tune them on your own validation set.

```python
import akida_models, tf_keras as keras
from akida_models.param_scheduler import CosineDecayWithLinearWarmup
from akida_models.training import RestoreBest

NUM_CLASSES = 8                                          # your keywords + silence + unknown

pre = akida_models.tenn_recurrent_sc12()                 # the zoo architecture, 12 classes
pre.load_weights("tenn_recurrent_sc12.h5")               # zoo kernelized weights

model = akida_models.tenn_recurrent_sc12(num_classes=NUM_CLASSES)
for layer in model.layers:                               # copy every block except the head
    if layer.name != "ssm_layer_head":
        layer.set_weights(pre.get_layer(layer.name).get_weights())

def compile_for(lr, steps):
    sched = CosineDecayWithLinearWarmup(lr, 0.025, steps)
    model.compile(
        optimizer=keras.optimizers.experimental.AdamW(learning_rate=sched, weight_decay=0.05),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()])

def best_epoch():                                        # one per phase: it cleans up after itself
    return RestoreBest(model, monitor="val_sparse_categorical_accuracy")

# Phase 1: new head only
for layer in model.layers:
    layer.trainable = layer.name == "ssm_layer_head"
compile_for(1e-3, steps_per_epoch * 8)
model.fit(train_ds, validation_data=val_ds, epochs=8, steps_per_epoch=steps_per_epoch, callbacks=[best_epoch()])

# Phase 2: all layers, lower learning rate
for layer in model.layers:
    layer.trainable = True
compile_for(1e-4, steps_per_epoch * 20)
model.fit(train_ds, validation_data=val_ds, epochs=20, steps_per_epoch=steps_per_epoch, callbacks=[best_epoch()])

model.save("my_kws.h5", include_optimizer=False)
```

`train_ds` and `val_ds` yield batches of `(int16 windows, integer labels)`. The head is the only layer with a new shape, so it starts from a fresh initialization.

To train from scratch instead, skip the weight copy and use the full recipe in the Training & Fine-Tuning Guide (AdamW, peak learning rate 0.01, cosine schedule, 200 epochs for the reference model).

Evaluate the untouched zoo model on your test set first. If your fine-tuned model does not beat that baseline, or a from-scratch run on the same split, train from scratch.

## 4. Convert, calibrate and quantize

```python
import numpy as np, quantizeml
from akida_models.tenn_recurrent.convert_recurrent import convert_to_stateful
from quantizeml.models import QuantizationParams

stateful = convert_to_stateful(model, 256)                 # streams 256 samples at a time

# Calibration: about 100 clips from your training set, 1,024 samples cut from the middle of each
samples = X_train[calib_idx][:, 8192:8192 + 1024, :].astype(np.int16)

quantized = quantizeml.models.quantize(
    stateful,
    qparams=QuantizationParams(activation_bits=8, weight_bits=8, input_dtype="int16"),
    batch_size=100, epochs=1, samples=samples)
```

Calibrate on your own data, covering every keyword, `silence` and `unknown`, and the noise conditions you expect. The calibration file shipped with the zoo model does not represent your data.

## 5. Map to Pico and check each stage

```python
import akida, cnn2snn
model_akida = cnn2snn.convert(quantized)
model_akida.map(akida.devices()[0], hw_only=True)          # akida.PicoIP() without the board
model_akida.summary()                                      # expect 1 sequence, 1 x TNP_R
```

Score the same held-out clips at four stages and compare them in one table:

| Stage | Expect |
| --- | --- |
| Float, kernelized | Your training result |
| Float, stateful | Within about 1 point of the stage above |
| Quantized, 8-bit stateful | A drop of about 0.3 to 1 point on the zoo model |
| Akida, mapped | Within about 0.3 point of the quantized score |

Score each clip by averaging the model's 64 per-chunk logits over the second and taking the argmax. Do not average softmax probabilities. `evaluate_stateful_full` in `kws_utils.py` in the `bc_cloud_pico` examples does this for the Akida model.

A large drop at the quantized stage usually means the calibration data does not match your deployment audio. A large drop at the mapped stage is a conversion or mapping problem; please report it to BrainChip.

## 6. Trigger logic and false accepts

A live system does not wait for a full second. It runs one chunk at a time and applies its own logic to the per-chunk outputs.

1. Call `model_akida.forward` once per 256-sample chunk, with input shape `(1, 1, 256, 1)` as int16. The state is carried between calls.
2. Smooth the logits with a sliding mean, then apply softmax.
3. Fire when a keyword's smoothed probability passes a threshold for a number of consecutive chunks.

The threshold and the smoothing window set the balance between false accepts and false rejects. Choose them on your own recordings: record hours of realistic background audio, count the false accepts, and measure how often real keywords are missed. The accuracy figure on a benchmark clip set does not tell you the false-accept rate of a product.

The mapped model measures its own cost. For the zoo KWS model at 50 MHz, one chunk takes about 1.45 ms against a 16 ms period, and the estimated average power is about 64 µW. Power is an estimate from hardware characterization, not an FPGA measurement. Measure your customized model's cycles with the method in the Model Development Guide.

## What to expect

- **Your data decides the accuracy.** The zoo figures (about 94 to 96% on Speech Commands, 12 classes) do not transfer to other keywords or noise.
- **Runs vary.** Two identical training runs can differ by 1 to 1.5 points, so compare designs over several seeds before drawing conclusions.
- **Keywords are set at build time.** Pico does not train. Changing a keyword means retraining off-device, then quantizing again and reloading the whole model.
- **BrainChip does not guarantee accuracy on customer data.** This guide shows the flow and the checks.

## Where to go next

- *Model Training & Fine-Tuning Guide*: full data preparation, evaluation protocol, architecture changes and troubleshooting.
- *Model Development Guide*: hardware limits, validation at every stage, deployment and power measurement.
- [MetaTF documentation](https://doc.brainchipinc.com/index.html) and the [bc_cloud_pico](https://github.com/Brainchip-Inc/bc_cloud_pico) KWS notebook, which runs the pipeline end to end on the Akida Pico FPGA cloud platform.

© 2026 BrainChip Holdings Ltd. All rights reserved.
