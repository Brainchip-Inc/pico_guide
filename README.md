# Akida Pico Guides

Draft guides for building, training and fine-tuning models for **BrainChip Akida Pico**, an ultra-low-power, event-based neural IP for always-on 1-D sensing (audio, vibration, biosignals).

> **Status: draft for review.** Both guides were assembled from internal documents, tickets and the public `akida_models` 1.14.1 source. The pipeline was not re-run on the Pico FPGA while writing them. Each guide ends with a checklist of open items that need an owner's answer before publication.

## Contents

| File | What it is |
| --- | --- |
| [Akida_Pico_Model_Development_Guide.md](Akida_Pico_Model_Development_Guide.md) | From a trained 1-D model to a quantized network running on Pico: recurrent TENN concepts, the two zoo models, design rules and hardware limits, the convert-quantize-map workflow, validation, on-device measurement, application status, troubleshooting. |
| [Akida_Pico_Model_Training_and_Fine-Tuning_Guide.md](Akida_Pico_Model_Training_and_Fine-Tuning_Guide.md) | How to get trained weights: environment, data preparation, training from scratch, fine-tuning, evaluation and reproducibility, post-training steps, architecture changes, anomaly detection notes, troubleshooting. |
| [resources.md](resources.md) | Every source used to write the guides, what was taken from each, what was only seen in search results, and what could not be accessed. |

## How to read them

1. Start with the **Model Development Guide** to learn what Pico runs and what limits a model must meet.
2. Use the **Training & Fine-Tuning Guide** when you need weights for your own data. It stops where the first guide's workflow begins and links back to it.
3. Check **resources.md** to trace any claim to its source.

## Audience

ML engineers at Pico licensees, and BrainChip field engineers supporting them. The guides assume Keras and TensorFlow experience and no prior Akida knowledge.

## Scope

- **Covered with runnable models:** 12-class keyword spotting (`tenn_recurrent_sc12`) and 4-label bearing fault classification (`tenn_recurrent_uored`).
- **Covered as status and design guidance only:** 1-D anomaly detection, wake word and health monitoring. These are Pico requirements without a packaged public model yet.
- **Toolchain:** MetaTF (`akida`, `cnn2snn`, `quantizeml`, `akida_models`). Examples target the Akida Pico FPGA cloud platform and its software model.

## Known limits of this draft

- The **fine-tuning recipe is proposed, not validated.** The public training scripts have no tuning mode; learning rates, epoch counts and the freezing schedule are starting points to replace with measured values.
- The **power figures** (about 64 µW and 1 µJ for KWS) are estimates from hardware characterization, not FPGA measurements.
- The **hardware-limits table** mixes current mapping checks with the 2024 hardware specification. The mapping error is the final authority.
- **Before external release,** remove the internal Jira references, confirm approval to describe the recurrent TENN layer externally, and keep any client-data models out entirely.

## Feedback

Open an issue or comment on a pull request. Useful feedback names the section, the claim, and the source or measurement that supports a correction.

© 2026 BrainChip Holdings Ltd. All rights reserved.
