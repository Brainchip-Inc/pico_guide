# Resources used for the Akida Pico guides

Sources behind the *Model Development Guide* (**MDG**) and the *Model Training & Fine-Tuning Guide* (**TFG**), compiled 2026-11-11.

"Read" means I opened and read the content. "Surfaced only" means it appeared in a search result and I did not open it. Nothing in the second group was used as a source.

## 1. Meeting notes

| Resource | Used for |
| --- | --- |
| `plan.md` (local, `/home/akayyam/Pico/plan.md`) | Transcript fragment from the meeting. It holds only the closing minutes: a feedback round with Nikunj Kotecha, and reuse of the structure for Sequora. It has no content about the guides themselves. Used only for the "Reuse" open item. |

## 2. Requirements and specifications (SharePoint)

| Resource | Read | Used for | Guide |
| --- | --- | --- | --- |
| [Akida Pico PRD](https://brainchipinc.sharepoint.com/sites/MarketingSales/Shared%20Documents/Marketing_2026-25/PRDs/Akida%202/Akida%20Pico/Akida%20Pico%20PRD.docx) | Yes, full | The required guide structure ("How To Guides for Pico Models"), application KPIs, input and output requirements, retraining and reload assumptions, the release-approval constraint | Both |
| [Akida Pico 2024 Specification](https://brainchipinc.sharepoint.com/sites/Research/Shared%20Documents/Akida%20Pico%20Development/Original%202024%20Pico%20HW%20Design%20Documents/Akida%20Pico%202024%20Specification.docx) | Yes, full | Legacy hardware limits: 8-bit weights and activations, channel and state limits, layer types, no skip connections | MDG, TFG |
| [Pico Analysis Executive Summary, 2025-09-30](https://brainchipinc.sharepoint.com/sites/Research/Shared%20Documents/Akida%20Pico%20Development/2025-09-30%20-%20Akida%20Pico%20Analysis%20Executive%20Summary.pptx) | Yes, full | KWS latency target (under 250 ms, which conflicts with the PRD), application requirement tables | MDG |
| [Additional Pico Operators](https://brainchipinc.sharepoint.com/sites/Research/Shared%20Documents/Akida%20Pico%20Development/Additional%20Pico%20Operators.docx) | Yes, full | Possible extra operators: upsampling, stand-alone dense layer, L1/L2 distance with threshold, regression post-processing | MDG, TFG |
| [TENNs Cookbook Sketch](https://brainchipinc.sharepoint.com/sites/Research/Shared%20Documents/Advanced%20Research/Research%20Topics/TENNs%20cookbook/TENNs%20Cookbook%20Sketch.pptx) | Yes, full | Plan for public training code, Hugging Face models and tutorials | TFG |

## 3. Code and technical documentation

| Resource | Read | Used for | Guide |
| --- | --- | --- | --- |
| [bc_cloud_pico](https://github.com/Brainchip-Inc/bc_cloud_pico), `main` at commit 818dfc9 (cloned locally) | `README.md`, `requirements.txt`, `start-jupyterlab.sh`, `examples/Akida_Cloud_Specs.md`, both example READMEs, both notebooks, `kws_utils.py` and `fd_utils.py` (evaluation functions) | Pipeline steps and code, platform specification, power model and constants, latency measurement, streaming demo, calibration file shapes, expected metrics | Both |
| `akida_models` 1.14.1 source, downloaded from PyPI | `tenn_recurrent/` (model definitions, `convert_recurrent.py`, `sc12_train.py`, `uored_train.py`, `sc12_preprocessing.py`, `sc12_training_utils.py`), `layer_blocks.py` (`kernelized_block`), `custom_layers.py` (`Kernelized`, `Stride`), `cli.py`, `training.py` (parser options, `RestoreBest`), `setup.py` (entry points) | Architecture tables, stateful conversion steps, training recipes (optimizer, learning rate, loss, balancing, augmentation), CLI options, constraints in the conversion code | Both |

## 4. Jira (brainchip.atlassian.net)

| Ticket | Read | Used for | Guide |
| --- | --- | --- | --- |
| [AK-7479](https://brainchip.atlassian.net/browse/AK-7479), integrate the 12-class KWS model | Yes, with all comments | The SC12 training runs, ablations on augmentation and balancing, run-to-run variance, training commands, the stale-package problem | Both |
| [AK-7506](https://brainchip.atlassian.net/browse/AK-7506), make the fault model hardware compatible | Yes, with all comments | Subsample-limit error, the UORED redesign, AUROC across the four stages, 86 to 93% run-to-run range, hardware program size | Both |
| [AK-7507](https://brainchip.atlassian.net/browse/AK-7507) and [AK-7508](https://brainchip.atlassian.net/browse/AK-7508), TNP-R input-size and input-channel constraints | Yes | First-layer packed-row rule | Both |
| [AK-7446](https://brainchip.atlassian.net/browse/AK-7446), anomaly detection from applied research | Yes, with all comments | Window length, error-averaging span, state reset, input-scale caveat, evaluation time, closed as stale | Both |
| [AK-7516](https://brainchip.atlassian.net/browse/AK-7516), Pico documentation visibility on MetaTF | Yes | Where Pico model zoo and constraints documentation was missing | MDG |
| [AK-7518](https://brainchip.atlassian.net/browse/AK-7518), MetaTF developer-experience audit | Yes | Background on documentation quality goals; no content taken from it | Neither |
| [HUB-3](https://brainchip.atlassian.net/browse/HUB-3), Pico Model Zoo epic | Yes | Scope decisions: what is out of scope for public release, including client-data models | Both |
| [HUB-22](https://brainchip.atlassian.net/browse/HUB-22), SC12 example | Yes | Quantization command, calibration pack, `-ak` evaluation, dataset licence, instruction not to estimate hardware numbers | Both |
| [HUB-24](https://brainchip.atlassian.net/browse/HUB-24), UORED example | Yes | Planned rework of the fault classification example | Both |

I also ran a Jira text search for "pico" and read the key and title of about 50 results to find the tickets above. Only the tickets in this table were opened.

## 5. Asana (workspace brainchip.com)

| Resource | Read | Used for | Guide |
| --- | --- | --- | --- |
| [Akida Pico Models 2025](https://app.asana.com/1/754105566748081/project/1210477579865437) (project description and all 17 tasks) | Yes | Project background and compute limit (32 MACs per clock, SRAM-resident models), applications, the anomaly detection history and results table | Both |
| [[Akida Pico Cookbook]: Improve Performance of KWS model](https://app.asana.com/1/754105566748081/project/1210477579865437/task/1213531883908501) | Yes | Open task with no notes; open-items entry | TFG |
| [8-bit Pico (2025)](https://app.asana.com/1/754105566748081/project/1210775846141012) (project metadata only) | Partly | Confirmed the hardware and software workstreams; no content used | Neither |
| Akida PICO (8-bit) KWS, project `1207937668228770` (task list) | Yes | Background on the 8-bit hardware work; no content used | Neither |
| TENNs Cookbook, project `1208513746520582` (five tasks, via search) | Titles and notes only | Context for the cookbook and licensing effort | TFG |
| [Migrate Model interactive Guide to Developer Model page](https://app.asana.com/1/754105566748081/project/1209897724395513/task/1210828912026166) | Yes | Context only; no content used | Neither |

## 6. Slides supplied by the author (`figs/`)

| Slide | Status | Used for |
| --- | --- | --- |
| Always-On KWS Pico Core Power, dated 2026-04-02 (`figs/pico_kws_core_power.png`) | Embedded in the Model Development Guide | The 64 µW breakdown (50 µW active average plus 14 µW leakage), core size, battery life. It confirms the power model in the guide. |
| TENNs-R Bearing Fault Detection (`figs/uored_fault_detection_pico.png`) | Embedded in the Model Development Guide | Measured latency and throughput at 50 MHz, estimated power and energy, 16,564 parameters, the 240-segment test set. Its numbers reproduce from the guide's power model. |
| A slide built on private client data | Not used and not committed | Excluded because models built on client data must never be published (HUB-3). The file is listed in `.gitignore`. |
| Pico vs. Syntiant vs. Akida 1 IP comparison | Not used and not committed | Competitor figures are third-party claims and a marketing decision. Its energy figure for Pico (about 0.8 µJ) differs from the guide's (about 1 µJ) and is flagged as an open item. |

## 7. Searched but not opened

SharePoint search surfaced these. I did not open them and took nothing from them: Akida Pico datasheets and product briefs, the Pico FPGA cloud product brief, the Pico PRD in the `Obsolete` folder, the Akida Pico Autonomous Mode PRD, the Sequora PRDs and decks, the Analog Devices workshop deck, and the TENNs Model matrix spreadsheet.

If the guides are reworked, the datasheet, the FPGA cloud brief and the TENNs Model matrix are the first places to check for hardware and model details I did not see.

## 8. Could not access or not fetched

| Resource | Status |
| --- | --- |
| Confluence | Search returned 403 (the app is not installed on the instance) |
| BrainChip internal git (`git.corp.brainchipinc.com`, including `pico_model_factory`) | Not reachable. Training code was taken from the public `akida_models` package instead |
| [MetaTF documentation](https://doc.brainchipinc.com/index.html) | Linked in the guides but not fetched, so hardware-constraint statements were not checked against it |
| Slack, Google Drive, Notion and other connectors listed as needing authorization | Not authorized in this session, not searched |

## 9. Corrections made while writing

Claims I checked against the sources and changed:

- The "augmentation is worth 2 points" reading in AK-7479 compares two different runs; a repeat run shows it is within run-to-run noise.
- The 2-point effect of removing class balancing is the same size as the spread between identical runs, so the guides say to confirm it on your own data.
- The earlier UORED model mapped as two sequences, with the first layer in software, not as one.
- The extract option is `-s`, not `-sf`, per the `akida_models` 1.14.1 source.

© 2026 BrainChip Holdings Ltd. All rights reserved.
