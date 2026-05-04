# Documentation Index

All design docs live here. Read in roughly this order if you're new
to the project.

## Start here
| Doc | Why read it |
|---|---|
| **`path_b_implementation_roadmap.md`** | The master 6-week plan this project executes |
| **`comparison_kaggle_pipeline.md`** | Why we use ALIKED+LightGlue (Path B) instead of OnePose++ matcher |
| **`mobile_inference.md`** | Per-stage runtime patterns for mobile |

## Background — original OnePose++ pipeline
| Doc | Why read it |
|---|---|
| `codebase_overview.md` | Top-level architecture of the OnePose++ research repo |
| `pipeline_steps.md` | Step-by-step (A–S) walkthrough of OnePose++'s 3 pipelines |
| `sfm_and_descriptors.md` | What's in COLMAP outputs and how descriptors get sampled |
| `loftr_features.md` | What LoFTR is and how it's used in OnePose++ |

## Mobile deployment
| Doc | Why read it |
|---|---|
| `mobile_deployment.md` | Feasibility analysis: iOS+Android + ONNX Runtime |
| `mobile_inference.md` | Concrete per-stage implementation guidance |

## Upstream OnePose++ docs (reference only)
| Doc | What it is |
|---|---|
| `demo.md` | Original OnePose++ demo instructions (not directly relevant for us) |
| `dataset_document.md` | OnePose dataset format reference (not used by us) |

## Living docs in this project
- `architecture.md` — current architecture decisions for THIS project
- `decisions/` — ADR-style records of choices made (one file per decision)
