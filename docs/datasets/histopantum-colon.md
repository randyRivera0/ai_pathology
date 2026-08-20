# HISTOPANTUM colorectal subset

This dataset card documents the local colorectal subset used by Experiments 6
and 7. The image files are not committed to the repository. This project uses
them for research and education only, not for diagnosis or clinical
decision-making.

## Local inventory

The audited local copy contains 27,248 unique 224 x 224 RGB JPEG patches:

| Class | Patches |
| --- | ---: |
| `non-tumour` | 10,299 |
| `tumour` | 16,949 |
| **Total** | **27,248** |

The files represent 40 TCGA case IDs and 40 slide IDs. In this local release,
each case has one slide. A case therefore corresponds approximately to one
independent patient, while each slide contributes many correlated patches:

```text
TCGA case (patient-level unit)
`-- one digital slide in this release
    `-- multiple 224 x 224 patches
```

Patch count is not patient count. Thousands of patches from a small number of
cases do not provide thousands of independent observations.

## Filename contract

Each filename follows this structure:

```text
<TCGA slide ID>_<x>_<y>.jpg
```

For example:

```text
TCGA-3L-AA1B-01Z-00-DX1_10240_15360.jpg
```

The case ID is `TCGA-3L-AA1B`, the slide ID is
`TCGA-3L-AA1B-01Z-00-DX1`, and the final two fields are patch coordinates.
Experiments derive grouping identifiers from this contract instead of from
machine-specific paths.

## Shared experimental assignment

Experiments 6 and 7 use the same deterministic, case-disjoint assignment:

| Split | Cases | Slides | Patches | Non-tumour | Tumour |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 28 | 28 | 19,092 | 7,224 | 11,868 |
| Validation | 6 | 6 | 3,985 | 1,535 | 2,450 |
| Test | 6 | 6 | 4,171 | 1,540 | 2,631 |
| **Total** | **40** | **40** | **27,248** | **10,299** | **16,949** |

Every case and slide belongs to exactly one split. All patches derived from a
case remain together, preventing tissue from the same patient-level unit from
appearing in both training and evaluation.

The canonical case assignment has SHA-256:

```text
b8bca9f2baabca3e3bf3f118a321f8e65fa146f35858f597645aaeb1df85a37a
```

The exact per-patch manifests are committed independently under
`experiments/exp-6/outputs/` and `experiments/exp-7/outputs/`.

## Integrity observations

- All 27,248 relative paths are unique.
- All filenames match the expected slide-and-coordinate structure.
- No case or slide crosses split boundaries.
- One slide is available per case in this local colorectal release.
- Thirty-six cases contain patches from both classes; four are
  class-exclusive.
- Patch counts per case are uneven, so pooled patch metrics can be dominated by
  cases that contribute more patches.

## Limitations

- The dataset has only 40 independent cases despite containing 27,248 patches.
- The fixed test split contains 4,171 patches but only six cases.
- Patches from the same case share tissue, staining, scanner, and acquisition
  characteristics and are not statistically independent.
- Training, validation, and test data come from the same dataset domain.
- Internal performance does not establish robustness across hospitals,
  scanners, staining protocols, or external datasets.
- Additional augmented patches would not increase the number of independent
  patients.

More reliable future evidence should prioritize additional independent cases,
case-level cross-validation, balanced sampling across cases, uncertainty
reported over cases, and an external test source that is not used for training,
model selection, threshold selection, or hyperparameter tuning.
