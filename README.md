# MedImageInsight Zero-Shot Evaluation (Azure ML)

Tile H&E slides, generate **MedImageInsight-Premium** embeddings, and rank images using cosine
similarity to Responder and Non-Responder text prompts. **No model training is performed.**

```text
SVS images -> Tile embedding -> Saved image embeddings -> Prompt scoring -> ROC/AUC
```

The embedding step uses file names to select images, but response labels do not influence embedding
generation. Scoring uses the labels for evaluation. Separating these steps allows new prompts to be
tested against saved embeddings without sending image tiles again. Preprocessing runs on CPU;
inference runs on the remote Foundry service.

**For research and evaluation only. Not for diagnosis, treatment, or clinical decision-making.**
Supply your own authorized data and Azure resources. Dataset-specific results are not distributed.

## Folder Layout

This folder is the project root. Open it directly in VS Code and run commands and notebooks from here.
No files in its parent directory are required.

```text
component/                 Azure ML command components
environments/              Job environment, including OpenSlide
scripts/                   Shared preprocessing, API client, scoring, and tests
00_local_zero_shot.ipynb   Local image-modality smoke test
01_aml_zero_shot_roc.ipynb Two-step Azure ML evaluation pipeline
labels.json                Modality prompts for the local notebook
sample.env                 Configuration template
.env                       Your private settings (ignored; create locally)
data/                      Your private images and label sheet (ignored; create locally)
outputs/                   Generated artifacts and private reference copies (ignored)
```

## Setup

1. Create a Python 3.12 environment and install [requirements.txt](requirements.txt):

   ```sh
   pip install -r requirements.txt
   ```

2. Copy [sample.env](sample.env) to `.env` **in this folder**, then fill in your own values.
   Select the same Python environment as the notebook kernel. Packages are not installed by notebook cells.
3. For the local notebook, create `data/` and place authorized PNG, JPEG, or SVS images there.
4. For the Azure ML notebook, configure an existing workspace, CPU cluster, and `uri_folder` data asset
   containing SVS slides. Set `HNE_DATA_NAME` to that asset's name; set `HNE_DATA_VERSION` to pin a version,
   or leave it empty to resolve the latest version. This notebook does not provision those resources.
5. Create your private label sheet at `data/labels.csv`, or set `LABELS_CSV` to another path in `.env`.
   See [Label Sheet](#label-sheet) for the required columns.
6. Configure a MedImageInsight-Premium deployment and the identities authorized to call it.
   Run cells in order only when you intend to make API calls or submit jobs; these operations can incur charges.

Offline tests use synthetic data and mocked API responses, with no Azure access or inference charges:

```sh
python scripts/test_zero_shot.py
```

## Notebooks

| Notebook | Purpose | Execution |
|---|---|---|
| [00_local_zero_shot.ipynb](00_local_zero_shot.ipynb) | Compare images with CT, H&E, X-ray, and MRI prompts in [labels.json](labels.json) | Local preprocessing and remote inference |
| [01_aml_zero_shot_roc.ipynb](01_aml_zero_shot_roc.ipynb) | Tile slides, score response prompts, and calculate ROC/AUC | Azure ML CPU cluster and remote inference |

The local notebook is an **API smoke test**, not a disease classifier or treatment-response evaluation.
For SVS input it prepares a low-resolution overview, not individual cellular-resolution tiles.

## Authentication and API

- The local client uses an existing Azure CLI sign-in through `AzureCliCredential`, with
  `process_timeout=120`. Do not embed keys or tokens in source files.
- On Windows, the helper can add the standard Azure CLI installation directory to the current process's
  PATH without changing persistent environment settings.
- Azure ML steps use the compute's managed identity through `ManagedIdentityConfiguration`.
  Configure its endpoint permissions separately from the local user's permissions. Set
  `AML_MANAGED_IDENTITY_CLIENT_ID` only for a user-assigned identity; leave it empty for a system-assigned identity.
- The client calls `https://<resource>.services.ai.azure.com/providers/microsoft/v2/embed`
  and requests an Entra token for `https://ai.azure.com/.default`.
- Images and text are sent in separate requests. The implementation consumes 1024-dimensional vectors
  from `embeddings.float` and normalizes them for cosine similarity.

Use your own endpoint and deployment settings. The [official Premium API guide](https://learn.microsoft.com/en-us/azure/foundry/how-to/healthcare-ai/deploy-medimageinsight-premium)
describes model access and request formats; its authentication example uses an API key.

## Pipeline Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `target_mpp` | `0.5` | Requested microns per pixel; a preprocessing choice, not a validated model requirement. |
| `tile_size` | `512` | Output tile width and height in pixels. |
| `occupancy_threshold` | `0.20` | Tissue occupancy threshold for candidate tiles. |
| `max_tiles` | `32` | Maximum selected tiles per image. |
| `batch_size` | `8` | Maximum tiles per image-embedding request. |

Candidate tiles are sampled at evenly spaced indices in row-major order, rather than ranked by tissue
density. **A capped sample does not evaluate all tissue.** Inspect `candidate_tiles`, `selected_tiles`,
and `sampling_fraction` in the generated tile metadata to assess coverage.

The notebook also defines positive and negative prompts. Record prompts and preprocessing parameters
with every experiment; changing them changes the evaluation conditions.

## Resolution Handling

The tiler reads microns-per-pixel metadata, selects a suitable pyramid level, and resizes the region.
It does not assume that a fixed level number represents a particular magnification.
Integer source-pixel dimensions can introduce a small difference between `effective_mpp` and
`target_mpp`. The tiler checks this difference against `MPP_TOLERANCE` (1%) and rejects unsupported
or inconsistent resolution settings. The notebook includes a scale and sampling inspection step.

## Label Sheet

[scripts/cohort.py](scripts/cohort.py) extracts file names and labels from a CSV with these columns:

| Column | Purpose |
|---|---|
| `H&E Images` | Image path or file name. The base file name must match a slide in the data asset. |
| `Responder/Not Responder` | `Responder`, `Non-Responder`, or an empty value. Negative spelling variants are normalized. |

- Unlabeled rows are excluded before embedding and scoring. The source CSV is not modified.
- Evaluation is per image, without aggregation or deduplication by subject. Confirm that this unit
  is appropriate for your dataset and account for related images in any later validation study.
- **File names and labels are not anonymized data.** Names may contain patient or specimen identifiers.
  Protect the derived cohort as carefully as the source sheet.

The notebook creates `outputs/cohort.csv` and registers it as a pipeline input. It can be regenerated
from the private label sheet, so it must not be included in a public release.

To check the CSV locally without contacting Azure:

```sh
python scripts/check_dataset.py --csv-only
```

To also compare names against Blob Storage, configure `HNE_BLOB_CONTAINER_URL` and `HNE_BLOB_PREFIX`
in `.env`, then run the command without `--csv-only`. It lists files but does not download images or call the model.

## Scoring and Interpretation

Responder is the positive class. Each tile receives
`cosine(positive prompt) - cosine(negative prompt)`; the image score is the mean across its selected tiles.

- Predict Responder when the score is greater than zero; otherwise predict Non-Responder.
- The threshold is fixed. There is no training, probability calibration, or cross-validation in this workflow.
- Missing embeddings or inconsistent cohorts cause an error rather than silently dropping images.
- Consider ROC-AUC, balanced accuracy, class-specific metrics, and the confusion matrix together.
  Accuracy alone can be misleading with imbalanced labels.
- Cosine similarity is a ranking signal, not confidence or a diagnostic probability.
- Choosing prompts using the evaluation labels is not independent validation. Generating embeddings
  successfully does not establish predictive performance for a clinical task.

Plot your own saved evaluation results with:

```sh
python scripts/plot_zero_shot.py outputs/named-outputs/evaluation
```

The notebooks use the same plotting helper. Charts become embedded notebook output when executed.

## Data Handling and Publication

- `.env`, `data/`, and `outputs/` are local-only and ignored by Git. Keep the configuration template empty
  of resource identifiers and credentials. Do not distribute clinical inputs or generated results.
- Treat cohorts, prepared images, embeddings, predictions, tile metadata, and metrics as sensitive derived data.
- Clear all notebook outputs and execution counts before publication. Review Markdown, comments, file names,
  resource settings, and logs. Ignoring `outputs/` does not remove embedded notebook output.
- Review staged files and any existing repository history. Ignore rules do not protect already tracked files
  or files uploaded manually through a browser or archive.
- PNG conversion does not remove identifiers burned into pixels. Inspect images before sending or sharing them.
- Azure ML component code is scoped to `scripts/`; authorized data is supplied separately as job inputs.

## Operational Notes

- The job environment installs OpenSlide through conda; `openslide-python` is its Python binding.
  The environment also constrains the pixman version used by the native library.
- The client does not automatically retry failed inference requests. Inspect existing job status and
  partial outputs before resubmitting to avoid duplicate work and charges.
- Re-scoring saved embeddings still requests text embeddings, but does not re-send image tiles.

## References

- [Deploy and use MedImageInsight Premium](https://learn.microsoft.com/en-us/azure/foundry/how-to/healthcare-ai/deploy-medimageinsight-premium)
- [Read and write data in Azure ML jobs](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-read-write-data-v2?view=azureml-api-2)
- [Legacy Azure ML zero-shot example](https://github.com/microsoft/healthcareai-examples/blob/main/azureml/medimageinsight/zero-shot-classification.ipynb)

The legacy example uses a different API schema. Refer to the Premium documentation for the current payload.
