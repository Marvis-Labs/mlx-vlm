# Model Testing Process

Model-path CI is selected by capability rather than assuming every model exposes the same interface.

## Test layers

Every configured model runs a synthetic test first. The runner instantiates the architecture with the tiny random-weight profile from `ci/model_path.yaml`, applies the selected scenarios, and checks construction, modality paths, output shapes, dtypes, and finite values. Synthetic output is never judged for semantic correctness.

A real-checkpoint smoke test then downloads the checkpoint pinned in `ci/model_path.yaml`, verifies the declared model type, loads the model and processor, applies the checkpoint's own chat template, and executes the selected scenarios deterministically. Real tests enforce output contracts and minimal semantic assertions where the capability supports a stable oracle.

Model-specific adapters may translate a shared scenario into processor-specific control tokens or input fields. They must not change the scenario's underlying fixture, question, or expected result.

## Shared scenarios

The canonical prompts, deterministic fixture generators, generation settings, and assertions live in `ci/scenarios.yaml`.

| Capability | Fixed scenario | Real-checkpoint expectation |
|---|---|---|
| Text completion | Continue `The sequence is 1, 2, 3,` | Non-empty output |
| Text instruction | Ask for `2 + 2` with a number-only response | Normalized output equals `4` |
| Vision-language | Ask the color of a generated red square | Normalized output equals `red` |
| OCR | Transcribe a generated image containing `MLX 42` | Normalized output contains `MLX 42` |
| Image generation | Generate a red square on white | Valid non-uniform image with red-channel dominance |
| Image editing | Change the generated red square to blue | Valid non-uniform image with blue-channel dominance |
| Embedding | Compare matching and mismatched text/image pairs | Matching similarity exceeds mismatch |
| Detection | Locate the generated red square | Non-empty in-bounds coordinates |
| Segmentation | Segment the generated red square | Valid non-empty mask |
| Audio-language | Describe a deterministic 440 Hz tone | Non-empty output |

## Determinism and formatting

Shared scenarios use seed zero and temperature zero. Text and multimodal prompts are passed through the checkpoint processor rather than manually embedding model-specific chat syntax. Image and audio fixtures are generated from the scenario description so CI does not depend on external media.

Exact token sequences and exact generated pixels are not cross-model assertions. Model-specific regression tests may add stronger golden checks when they are stable.
