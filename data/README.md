# Data layout

The FruitNeRF dataset is hosted on Zenodo at DOI 10.5281/zenodo.10869455. Do not commit the full dataset to GitHub.

Expected local layout after extracting the real dataset:

```text
data/
  external/
    FruitNeRF_Real/
      FruitNeRF_Dataset/
        tree_01/
          images/
          semantics_sam/
          semantics_unet/
        tree_02/
          images/
          semantics_sam/
          semantics_unet/
        tree_03/
          images/
          semantics_sam/
          semantics_unet/
  raw/
  processed/
```

`data/external` is ignored by Git.
