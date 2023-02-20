# supervisely_assignment

## Dev installation
- Create and activate environment.
- Launch `pip install -e \."[dev,tests]"` (zsh) `pip install -e [dev,tests]` (bash)

# Split and Merge
[Main report](src/module/scripts/splitmerge/README.md)

# DAVIS merge script
[Main report](src/module/scripts/davis_merge/README.md)

## Merge script
```
usage: merge_davis [-h] [--wks WKS] [--fr FR] [--alpha ALPHA] [--thck THCK] images annotations savepath name

positional arguments:
  images         folder with subfolders containing images
  annotations    folder with subfolders containing annotations
  savepath       path to save
  name           name of final video

optional arguments:
  -h, --help     show this help message and exit
  --wks WKS      number of subprocess for parallel image annotation; defaults to 1
  --fr FR        framerate; defaults to 15
  --alpha ALPHA  transperancy factor for annotation; defaults to 0.5
  --thck THCK    annotation border thickness; defaults to 1
```

## Example
Clickable image.
[![example](artifacts/example.png)](https://drive.google.com/file/d/1ccyOrO386El73O34CmxcsDpJTTuIsruq/view?usp=share_link)
