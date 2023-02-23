# Process
To begin, I looked for ready-to-use libraries (like [this](https://github.com/dovahcrow/patchify.py)) to make the work easier.
But I was not satisfied with the limitations of the functionality:
- squared shifts (x-shift == y-shift)
- 2d-3d images separation

# TODO
- add more tests; corner cases;
- separate merge logic and reading splits;
- add checks and limitations (ex.: to big sliding window size)
- add entry point;
- add the ability to parallelize calculations (on script level). 