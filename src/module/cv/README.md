# Motivation of 3-part sructure
The main goal of experiments is to find the best way to create images for a synthetic set of data. 
We will compare experiments using the classification's quality on real data as a metric. 
All parts (except generation model) should be deterministic for this reason.

[Image constructor](utils/constructor.py) can be configurate with GenParams class. It contains local numpy random generator with fixed seed.

# Primitives generation
## Model
Of the proposed SOTA solutions the choice fell on Diffusion models. 
The main constraint was a shortage of PC (colab) capacity. 
So StableDiffusion 1.4 was chosen.
This approach is advantageous because, in addition to being resource-efficient, it accommodates a variety of generations (text2image, image2image).

## Separation
In practice, choosing the right prompts for even the production of basic primitives, like apples, is difficult.
As a result, backgrounds and primitives are generated/seek independently.
One of the advantages of this approach is the ability to automatically create annotation for segmentation tasks.

## Text2Img
Pros:
- a variety of generated primitives;

Cons:
- difficulty in the selection of good promts;