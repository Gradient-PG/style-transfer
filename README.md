# Style Transfer with Stable Diffusion 

This repository uses code from [Dream Styler](https://nmhkahn.github.io/dreamstyler/) please visit their website and read their paper to get to know more about model design.

## Running the app

```bash
make run
```

Before running the app make sure you have some styles trained, you have set control-net settings and est the demo_config
for style thumbnails.

## Config

In the `./configs/train.yml` are paths to training images for all styles and prompt related to each image. 

In the `./configs/demo_config.yml` are descriptions of trained styles and path to thumbnail images per style.

In the `./configs/config.yml` allow to specify what Control-Nets should be used when transferring a style onto an image. Feel free to experiment with changing combinations and checking results.

## Useful commands

### Training new style from an image

```bash
make train TOKEN="tstN"
```

- **TOKEN**: Identifier of the style, in the `./configs/train.yml` file.

For training make sure you fill out the `./configs/train.yml` with the information about your training style samples.

For the need of storing multiple style images, for training of a single style, not just one as in original paper we use
a config file and place all descriptions and paths there.

```yaml
datasets:
  TOKEN:
    - path: "./data/path/to/your/image"
      prompt: "Picture of description of what is inside the style training image, in the style of {}"
```

Be sure to replace the `TOKEN` with the actual token that indicates your style, following the example above `"tstN"`.

### Text-to-Image generation

```bash
make test-text TOKEN="tstN" PROMPT="a cat"
```

- **TOKEN**: Already trained identifier of a style.
- **PROMPT**: Short description of what you want the image to contain, only the objects.

Output can be found in the folder `outputs/{TOKEN}/` with the name similar to `Painting of {PROMPT}.png`

Warning! if the command for generation is run multiple times for the same TOKEN & CONTENT results will be overwritten.

### Style-Transfer

```bash
make test-style TOKEN="tstN" CONTENT="girls/test00.jpg" PROMPT="face of a beautiful woman"
```

- **TOKEN**: Already trained identifier of a style.
- **CONTENT**: Image from which content should be taken for the transfer. This time remember to add the file extension
  and make sure the file is present in `data/test/`
- **PROMPT**: Helping prompt with description of what is in the content image.

Output can be found in the folder `outputs/{TOKEN}/` with the name `{CONTENT}.png`

Warning! if the command for generation is run multiple times for the same TOKEN & CONTENT results will be overwritten.

### Style-grids

```bash
make test-stgrid TOKEN="tstN" PROMPT="face of a beautiful woman"
```

We added a possibility to make style grids, so to check style transfer on multiple images at once.
Currently, it is set generate a grid of entire sub-dir `girls/` in the `./data/test/` directory. As it is generating images with prompt `"beautiful woman"`

## Problems and future notes

A lot of styles lack content fidelity it would be highly beneficial to try to improve it, with something like InstantID models, maybe.

Main way we can improve fidelity id to up the weight of canny control-net however this comes at a tradeoff of loosing a lot of stylistic value.
We think incorporating a separate part of just transferring information about face can lead to a lot more pleasing results.
