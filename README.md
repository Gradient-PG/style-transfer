# style-transfer model 

_Diffusion_

## Useful commands

### Training new style form an image

```bash
make train TOKEN="tstN" PROMPT="a lonely tree in the field during sunset"
```
- **TOKEN**: Identifier of the style, in the `data/train/` folder should be an image with name structured `{TOKEN}.jpg` ex. `tst1.jpg`
- **PROMPT**: Short description of what is inside the image, try describing the objects without mentioning the style.

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
make test-style TOKEN="tstN" CONTENT="test00.jpg" PROMPT="face of a beautiful woman"
```
- **TOKEN**: Already trained identifier of a style.
- **CONTENT**: Image from which content should be taken for the transfer. This time remember to add the file extension and make sure the file is present in `data/test/`
- **PROMPT**: Helping prompt with description of what is in the content image.

Output can be found in the folder `outputs/{TOKEN}/` with the name `{CONTENT}.png`

Warning! if the command for generation is run multiple times for the same TOKEN & CONTENT results will be overwritten. 

## Config

In the `configs/config.yml` so far are only added possibilities to specify what Control-Nets should be used when transferring a style onto an image. Feel free to experiment with changing combinations and checking results.
