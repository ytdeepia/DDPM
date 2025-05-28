# Unadjusted Langevin Algorithm

## What is this repo

This repository contains all the code used to generate the animations for the video ["Diffusion Models: DDPM"](https://youtu.be/EhndHhIvWWw) on the youtube channel [Deepia](https://www.youtube.com/@Deepia-ls2fo).

All references to the script and the voiceover service have been removed, so you are left with only raw animations.

There is no guarantee that any of these scripts can actually run. The code was not meant to be re-used, so it might need some external data in some places that I may or may not have put in this repository. 

You can reuse any piece of code to make the same visualizations, crediting the youtube channel [Deepia](https://www.youtube.com/@Deepia-ls2fo) would be nice but is not required.

## Other files

In this repo, you will also find a ``src`` folder containing all the code used to train the diffusion models used in the video, and to create new samples. 
This also include a very compact implementation of the DDPM paper (under 100 lines of code) displayed in the video. 
This code is available under the file ``src/training_minimal.py`` and ``src/inference_minimal.py``.

## Environment

There are two different environements for this repo, one for animations (``animations/requirements.txt``) and one for the training and sampling of diffusion models (``src/requirements.txt``). 
I switched to ``uv`` to manage my environments and recommend using it. 
You can create a ``.venv`` from the requirements using these commands:
```bash
python -m venv .animation_venv
source .animation_venv/bin/activate  
pip install -r animations/requirements.txt
```
and: 
```bash
python -m venv .exp_venv
source .exp_venv/bin/activate  
pip install -r src/requirements.txt
```
## Generate a scene

You should move to the animation subfolder then run the regular Manim commands to generate a video such as:

```bash
manim -pqh scene_1.py --disable_caching
```

The video will then be written in the ``./media/videos/scene_1/1080p60/`` subdirectory.

I recommand the ``--disable_caching`` flag when using voiceover.
