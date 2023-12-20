# Perspec2Equirec

## Introduction
The file `Perspec2Equirec.py` is the solution to the homework assignment.

## Installation
Run the following commands to create venv and install the necessary packages
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Below shows an example of code for carving out a perspective image from the original Equirectangular image using `Equirec2Perspec` and projecting the perspective back in an empty canvas of the same size using my `Perspec2Equirec`.

```python
import os
import Equirec2Perspec as E2P
import Perspec2Equirec as P2E

FOV, THETA, PHI = 60, 100, 0
equ = E2P.Equirectangular(os.path.join('src', 'image.jpg'))    # Load equirectangular image
persp_img = equ.GetPerspective(FOV, THETA, PHI, 720, 1080)
persp = P2E.Perspective(persp_img)
equ_img = persp.GetEquirectangular(FOV,THETA,PHI, equ._height, equ._width)
```

An example of usage can also be seen in `test.ipynb`

# Equirec2Perspec
## Introduction
<strong>Equirec2Perspec</strong> is a python tool to split equirectangular panorama into normal perspective view.

## Panorama
Given an input of 360 degree panorama
<center><img src="src/image.jpg"></center>

## Perpective
Split panorama into perspective view with given parameters
<center><img src="src/perspective.jpg"></center>

## Usage
```python
import os
import cv2 
import Equirec2Perspec as E2P 

if __name__ == '__main__':
    equ = E2P.Equirectangular('src/image.jpg')    # Load equirectangular image
    
    #
    # FOV unit is degree 
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension 
    #
    img = equ.GetPerspective(60, 0, 0, 720, 1080) # Specify parameters(FOV, theta, phi, height, width)
```

