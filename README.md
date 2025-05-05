# VFX-Project2
This project focuses on building a panorama from multiple input images by detecting and matching features. We implemented a custom pipeline using Harris corner detection / MSOP feature detection and a simplified version of the SIFT feature descriptor / MSOP feature descriptor. 

## How to Recreate the Result

```bash
git clone https://github.com/clarence1121/VFX-Project2-image-stitching.git
cd your-repo-name
python3 vfx.py
python3 Harris_Corner_Detector.py
```

# Image Stiching

- team members 李孟學(R13922165), 温文安(R13944053)

This project focuses on building a panorama from multiple input images by detecting and matching features. We implemented a custom pipeline using Harris corner detection & MSOP feature detection and a simplified version of the SIFT feature descriptor & MSOP feature descriptor.

You could review our code and recreate all the result by visiting our [GitHub repo](https://github.com/clarence1121/VFX-Project2-image-stitching) .
### Bonus
- end to end alignment
- Blending methods 
- More than one feature detection or description(multiscale harris,harris and MSOP,SIFT)

## 1. Feature detection

### Harris Corner Detection

#### 1 Harris Algorithm

To find stable feature points, we implemented the Harris corner detector.

##### Steps:

1. Compute gradients $I_x$, $I_y$ using Sobel filters.
2. Form second-moment matrix:
  $$
   M = \begin{bmatrix}
   I_x^2 & I_x I_y \\
   I_x I_y & I_y^2
   \end{bmatrix}
$$
3. Smooth each component with a Gaussian filter.
4. Compute corner response:
	   $R = \text{det}(M) - k \cdot (\text{trace}(M))^2$
5. Normalize and threshold to keep strong responses.

#### 2 Adaptive Non-Maximum Suppression

To ensure good spatial distribution, we applied adaptive non-maximum suppression:

- Sort corners by \( R \) (descending)
- Retain a keypoint if no stronger one exists within a radius \( r \)
- Gradually reduce \( r \) until ~500 keypoints remain

#### Figures
<p align="center">
  <img src="https://hackmd.io/_uploads/SJzjo_1exx.jpg" height="200px" />
  <img src="https://hackmd.io/_uploads/HJxT3OJlgx.jpg" height="200px" />
  <img src="https://hackmd.io/_uploads/Hkep3dyege.jpg" height="200px" />
</p>
  <img src="https://hackmd.io/_uploads/SJlp3dkgge.jpg" height="200px" />

**Figure 3.0**: Raw image
**Figure 3.1**: Raw corners before NMS (red dots)  
**Figure 3.2**: Final 500 keypoints after adaptive NMS
**Figure 3.3**: Gradient images $I_x$ and $I_y$  


## Feature description

### SIFT (Leemeng)
SIFT-style Feature Description

#### 1 Orientation Assignment

Each keypoint is assigned a dominant orientation to ensure rotation invariance:

- Extract a 16×16 patch around the keypoint
- Compute the gradient magnitude and orientation at each pixel
- Build a 36-bin histogram of orientation
- Assign a dominant angle to a keypoint

#### 2 Descriptor Extraction

Each keypoint’s descriptor is built as follows:

1. Align the patch to the keypoint orientation
2. Divide into 4×4 cells (each 4×4 pixels)
3. For each cell, build an 8-bin histogram of orientations
4. Concatenate → 128-d descriptor
5. Normalize the vector

#### Figures

**Figure 4.1**: Keypoints with orientation arrows  
![orientation arrows  ](https://hackmd.io/_uploads/B1qPxY1elx.jpg)

**Figure 4.2**: Gradient arrows in a 16×16 patch (ex, randomly selecting two key points)
![Gradient arrows ](https://hackmd.io/_uploads/BJQxxt1xle.jpg)
**Figure 4.3**: Histogram of a sample 128-d descriptor  
![Histogram](https://hackmd.io/_uploads/Bk7xlt1leg.jpg)

### MSOP (Kevin)
#### 1. feature detection(corner detection)
The MSOP algorithm using multiscale harris corner dection instead of the tranditional harris corner detection , below is the same picture with different corner detection  algorithm and the response heat map.
<p align="center">
  <img src="https://hackmd.io/_uploads/r15lxY1exe.png" width="45%" />
  <img src="https://hackmd.io/_uploads/ryswgKkege.png" width="45%" />
</p>
<p align="center">
  <span style="display: inline-block; width: 45%; text-align: center;">harris descriptor</span>
  <span style="display: inline-block; width: 45%; text-align: center;">response heat map</span>
</p>

<p align="center">
  <img src="https://hackmd.io/_uploads/HkuvVY1gge.png" width="45%" />
  <img src="https://hackmd.io/_uploads/Hy3kmKJegl.png" width="45%" />
</p>
<p align="center">
  <span style="display: inline-block; width: 45%; text-align: center;">multiscale harris corner descriptor</span>
  <span style="display: inline-block; width: 45%; text-align: center;">response heat map</span>
</p>

**Multiscale Harris** corner detection constructs a Gaussian pyramid by progressively downsampling the image, then computes the Harris response at each level using separate smoothing parameters (sigma_d and sigma_i). At each scale, it applies thresholding and non-maximum suppression to detect keypoints,then rescaled back to the original image size. Finally, responses from all levels are upsampled and combined into a single response map (R_comb) using interpolation to highlight the strongest features across scales,reach detail in our github.

#### 2. descriptor(compute_msop_descriptor )
This function computes an MSOP (Multi-Scale Oriented Patch) descriptor for a given keypoint. It extracts a normalized patch around the keypoint, aligns it based on the keypoint’s orientation and scale, and then applies the python $pywt.dwt2$ for wavelet transform to produce a 64 dimensions descriptor.
## Feature matching (match_keypoints)
We do the feature matching using $knn$ and $BF2$ and we also do the double direction ratio test to improve robustness,also we implement $Ransac$ to remove outlier in the same function.
<p align="center">
  <img src="https://hackmd.io/_uploads/rJ-H05Zlle.jpg" width="100%" />
  <br />
  <span style="text-align: center; display: block;">feature matching disable Ransac</span>
</p>

<p align="center">
  <img src="https://hackmd.io/_uploads/BJc209Zxlx.jpg" width="100%" />
  <br />
  <span style="text-align: center; display: block;">feature matching enable Ransac</span>
</p>
<div style="clear: both;"></div>



## Blending


### 1.**Linear Blending**

- **Idea:** In the overlapping areas, gradually interpolate pixel values based on distance to each image's edge. 
- **Simple and fast**, but can cause visible seams if exposure differs.
![linear](https://hackmd.io/_uploads/Hkd0Hqkgex.jpg)

---

### 2. **Multi-Band Blending** (Pyramid Blending)

- **Idea:** Blend images at multiple frequency bands (low-frequency, high-frequency) separately. 
- First, build **Gaussian pyramids** of images.
- Blend **each level** of the pyramid.
- Reconstruct the final image from the blended pyramid.
![multiblending](https://hackmd.io/_uploads/HyvkI5ygeg.jpg)



## End to End Alignment

From the images, it's clear that without using end-to-end alignment, there is a noticeable vertical misalignment from top to bottom. By enabling end-to-end alignment, the errors are distributed across each homography transformation, eliminating the shift in the final stitched image.

End to end alignment reduces vertical drift by computing the y-offset between the last and first image and evenly distributing it using SVD decomposition across all homographies.


<p align="center">
  <img src="https://hackmd.io/_uploads/HJsqL6kxle.jpg" width="100%" />
  <br />
  <span style="text-align: center; display: block;">Panorama using end to end alignment</span>
</p>

<p align="center">
  <img src="https://hackmd.io/_uploads/B1HIDpJlxg.jpg" width="100%" />
  <br />
  <span style="text-align: center; display: block;">Panoram without end to end alignment</span>
</p>

## Result

After adjusting the parameter of Multi-Band Blending and turning both Ransac and End to End alignment ON we apply the two descriptor

The result shows our matching and blending algorithm works well
<p align="center">
  <img src="https://hackmd.io/_uploads/BJKM_pyxex.jpg" width="100%" />
  <br />
  <span style="text-align: center; display: block;">Final panorama result with MSOP descriptor applied</span>
</p>

<p align="center">
  <img src="https://hackmd.io/_uploads/SJdtOTkgel.jpg" width="100%" />
  <br />
  <span style="text-align: center; display: block;">Final panorama result with SIFT descriptor applied</span>
</p>

<p align="center">
  <img src="https://hackmd.io/_uploads/SJmLtJglxg.jpg" width="100%" />
  <br />
    <span style="text-align: center; display: block;">simply apply our blending on our photo </span>
</p>


<p align="center">
  <img src="https://hackmd.io/_uploads/H1XiJC1lgx.jpg" width="100%" />
  <br />
  <span style="text-align: center; display: block;">Our best result</span>
</p>


## Summary

In our final results, we actually chose not to apply blending, because although it worked perfectly on the example photos, when we applied it directly to our own images it produced a full-image feathering effect that looked terrible. We believe this happened because the overlap between our shots was simply too large. Laplacian-pyramid blending works by locally weighting and recombining different frequency bands, but when the mask loses its edge-falloff behavior—becoming almost constant at every scale—each band gets blended uniformly across the entire image. As a result, high-frequency details (edges and textures) are smoothed out, and low-frequency components (brightness and tone) are further flattened, making the whole panorama look as if it’s been uniformly “feathered” rather than just along the seams.

Another issue we encountered is exposure inconsistency. We used iPhone13 to take the photos, but overlooked the fact that it automatically adjusts exposure settings such as shutter speed and ISO. As a result, each image has different exposure levels, which caused visible seams and abrupt transitions in brightness after stitching.

We could climb the mountain again to retake the photos, but for two nerds like us, that would be a life-threatening mission—so we had no choice but to live with the regret. But At the same time, this mistake helped us gain a deeper understanding of the stitching process and its underlying principles through our discussions.



