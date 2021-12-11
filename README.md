# robust-nerf
A robust Neural Radiance Fields Implementation for 3D Reconstruction

# Reading List

There is no need to read all papers in an in-depth manner except when its time to implement specific methods presented inside. Note that this list is in no way exhaustive. You can easily check for new publications in the field through the citations of the original paper on [google scholar](https://scholar.google.de/scholar?hl=de&as_sdt=2005&sciodt=0,5&cites=9378169911033868166&scipsc=&q=&scisbd=1). Sort by date instead of importance.


#### The one that started it all
- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf), Mildenhall et al., ECCV 2020 | [github](https://github.com/bmild/nerf) 

The original implementation. If you do not have a lot of time, i would suggest you start here. The most important take-aways in my opinion are the used Encoding Function and the Rendering Equations.

#### Faster Training
- [Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction](https://arxiv.org/abs/2111.11215), Sun et al., Arxiv | [github](https://github.com/sunset1995/DirectVoxGO)

This paper is to my knowledge currently the fastest implementation of Neural Radiance Fields. It takes around 10 mins to train on a 2080 ti, on my own gtx970, its around half an hour. Because the space representation is build upon Voxels, not all enhancements are directly transferable. 

See their teaser here:

https://user-images.githubusercontent.com/2712505/142961346-82cd84f5-d46e-4cfc-bce5-2bbb78f16272.mp4

#### Pose Reconstruction / Optimization
- [BARF: Bundle-Adjusting Neural Radiance Fields](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/), Lin et al., ICCV 2021 
- [Self-Calibrating Neural Radiance Fields](https://postech-cvlab.github.io/SCNeRF/), Park et al., ICCV 2021 | [github](https://github.com/POSTECH-CVLab/SCNeRF) 
- [NeRF--: Neural Radiance Fields Without Known Camera Parameters](http://nerfmm.active.vision/), Wang et al., Arxiv 2021 | [github](https://github.com/ActiveVisionLab/nerfmm)
- [GNeRF: GAN-based Neural Radiance Field without Posed Camera](https://arxiv.org/abs/2103.15606), Meng et al., Arxiv 2021 

Note that some of these methods are only considering the frontal facing case as presented in the LLFF-Nerf Dataset, as seen in the examples on the [NeRF-- website](http://nerfmm.active.vision/).

#### Camera Intrinsic Optimization
- [Self-Calibrating Neural Radiance Fields](https://postech-cvlab.github.io/SCNeRF/), Park et al., ICCV 2021 | [github](https://github.com/POSTECH-CVLab/SCNeRF)

This paper is also listed under Pose Reconstruction, since it does both.

#### Exposure Compensation
- [ADOP: Approximate Differentiable One-Pixel Point Rendering](https://arxiv.org/abs/2110.06635), Rückert et al., Arxiv 2021

The most important takeaway is the differentiable tonemapper, the rest, while interesting, works differently than NeRF

#### Different Image Sizes/Multiscale Images/Anti Aliasing
- [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://jonbarron.info/mipnerf/), Barron et al., Arxiv 2021 | [github](https://github.com/google/mipnerf) | [bibtex](./citations/mip-nerf.txt)

This work introduces some really interesting tricks to compensate for different images with varying distances to the to-be reconstructed object. 

#### Unbounded Scenes
- [Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields](https://jonbarron.info/mipnerf360/), Barron et al., Arxiv 2021

Unbounded scenes are scenes where it is unclear where the scene ends. The easiest solution is to introduce a spherical background, which is the 'default' color in case a ray does not hit anything.


#### Deblurring
- [Deblur-NeRF: Neural Radiance Fields from Blurry Images](https://arxiv.org/abs/2111.14292) Ma et al., Arxiv 2021

While the proposed method is quite useful, it is currently unclear how transferable this is for our work.


#### Bad Lighting Conditions
- [NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images](https://bmild.github.io/rawnerf/), Mildenhall et al., Arxiv 2021

#### General Overviews
- [Awesome NeRF](https://github.com/yenchenlin/awesome-NeRF) by yenchenlin
- [Neural Fields in Visual Computing and Beyond](https://neuralfields.cs.brown.edu/) Xie et al., Arxiv 2021


