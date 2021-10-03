demo results:

original demo only had 1 exterior building
so tested the pretrained model on a texture and interior

input images:

![](demos/demo_building/demo.jpg)
![](demos/demo_building/demo_mask.jpg)

output images:

![](demos/demo_building/output/albedo.jpg)
![](demos/demo_building/output/nm_pred.jpg)
![](demos/demo_building/output/shading.jpg)
![](demos/demo_building/output/lighting.jpg)

input images:

![](demos/demo_interior/input/window.jpg)
![](demos/demo_interior/input/window_mask.jpg)

output images:

![](demos/demo_interior/output/albedo.jpg)
![](demos/demo_interior/output/nm_pred.jpg)
![](demos/demo_interior/output/shading.jpg)
![](demos/demo_interior/output/lighting.png)

input images:

![](demos/demo_texture/demo2.jpg)

output images:

![](demos/demo_texture/albedo.jpg)
![](demos/demo_texture/lighting.png)
![](demos/demo_texture/nm_pred.jpg)
![](demos/demo_texture/shading.jpg)


demo output is saved as jpg to reduce filesize of repo. actual output is PNG and includes a PNG copy of the input image.
input can be jpg.

results are quite organic and would need some work to use in games, but can form a good base for texture creation.

results with the pretrained model are better at a low inputsize (default 200) but sharper at a higher input size (same resolution as input texture)
