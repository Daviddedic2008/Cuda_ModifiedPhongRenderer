Renderer meant to be fast and realtime
Uses Phong for light intensity calcs
working on dynamically adjustable BVH rn(hopefully almost done)
smooth/realistic shadows that utilize gaussian blur(intensity dependent on distance from object casting shadow)
colors are planned to be lerped with colors calculated by a low-res path tracer

rasterizer benchmark #1: rasterizes 1,000,000 triangles per second on a 4050 mobile(pretty bad!)

first "fragment" shader written in my renderer:
![image](https://github.com/user-attachments/assets/4faa754d-d838-40df-b5b3-5c6bad525ff2)

