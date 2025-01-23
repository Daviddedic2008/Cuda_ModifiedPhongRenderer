Renderer meant to be fast and realtime
Uses Phong for light intensity calcs
working on dynamically adjustable BVH rn(hopefully almost done)
smooth/realistic shadows that utilize gaussian blur(intensity dependent on distance from object casting shadow)
colors are planned to be lerped with colors calculated by a low-res path tracer

rasterizer benchmark #1: rasterizes 1,000,000 triangles per second on a 4050 mobile(pretty bad!)

first "fragment" shader written in my renderer:
![image](https://github.com/user-attachments/assets/4faa754d-d838-40df-b5b3-5c6bad525ff2)

depth masks done:
![image](https://github.com/user-attachments/assets/dd232935-21ca-4803-8ed4-dcde65fdfb8f)

first model rendered!!(low-poly knight figure from chess):
![image](https://github.com/user-attachments/assets/5a7d93d4-0252-4fa7-83a1-15a2170bf2ae)

depth mask of said model: ![image](https://github.com/user-attachments/assets/7269514f-2eed-4a95-8ce2-85590ff37d39)

