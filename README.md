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
![image](https://github.com/user-attachments/assets/c112f982-3be8-45be-b9c2-6a451f255699)


depth mask of said model: ![image](https://github.com/user-attachments/assets/cf77ef04-c988-40ea-a63d-1255bd15e275)


