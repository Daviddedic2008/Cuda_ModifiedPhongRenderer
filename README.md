Renderer meant to be fast and realtime
Uses Phong for light intensity calcs
working on dynamically adjustable BVH rn(hopefully almost done)
smooth/realistic shadows that utilize gaussian blur(intensity dependent on distance from object casting shadow)
colors are planned to be lerped with colors calculated by a low-res path tracer

rasterizer benchmark #1: rasterizes 1,000,000 tris per second on a 4050 mobile(pretty bad!)
