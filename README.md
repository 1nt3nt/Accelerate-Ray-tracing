# Accelerate-Ray-tracing

## Background ##
This CUDA accelerate ray tracing program is based on a OpenGL assignment.[OpenGL assignment](https://github.com/1nt3nt/Computer-Graphics/tree/main/asn8).

## Basic Idea ##
1. Calculating every single pixel color by GPU is my basic idea. To do that, I send an array, carrying all the frame, to CUDA kernel, then generate all ray infomation, such as position, direction and initialized color.
2. Using another kernel to calculate shadow and specific pixel color then save it back to a pointer array.
3. Back to the host, travel through that pointer array to render all pixels'color.
