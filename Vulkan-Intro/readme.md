# Presentation
I will write here information in the Vulkan that I find usefull to know.

The tutorial that was given to me : https://kylemayes.github.io/vulkanalia/
is about vulkania, which is a wrapper for the vulkan API but it is not the one I will use in my project for several reasons.

There are multiple possible crates to work with vulkan in Rust which are:

* Ash, a thin wrapper around the vuklan API
* Boson, fairly new, the same as Ash but with some abstraction
* Vulkania, fairly new, try to balance between low level API and good abstraction to avoid verbose Vuklan commands
* Vulkano, Rust-like abstraction of the Vuklan APIs, we still have low level controls but a lot is abstracted to focus on the essential.
It also implements the Rust safety guarentees.


As beginner in both Rust and Vulkan, I have chosen to begin with Vulkano, with a higher level of abstraction and the safety guarantees.

I may switch or Use another crate if I need it in my project.

For now i'm following the tutorial on: https://vulkano.rs/.

And after i will continue with the tutorial at https://taidaesal.github.io/vulkano_tutorial/. It uses on older d of vulkano but it goes very far explaining the features of vulkan.

# Rendering Basics
You can find a quick explanation on all the components of a rendering system based on the Vulkano triangle tutatial in this [document](https://github.com/Longferret/Minecraft-Clone/blob/main/report.pdf), Section 5.2

# Project 1: Triangle and input
```
cargo run -p p_1 // to run the project
```

The code of this project is fully documented to better understand the component of vulkano, [see there](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Vulkan-Intro/1.%20triangle_inputs/src/main.rs).

Next projects will take the code of the previous one as a base, only the added part will be documented.

## Objective 
My objective for this project is to modify the code of the tutorial at <http://vulkano.rs>.

The tutorial implements a rendering system to draw a simple static triangle to the screen.
A copy of their code is also available in this repository [here](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Vulkan-Intro/0.%20triangle/src/main.rs)

My goal is to modify the vertices of the triangle at runtime.

## Description
This code displays a triangle on screen and accept zqsd key inputs to move the  triangle up/left/down/right.

## Implementation
I achieved this by creating as many command buffers as there are images in the swapchain.

Of course each command buffer has its own different vertex buffer.

This strategy allowed me to write in a vertex buffer while the other one is used by the GPU.

By adding some conditions on keys in the main loop and writing the changes in the vertex buffer, I have a moving triangle !

More detailled explaination are in the [code](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Vulkan-Intro/1.%20triangle_inputs/src/main.rs).
 

# Project 2: Square and perspective
```
cargo run -p p_2 // to run the project
```

## Objective 
My goal is to draw one or multiple cubes to the screen and create a first-person perspective.

## Description
This program displays multiple cubes.

It is possible to move around with zqsd-space-shift and move the camera with the mouse.

## Implementation
My code is inspired by [this tutorial](https://github.com/taidaesal/vulkano_tutorial/tree/master).

Here is a brief description on how I implemented those features,
more detailled explaination are in the [code](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/tree/main/Vulkan-Intro/2.%20square/src).

### 1. The InputHandler
I added a new module (inputs.rs) that stores all the information of the key pressed and mouse moves.

It also update the angles of view and the position of the player thank the current state of the structure.

It calculate the distance moved by the player not by frame but by unit of time (real speed).

### 2. The MVP matrix
This is structure I created to transform the 3D game coordinates into the 2D screen coordinates.

I needed to create X uniform buffers (X = nbr of image in the swapchain). A uniform buffer is used to give data to the GPU that will be used by all shaders.

I created multiple buffers to be able to write in one while the other is used by the GPU.

After creating the uniform buffer, I defined a descriptor set for each.

Then I bind each descriptor set to each commandbuffer, and finally use the new variable in the vertex shader to calculate the coordinates.

### 3. Depth Testing
I needed to enable depth testing because the triangles overlap and are not rendered in the right order.

I achieved this by:
1. Creating a depth image (that will be used by the GPU) for each framebuffer.
2. Adding a new attachment to the render pass and pass it to "depth_stencil".
3. Specifying the clear value of depth in the command buffers.
4. Activate the simple depth testing option in the pipeline.

There are 2 features in Vulkan depth testing.

The first that we use is to render all triangles in the right order.

The second that we don't use is to discard all triangles that are greater than a certain depth. That's why we provide a clear value = 1. It means that no triangle will be discarded.


# Project 3: Textures and Simple culling
```
cargo run -p p_3 // to run the project
```

## Objective 
My goal is to add textures to the cube from the last project and try to do some culling.

## Description
This program displays multiple cubes with a stone texture.

It is possible to move around with zqsd-space-shift and move the camera with the mouse.

## Implementation
My code is inspired by [this tutorial](https://github.com/taidaesal/vulkano_tutorial/tree/master).

Here is brief description on how I implemented those features,
more detailled explaination are in the [code](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/tree/main/Vulkan-Intro/3.%20textures/src).

### 1. Culling
I implemented face culling, which means the triangle not facing us will not be displayed.

In vulkan a triangle facing us is defined as a triangle that is created counter-clockwise.

I enabled it by adding the parameter "cull back" in the pipeline for the rasterization state.

### 2. Textures
In order to apply textures on the triangles rendered we first need to create an GPU image and upload data to it.
It can be done in 4 steps:
1. Create an image only visible by the GPU for a fast access.
2. Create a sampler to be able to scale/transform image to the triangles
3. Create a staging buffer, a buffer for fast CPU acces and used to upload data to the GPU.
4. Create a one submit command buffer to upload the data to the GPU image.

Then we need to add a binding to our descriptors sets (we could also create new descriptor sets) and also change the layout of the pipeline.

After we need to update the vertex structure to add a field specifying the coordinate of the texture in each vertex.
There are called uv coordinates [0,1]x[0,1].

And finally we can use the texture in the fragment shader.

# Project 4. Refactoring
```
cargo run -p p_4 // to run the project
```

## Objective 
My goal is to add the possibility to add any textures and refactor the code to be able to add new functionnalities easily.


## Description
This program displays multiple cubes with different textures.

The code is cleaner and I defined basic functions to draw/remove a square of screen.

It is possible to move around with zqsd-space-shift and move the camera with the mouse.

## Implementation

### 1. Textures
With little change to the image/imageView, I managed to import multiple texture in a GPU buffer.
I did it by defining Image array and using the same sampler for all images.

Then I had to chage a bit the shaders to access an array instead of a pixel color.

After I had to add a variable "block type" to the vertices in order for the shader to pick the good texture to display.

These textures now are present in my rendering system:
* Dirt Block
* Stone
* Cobblestone
* Wooden Planks
* Gravel
* Sand

Discussion: It is better to import an array of image because the descriptors (often 4) sets and the bindings (often 16) are limited.
In the other hand an Image can have 2048 layers.

### 2. Interface
Since I want to build a mincraft clone, I need functions to simply add/remove cubes to/from the rendering system.

I have chosen to render squares, not cube to be able to render only the visible surfaces

I decided to have all my vertices in the same buffer and its size is decided at compile time (not safe nor efficient).

I have 3 new variables:
* free_indexes, a vector of the free index for a vertice (they are by group of 6 because 1 square = 6 vertices).
* square_to_index,  a hashmap with the square charasteritiscs as key and its index on the vertxbuffer as value.
* action_per_image, a vector of vector of actions to keep track of all action the rendering have still to do an each image.

More on action_per_image: since I have 2 or more vertex buffers for multi-buffering, I have to update all vertex buffers. That's why I need this variable that keep a vector of action to do on each image. When the action had been done the vector for that image is cleared.

It ensures that all the images have the same modification.

To add a square:
1. The other program specify the characteristics of the square to draw.
2. Verification that the square is not already in the hashmap (return if it is the case)
3. Get the next free index (pop element)
4. Add the square to the hashmap with the free index
5. Notify that we want to draw a square at this place to the rendering system

To remove a square:
1. The other program specify the characteristics of the square to remove.
2. Verification that the square is in the hashmap (return if it is not the case)
3. Get the index of the square in the vertex buffer
4. Remove the square of the hashmap
5. Notify the rendering sytem to delete this cube (set everything at 0)


This solution is not the best, the whole buffer goes into the shaders with 0s data which is inneficient, but the vertexbuffer stays in memory and we don't reallocate a buffer at each addition/removal.

### 3. Projection when resizing
Using the Action_per_image variable, I defined a change of the projection (in the MVP matrix) to apply on all image when the window is resized.

### 4. Performance
| Vertex Count  | Frame Rate (FPS) |
|---------------|------------------|
| 8,000         | 2300 FPS         |
| 120,000       | 2000 FPS         |
| 2,000,000     | 2000 FPS         |
| 2,800,000     | 2000 FPS         |
| 3,500,000     | 250 FPS          |
| 4,000,000     | 200 FPS          |
| 6,000,000     | 140 FPS          |
| 10,000,000    | 50 FPS           |

# Project 5. Animations and features
```
cargo run -p p_5 // to run the project
```

## Objective 
My goal is to provide the CPU the possibility to group multiple same square into one for drawing. This would offload by a lot the GPU.

I also want to test MSAA and anisotropy.

This project will serve as a base rendering system for my minecraft clone.

## Description
This program displays multiple cubes with different textures.

The Interface now gives a way to draw multiple cube at once.

It is possible to move around with zqsd-space-shift and move the camera with the mouse.

## Implementation
Note: blending variables are present but transparency is not implemented in the rendering system.

### 1. A better Interface
The program that calls the function must specify 2 new parameters, extend 1 and 2.

Extends is used to add a cube on the first or second dimension (along x,y,z).

Example:

For a cube with LEFT parameter, extend1 = 1 will add 1 square at y+1 (above). extend2=10 will add 10 squares at z+1 to z+10. (with LEFT x coordinate is fixed).

It is highly recommended to use as much as possible the extends of the cubes as it reduces immensly the number of computed vertices.

### 2. Multi-Sampling
The renderer now uses Multisample Anti-Aliasing (MSAA) x4, to smooth jagged edges and reduce visual artifacts.

I implemented this by:
1. Add the TRANSFER_DST property to the image of the swapchain (as they now have to receive data from the sampled images)
2. Modify the MultisampleState of the pipeline to 4 samples.
3. In the frame buffer, create a new image with 4 samples and set the depth image to also 4 samples
4. Change the render pass to indicate that we will use a 4 sample image and then transfer to the swapchain image.

<div style="display: inline-block; margin-right: 10px; text-align: center;">
    <p>Without Anti-Aliasing</p>
    <img src="Vulkan-Intro/report-images/AAbefore.PNG" width="500" />
</div>
<div style="display: inline-block; text-align: center;">
    <p>With Anti-Aliasing</p>
    <img src="Vulkan-Intro/report-images/AAafter.PNG" width="500" />
</div>

We can remark that the jagged effect is not here anymore, especially on the tower of sand.
## 3. Anisotropy
The renderer now uses Anisotropy to smoothen textures.

I simply added the anisotropy on the device at creation and fixed its value in the sampler to the max possible value.

<div style="display: inline-block; margin-right: 10px; text-align: center;">
    <p>Without Anisotropy</p>
    <img src="Vulkan-Intro/report-images/Anibefore.PNG" width="300" />

</div>
<div style="display: inline-block; text-align: center;">
    <p>With Anisotropy</p>
    <img src="Vulkan-Intro/report-images/Aniafter.PNG" width="300" />
</div>

We can remark that the floor is smoothen and has a better visual.

## 4. Animations
The texture definition and loading is modified and everything is now is the texture module.

Each block has its own charateristics, name, animation speed and number of textures.

The renderer now keep an hashmap of dynamic squares, there are squares that have an animation (water,lava,..).

Each 50ms, the renderer verify if the block must be updated (via block animation speed). And if so, it adds an action_per_image to execute the changes on the square texture.

## 5. Bug Fix

* Unknow key inputs are now ignored (before it made the application crash), unwrap that was not handled