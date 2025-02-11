# Minecraft Clone Version 0.1
This is the first version of the minecraft clone.

I definined the crate "renderer" containing my basic rendering system.

I will define the crate "physics" containing my basic game engine.

To run the application enter in the Minecraftv0.1 folder and enter in terminal:
```
cargo run
```
* Use zqsd to move around the world
* Space to jump
* Use the mouse to look around
* Press escape to quit the game
* Press left mouse click to delete a block

## The rendering system
The rendering system is based on my last project of the the Vulkan-Intro.

With the little change that now block 0 is not [-0.5,0.5] but [0,1].

### Improvements
* Addition of transparency by adding a new render pass + pipeline, transparent block have a dedicated part of the vertex buffer (last third).
* Addition of a cursor by adding a new render pass + pipeline
* Change the camera to origin centric to avoid floating point errors (player is always in 0,0,0 and the world move around him).
* Change to the depth buffer format from D16_UNORM to D32_SFLOAT to avoid long distance flickening.

### Charasteristics

Main components:
* X images in the swapchain, X=2 in most device (the minimum).
* X frambuffers
* X command buffers to send execution instruction to GPU.
* X vertex buffers to store vertex data (opaque and transparent).
* X interface buffers to store vertex data (interface elements).
* X uniform buffers to store MVP matrix (for 3D perspective).
* 1 queue for everything (texture upload, drawing).
* 3 pipelines for opaque blocks, transparent blocks and interface (cursor).
* 1 render pass for 3 sub passes (same as 3 pipelines).
* 2 Shaders for all blocks.
* 2 Shaders for all interface (only cursor atm).
* 1 descriptor set for all blocks
* 1 descriptor set for interfaces (only cursor atm).

Feature:
* MSAA x4
* Anisotropy maximum of the device
* Face culling
* Depth testing+writing for opaque blocks
* Depth testing+no writing for transparent blocks
* Face culling
* GPU execution is asynchronous

Runtime buffer modifications:

All buffer modifications like the MVP matrix or the vertices of the blocks must be done on the X buffers defined. To do so the rendering system keeps tracks off all the actions to execute on each image of the swapchain (often 2) and execute them (modify the vertices, the MVP matrix or update the dynamic blocks).


### API
* add_square, add a square defined by a Square structure.
* remove_square, remove a square defined by a Square structure.
* set_view_position, actualize the MVP matrix from the player position and its angle of view. This funciton must be called at every loop cycle, it does not use the action to execute of the rendering system. 

### Bugs
* Lava moving texture make a line appear between 2 or more of the same block.
* Static far distance rendering is ugly, I might need a post processing shader to fix this.
* Transparent blocks (water) are not sorted, some visual might not be on point. 
* Cursor is distorded when the windows is resized due to the use of the projection matrix.

### Possible Improvement 
* Proper water shaders with distinct API
* Addition of lighting shaders.
* Possibility to toggle anisotropy/MSAA/fullscreen
* Dynamic viewport implementation
* Interface implementation (menu, FPS counter)

## The game engine

This part is in developpement.

For the moment im experiencing the basics like the collisions and the 
cursor that points to blocks to destroy/add blocks.

The world is 30x30 blocks and uses perlin noise to be generated.

For the moment the player spawn back in the middle when crossing boundaries

### The collision system
In my game we will only work with hitboxes that are cuboids and axis aligned (not rotation of hitboxes).
AABB(Axis-Aligned Bounding Box) for short.

#### 1. The base
The first thing that comes in mind when creating a collision system is simply verying if the boxes overlaps to detect a collision.

It works well to detect collision but it has 2 main problems.

The first is: How do we resolve the collision ?

We could take the smallest overlap and teleport the player to the obstacle AABB boundary in that direction. But this has major flaws, for movement in 3D we could for example fall of a cliff because the collision system detected that the X axis had a smaller overlap than the Y axis.

Having a good collision resolution is very hard when we detect collision as they come, we must go back in time and calculate precisely from which face the player entered the obstacle.

The second problem is the tunnelling effect.

If the player goes fast enough, it could go through an thin obstacle, that is because the update we do are not continuous, so if a pc lags a bit or the player goes to fast, we may not even detect any collision.

#### 2. Fixing tunneling

There are 2 main ways to fix tunneling.

The first is dicretize even more.
If for a frame the player moved a large distance, we will compute mulitple small distance for 1 update.

With this strategy, we must specify a constant of max distance or max time that 1 update of motion can do. It also means that we have a treshhold of speed and/or of object width to avoid tunneling. This strategy can become costly if we want good accuracy for high speed object.


The second strategy is to detect the collisions before they happen. To do so we can use for example raycasting.
We will cast a ray and the first obstacle that the ray encounters makes a collision.

This strategy do not need multiple motion updates pass but can be costly as we need all obstacle to be sorted by distance. It has the advantage of not having any threshhold.

#### 3. Raycasting to Facecasting
I wanted a strong collision system to not have problems later, so I have chosen to go with raycasting.

The only left problem is that "raycasting" works well for points but not for AABB.

AABB is composed of 8 points, even if we apply ray casting on the 8 points, we could go through small obstacles as long as the "rays" of the AABB vertex do not collide with the obstacle.

To fix this I thought of a new system that I call "Facecasting".

Instead of casting a ray, we will cast a surface, more specifically we will cast the 3 surfaces of the player direction.

The direction of the player is simply the speed direction along each axis.

To cast a surface, we will create a polygon from that face using the speed vector.
If any obstacle is in that polygon, a collision occured.

This system has the advantage to know exaclty wich face of the player AABB collided with the obstacle.

Here is an example of the detailled steps:
* We assume a random speed along y,z axis and a positive speed (going right) along the x axis.
1. Calculate the the initial x (of the face) and the final x position thanks to the speed.
2. Get around obstacles and sort them in ascending order thanks to their left face x.
3. For each obstacle verify that their left face is insinde [initial x, final x]

(a) if obstacle left face not inside continue loop 3.

(b) if obstacle left face is inside, go to 4.

4. Using the left face x of the obstacle compute the slice of the polygon ( formed by expanding x face of player)

If the slice and the left face of obstacle overlaps, a collision occured and we can return

If the slice and the left face of obstacle do not overlap, continue loop 3.


By doing this for the 3 faces of the player, we can get the first collision of each axys (if there is one) and choose to resolve the one that is the closest to us. 

This strategy is very good as it completely solve tunneling and the collision resolution is straigh forward.

The downside is that it is a bit complicated.

#### 4. Performances measurements & Potential improvements
The time it takes on my computer for the collision testing and resolution with 4 obstacles around is at most 20 micro seconds.

For a game to be fluid we can aim for 140 FPS, which means each frame must be computed in less than 7ms. Meaning that my collision system can handle around (7000 us/20us * 4 obstacles) 1400 collisions tests at each frame.

For the moment I use all the obstacles around the player to calculate the collision.

An improvement could be:
* Select the obstacles only in the tested direction
* Pre-sort the obstacles 

### Cursor pointing
Done with ray tracing.


### Bugs
* Problem with mouse key buffering
* Collision system not optimized

### Next steps
1. Define minecraft coordinate to array coordinate (array can't go in negative)
2. Define a field to store actual visible square to be able to add/remove them
3. Optimize the redering by implementing the Maximum size rectangle binary sub-matrix with all 1s (to have less vertex)
4. Define how to store/load chunk data
5. Split the physics /  chunks loading / collision detection in different file and/or thread. 
6. Asynchronous physics/renderer from main loop
