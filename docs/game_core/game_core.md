# Game Core Definition and Algorithms - [`lib.rs`](https://github.com/Longferret/Minecraft-Clone/blob/main/game_core/src/lib.rs)
This document contains all information on the game_logic implementation.

## The structure

The game_core structure contains:
* `inputhandle: InputHandle`, a structure to store inputs
* `view_angle_horizontal/vertical: f32`, the angles of view
* `mouse_sensitivity: f32`,
* `base_player_speed: f32`, the base speed of the splayer when jumping/moving
* `gravity_y: f32`,
* `time_last_update: Instant`,
* `world: World`, the world of the game
* `target_blocks: Vec<([i32;3],Face)>`, a list a block that intersect the player cursor with their face of intersection.
* `player: Entity`, a structure that contains the player characteristics
* `batch_draw_element: Vec<Vec<(bool,SurfaceElement)>>`, batches of `SurfaceElement`


The inputhandle structure is simply a place to store the inputs in an orginized manner.

The Entity structure contains:
* `chunk_position: [i64;2]`, the chunk where is the entity
* `relative_posititon: [f32;3]`, the relative position inside a chunk
* `bounding_box: [f32;3]`,
* `speeds: [f32;3]`,
* `on_ground: bool`, true is the entity is on the ground

Constants:
* `const FOLDER_CHUNK:&str = "chunk_data";`, the folder where chunk are saved.
* `const CHUNK_SIZE:usize = 16;`, the chunk size (X,Z)
* `const CHUNK_HEIGHT:usize = 512;`, the chunk hight (Y)
* `const PERLIN_FACTOR:f64 = 10.;`, a factor for chunk generation, higher value leads to smooth terrain.
* `const PERLIN_MAX_HEIGHT:f64 = 10.;`, maximal height where block will be generated.
* `const RENDER_DISTANCE:i64 = 4;`, the number of chunk loaded around the player.
* `const INTERACTION_RADIUS:f32 = 4.;`, the maximal length in blocks that the player can reach to interact.
* `const MAX_DRAW_BATCH:usize = 300;`, the maximal `SurfaceElement` output at each update call.

## Functions
This part talks about the algorithms and the logic of the functions defined inside the file `lib.rs`.

I don't explain all the functions here, only the ones that contain a special logic or algorithm.

### `pub fn update(&mut self) -> Vec<(bool,SurfaceElement)>`
This function update the state of the world based on the time elapsed and the inputs stored.

What is does:
1. Check if some chunks have finished loading/unloading
2. Update the view angles
3. Set the obstacle target in `target_blocks`
4. Handle the mouse button to destroy/add a block
5. Update the speed of the player 
6. Update the position of the player
7. Group by batch the `SurfaceElement` collected and return 1 batch

### `fn set_obstacle_target(&mut self)`
It uses a well known [ray tracing algorithm](http://www.cse.yorku.ca/~amana/research/grid.pdf) to get the blocks traversed from the view point to the `INTERACTION_RADIUS` in the direction of the view angles. I just modified the algorithm a little bit to be able to return the face traversed.

### `fn update_speed(&mut self,time_elapsed: f32)`
This function update the speed thanks to the input and normalize it to not have double speed when pressing 2 key at the same time.

### `fn update_position(&mut self,time_elapsed: f32)`
This function update the position thanks to the speeds of the player.

What is does in order:
#### 1. Get the around obstacles
The obstacles are gathered by calculating the next position of the player and all the obstacles inside the blocks that are in between the next position and the inital position are saved.

This is not a very efficient approach because we check blocks from the initial position to the next position instead of the blocks that the player actually traversed.

This is not a really big problem because the speed of the player is low so is the distance traveled. Meaning that the number of blocks to check stays low.

#### 2. Check for collision
I developped special algorithm to check for collision that can calculate the exact position of impact of a moving AABB (axis aligned bounding boxes) to other static AABBs.

I call it  *The Face Casting Algorithm*, it is explained below.

#### 3. Check if the player is below Y=0 and load/unload chunk if the player changed chunk
For now it replace the player in a safe spot.


## The Face Casting Algorithm
In my game, I only work with AABBs (axis aligned bounding boxes).

### A simple collision system
The first things that comes to mind when creating a collision system is to simply check is the two AABB overlaps.

It works but have 2 major flaws:

The first is that the collision resolution is hard, we don't know which face has collided and approximation like "the smaller overlap is the face that collided" is often false.

The second is that the collision system do not avoid tunneling. If the player goes fast enough and the obstacle is thin, the initial player position (before update) can be valid (no collision detected) and the final position of the player (after update) can also be valid (no collision detected) but a thin obstacle could be in between those 2 positions. 


I wanted my collision system to be robust. So I tried to build an algorithm that calculate the exact position of the collision and avoid tunneling.

### The algorithm
The idea is to pick the 3 faces in the direction of the speed vector and calculate where the 3 faces will be at the next frame.

I call them the initial face in X/Y/Z and the final face in X/Y/Z

I then create a polygon for each intial-final face pair and check if that polygon overlaps whith an obstacle.

#### Detect a collision between a polygon and an obstacle(AABB):
First sort the obstacles per distance of their face corresponding to the opposite face of the player.

For example, let's say the starting face of the player is NORTH, we will sort the obstacles by their SOUTH face distance from the NORTH initial face.

Then for each obstacles:

* check if the component of the obstacle face is in the bound of the initial/final face. 

The SOUTH/NORTH component is Z in my game. (EAST/WEST = X, TOP/BOTTOM = Y).

* If the obstacle face component is not in bound, take the next one.

* Check if the obstacle face overlaps with the inital face "casted"

Casting the initial face means that we move along the speed vector to arrive in the obstacle face component, for a Z component, we check overlapping in the XY plane.

* If the obstacle face do not overlaps, take the next one

* In the other case a collision is detected and the distance from the initial face to the collision and position of the collision are retuned.

* In case no collisions are detected, nothing is returned.

We get at the end 3 or less (distances,position) pairs.

We take the lowest distance, resolve the collision thanks to the position and make another "facecasting" until no more collision are detected.

### Advantages and Drawbacks
Advantages:
* Exact postion and face of collision
* Tunneling proof
* Can be improved by presorting the obstacles and reduce the number obstacles (by selecting only relevant ones).

Drawback:
* Can be computationaly intensive (make several collision detection on 3 polygons)
* A bit complicated to understand

### Comparaison with existing algorithms
I didn't base algorithm on any well known ones.

The idea if the same as [Swept AABB Collision Detection and Response](https://www.gamedev.net/tutorials/programming/general-and-gameplay-programming/swept-aabb-collision-detection-and-response-r3084/), but for 3D AABB.

An other algorithm to detect collisions is the separeting axis theorem, but it only detects collisions, it does not give any information on where the collision occured.



