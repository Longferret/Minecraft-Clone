# Minecraft-Clone v03
As part of my master thesis in the university of Liège, I created a Minecraft Clone using recent Tools such as 
the rust programming language and the Vulkan API.

I continued the developpement of the minecraft Clone after the presentation of my master thesis as a personnal project.

To run the application enter in the Minecraftv0.3 folder and enter in terminal:
```
cargo run
```
To clean:
```
cargo clean
```

### Controls
* Use zqsd to move around the world
* Space to jump
* Move the mouse to look around
* Press escape to quit the game
* Press left mouse click to delete a block
* Press right mouse click to add a block
* Press m to change mode (spectator or normal)

### Things to explore:
- You can remark that when you quit the game using the escape key and launch the app again, the modifications are saved.
- You can change the speed of the player in the `main_app`.
- You can play with the constants in the crate `game_core` in `lib.rs`but you should delete the folder 'chunk_data' to force the chunks to regenerate
  - `const CHUNK_SIZE: usize = 16;`
  - `const PERLIN_FACTOR: f64 = 10.0;`
  - `const PERLIN_MAX_HEIGHT: f64 = 10.0;`
  - `const RENDER_DISTANCE: i64 = 6;`
  - `const INTERACTION_RADIUS: f32 = 4.0;`
- You can delete the folder chunk data to regenerate the world

### Things to keep in mind:

The number of maximal vertices is fixed and the number of vertices per chunk could be further optimized so if you 
* increase too much the render distance 
* set a very low perlin factor
* set a very high perlin height

The rendering system could run out of space and do not save all the vertices, you will be left with transparent areas. You can increase `const MAX_VERTICES:usize = 500000;` in the `main_app` crate is your pc can support it.

It can also kill performance.

If your PC lags, reduce the number of max vertices in the `main_app` crate, the variable is `const MAX_VERTICES:usize = 500000;`

# Organisation
The application is organized in 4 differents crates.

1. game_core crate, which implements the logic and the chunk loading/unloading
2. renderer crate, which implements the rendering system
3. draw_element crate, which define a structure to exchange informations on how to draw
4. main_app crate, which coordone the renderer and the game_core

You can here find documentation on each crate:
- [main_app](https://github.com/Longferret/Minecraft-Clone/blob/main/docs/main.md)
- the renderer crate not documented in this version, you can find information on it directly in the code or by following the steps of the rendering system [here](https://github.com/Longferret/Minecraft-Clone/blob/main/Vulkan-Intro/readme.md) and also read the changes in version 0.1 [here](https://github.com/Longferret/Minecraft-Clone/blob/main/Minecraft-v0.1/readme.md).
- [draw_element](https://github.com/Longferret/Minecraft-Clone/blob/main/Minecraft-v0.3/docs/draw_element/draw_element.md)
- [game_core](https://github.com/Longferret/Minecraft-Clone/blob/main/Minecraft-v0.3/docs/game_core/game_core.md)
  - [world](https://github.com/Longferret/Minecraft-Clone/blob/main/Minecraft-v0.3/docs/game_core/world.md)
  - [chunk](https://github.com/Longferret/Minecraft-Clone/blob/main/Minecraft-v0.3/docs/game_core/chunk.md)

You can also find usefull information in my [report](https://github.com/Longferret/Minecraft-Clone/blob/main/report.pdf)
. It explains the main ideas behind the minecraft Clone, a performance test and the future work that could be done.

To learn Vulkan, I explained the all the steps of the rendering system developpement [here](https://github.com/Longferret/Minecraft-Clone/blob/main/Vulkan-Intro/readme.md).



# Improvements
These improvement are the improvement made after the presentation of my master thesis and solve some problem explained in my [report](https://github.com/Longferret/Minecraft-Clone/blob/main/report.pdf).

## Buffer all
Merge of the wait and exec GPU function for a better visibility and scalability. Which means the view and position are now buffered

## Relative coordinates
Changes in the rendering system to make relative calculation instead of absolute one to avoid floating point errors.
More specifically what I did 
* Add chunk position to vertex data
* Add chunk position of player to uniform data
* Changes in the vertex shader

## Little fixes and additions
* The m key change the mode (2 modes 1 gravity, the other free, no collision)
* Fixed little moves when high FPS
* Fixed player from jumping when in the air (by adding check if on ground at every iteration)

## Vertex buffer per chunk (in progress)
Creation of a new API, Load chunk, Delete Chunk --> I might want the rendere to be directly linked to the world of the game_core crate
* Load chunk will lauch a thread to create a new vertec buffer with all surface elements, then add it to hashap of chunks
* Delete chunk will just remove the vertex buffer from the hashmap

New structuration:
* Addition of Hashmap (chunk_coords) -> vertex buffer + free indexes + quad_to_index
* Change the quad addition removal to work inside the hashmap
* Creation of a secondary command buffer that contains everything but the verrtex buffer binding, the primary will just bind the vertex buffer and build (change from multiple to one-submit)

At each iteration:
* Transform the hashmap into a vec of vertices (0.1 ms).
* Build a one submit primary command buffer with these vertex (0.5ms) to verify
* Submit it

After:
* Add new APIs to set the visibility of chunks



## Things to consider
* The vertex buffer should have to possibility to grow in size 