# Minecraft-Clone v03
This is version 03 of my minecraft clone.

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
- [main_app](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Minecraft-v0.3/docs/main.md)
- the renderer crate not documented in this version, you can find information on it directly in the code or by following the steps of the rendering system [here](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Vulkan-Intro/readme.md) and also read the changes in version 0.1 [here](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Minecraft-v0.1/readme.md).
- [draw_element](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Minecraft-v0.3/docs/draw_element/draw_element.md)
- [game_core](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Minecraft-v0.3/docs/game_core/game_core.md)
  - [world](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Minecraft-v0.3/docs/game_core/world.md)
  - [chunk](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Minecraft-v0.3/docs/game_core/chunk.md)

You can also find usefull information in my [report](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/report.pdf)
. It explains the main ideas behind the minecraft Clone, a performance test and the future work that could be done.

To learn Vulkan, I explained the all the steps of the rendering system developpement [here](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Vulkan-Intro/readme.md).



# Improvements

## BUffer all
Mergerd wait and exec GPU, buffered set_view_position

## Vertex buffer per chunk

First relative coordinates
* Add to vertex their chunk position
* Add to uniform buffer the cunk position of the view 
* Make in vertex shader a calculation :D (difference of chunk coord then go to absolute)


Then vertex buffer per chunk 
* Vertex still have their chunk coords
* Still unirform buffer contains position

Hashmap of (chunk_coords) ->vertex buffer + free indexes + quad_to_index


Before: create a secondary vertex buffer with everything except the vertex buffer binding 
At each iteration:
* Transform the hashmap into a vec of vertices (0.1 ms).
* Build a one submit primary command buffer with these vertex
* Submit it



* Hashmap (chunk coords) -> vertex buffer + free indexes + quad_to_index
* A



## Things to consider
* The vertex buffer should have to possibility to grow in size 