# World Definition and Algorithms - [`world.rs`](https://github.com/Longferret/Minecraft-Clone/blob/main/Minecraft-v0.3/game_core/src/world.rs)
This document contains all information on the world implementation.


## The structure

The world structure contains:
* `chunks: HashMap<[i64;2],Chunk>`, chunks in the world.
* `perlin: Perlin`, the perlin noise used to generate the chunks.
* `modified_faces: Vec<(bool,SurfaceElement)>`, the currently modified face in the world.
* A bunch of variables for asynchronous and parrallel communications.

The `SurfaceElement` of `modified_faces` are added when a function that modifies a chunk is called. The boolean specify if the `SurfaceElement` is to be added (true) or removed (false).

The `modified_faces` vector is cleared when the function `get_and_clear_visibility_modifications` is called.

## Functions

### `get/set/remove_block (&mut self, relative_coord:[i32;3], chunk_coord:[i64;2])`
These are functions to get,set and remove a block inside any chunk. 

These functions do coordinate checking,
the relative coordinate inside a chunk must be:
* X: [0,CHUNK_SIZE[ 
* Y: [0,CHUNK_HEIGHT[
* Z: [0,CHUNK_SIZE[

If the Y component of the relative coordinates is not in the bound, they return directly. 

If the X or Z component are out of bound, the functions will just change the chunks for the call to be inbound. 

### `load/unload_chunk(&mut self,chunk_coord: [i64;2])`
These functions create new threads to handle the loading or unloading of a chunk.

### `pub fn check_loading_chunks(&mut self)`
This function do a non-blocking check to see if a chunk has finished loading/unloading.

If a chunk has finished loading, it adds the chunk in the hashmap.

If a chunk has finished unloading, it removes the chunk of the hasmap.

### `async fn load_chunk_async(perlin: Perlin ,chunk_coord: [i64;2]) -> (Vec<(bool,SurfaceElement)>,Chunk)`
This is the function that new thread will use to load a chunk in parrallel of the main thread.

It checks if the chunk is in the memory and load it. In the other case, it uses the perlin noise to create a new one.

The function returns the newly created chunk and a vector of `SurfaceElement`, the elements to add to the rendering system.

### `async fn unload_chunk_async(chunk: Chunk,chunk_coord: [i64;2]) -> Vec<(bool,SurfaceElement)>`
This is the function that new thread will use to unload a chunk in parrallel of the main thread.

It saves the chunk in memory and returns a vector of `SurfaceElement`, the elements to remove from the rendering system.

