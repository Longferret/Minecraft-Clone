# Chunk Definition and Algorithms - [`chunk.rs`](https://github.com/Longferret/Minecraft-Clone/blob/main/Minecraft-v0.3/game_core/src/world/chunk.rs)
This document contains all information on the chunk implementation.


## The structure

The chunk structure contains:
* `coordinates: [i64;2]`, the coordinates of the chunk
* `blocks: HashMap<[i32;3],Block>`, the blocks in the chunk
* `visible_faces: HashSet<([i32;3],Face,BlockType)>`, the visible face of the chunk
* `visible_rectangles: HashSet<SurfaceElement>`, a group of visible face

The `visible_faces` are set/unset by simply looking if a block is adjacent to the actual block. This check is always done when a block is added/removed.

The `visible_rectangles` are calculated when `visible_faces` is updated. It is an algorithm that tries to group the longest possible lines of faces. Its goal is to group faces to draw them together to reduce the number of vertices in the rendering system.

`SurfaceElement` is a special structure used to exchange information on how to draw a surface between the rendering system and the game core. It is explained in depth [here](https://github.com/Longferret/Minecraft-Clone/blob/main/Minecraft-v0.3/docs/draw_element/draw_element.md).


## Functions
This part talks about the algorithms and the logic of the functions defined inside the file `chunk.rs`.

I don't explain all the functions here, only the ones that contain a special logic or algorithm.

### `pub fn remove_block(&mut self,coord:[i32;3]) -> Vec<(bool,SurfaceElement)>`

This function is used to remove a block in the chunk, it returns a vector of the `SurfaceElement` to draw or remove. The boolean specify if the `SurfaceElement` is to be added (true) or removed (false).

The function simply removes the block from the chunk, delete the `visible_faces` of that block and add the `visible_faces of adjacent blocks to not leave a hole.


### `pub fn set_block(&mut self,coord:[i32;3], new_block: Block) -> Vec<(bool,SurfaceElement)>`

This function is used to set a block in the chunk, it returns a vector of the `SurfaceElement` to draw or remove. The boolean specify if the `SurfaceElement` is to be added (true) or removed (false).

The function adds the block in the chunk, add the `visible_faces` of that block and remove the `visible_faces of adjacent blocks.

It also remove the `visible_faces` of old block is there were one.

### `fn add_face(&mut self, coords: [i32;3],face: Face, blocktype: BlockType) -> Vec<(bool,SurfaceElement)>`

This function is used to add a `visible_faces` and calculate the `visible_rectangles`, it returns a vector of the `SurfaceElement`.

I implemented a special algorithm to group faces. The idea is to group visible faces on a line. 

Here are the detailled steps:
#### 1. Set the axis we are working on
It means that:
* if the face we are adding is pointing NORTH or SOUTH, it tries to create a group of face on the X axis.
* for BOTTOM or TOP, on the X axis. 
* for WEST or EAST, on the Z axis.

#### 2. Verify how many consecutive `visible_faces` are before and after the face added along the specified axis.
It is done by iterating in the negative and positive until no `visible_faces` are found.

#### 3. Remove the `SurfaceElement` if any
A draw element can encapsulate multiple `visible_faces`. If previous `visible_faces` were found, it means a `SurfaceElement` is there and it is deleted from the `visible_rectangles`. The same goes to next `visible_faces` found.

#### 4. Create the final `SurfaceElement`
After the deletion of the adjacent `SurfaceElement`, a new one is created and inserted in  `visible_rectangles`. This newly created `SurfaceElement` contains all the `visible_faces` found earlier and the new visible face added.

And Finally we insert the new face in `visible_faces`.

The returned value is a vector composed of the `SurfaceElement` deleted (if any) and the `SurfaceElement` created.

This algorithm works pretty well but can be further improved to group visible faces by rectangle instead of lines.

### `fn remove_face(&mut self, coords: [i32;3],face: Face, blocktype: BlockType) -> Vec<(bool,SurfaceElement)>`

This function is used to remove a `visible_faces` and calculate the `visible_rectangles`, it returns a vector of the `SurfaceElement`.

It is pretty much the same one that adds a new visible face, but instead of merging `SurfaceElement` it splits them. Meaning that 1 `SurfaceElement` will be deleted and 0 to 2 can be created.