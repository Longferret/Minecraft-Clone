# Draw Element Definition - [`lib.rs`](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Minecraft-v0.3/draw_element/src/lib.rs)
This document contains all information on the draw_element implementation.


## The objective

This library crate in an in between for the renderer crate and the game_core crate.
It is used to format the exchange between the two crates.

For the moment the crate only have one structure `SurfaceElement` that defines a a group a contiguous block face that form a rectangular shape.

`SurfaceElement` contains:
* `pub blocktype: BlockType`, the type of block to draw (enum in the library) 
* `pub relative_position: [i32;3]`, the relative position of the smallest face of the group.
* `pub chunk: [i64;2]`, the position of the chunk
* `pub extends: [u32;2]`, the number of (face-1) on the 2 first directions.
* `pub face: Face`, the face of the block to draw (enum in the library) 
* `pub orientation: Orientation`, the orientation of the block to draw (enum in the library) 


the `relative_position` is meant to stay in thoses bounds:
* (X) index 0:  [0,CHUN_NORM-1] 
* (Y) index 1:  any              
* (Z) index 2:  [0,CHUN_NORM-1] 

`CHUN_NORM` defines the size of a chunk in a `SurfaceElement`, it allows modularity. 
When creating a `SurfaceElement`, it is recommended to call the function `convert_from_chunksize`, that will convert the `SurfaceElement` from any chunksize to the chunk size of the crate.

It allows the renderer or the game_core crate to change their chunk_size.

More on `extends`, a face is fixed on 1 axis.

For example if the face is fixed on the Y axis (UP/BOTTOM face):
* extends[0] is how much face there are along X
* extends[1] is how much face there are along Z

## Functions

There are functions to convert a `SurfaceElement` from and to any chunk size.

There is a function to get the absolute postion of a `SurfaceElement`.
