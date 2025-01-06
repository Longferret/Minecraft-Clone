use serde::{Serialize, Deserialize};
use strum_macros::EnumIter;

/// This crate is used to define the draw elements.
/// It is a structured way to exchange message between the rendering system and other crates.
/// It only contains the structure "SurfaceElement" to draw a surface composed of blocks for the moment.

/// Size of a chunk (XZ) for a draw element
pub const CHUNK_SIZE:usize = 16;

/// A face of any block or entity 
/// * NORTH -> toward Z+
/// * SOUTH -> toward Z-
/// * EAST  -> toward X+
/// * WEST  -> toward X-
/// * TOP   -> toward Y+
/// * BOTTOM-> toward Y-
#[derive(Eq,PartialEq,Hash,Clone, Copy, Serialize, Deserialize, Debug,EnumIter)]
pub enum Face {
    NORTH,
    SOUTH, 
    WEST, 
    EAST,  
    TOP,  
    BOTTOM,
}

/// Block types
#[derive(Serialize, Deserialize,Eq, Hash, PartialEq,Clone,Copy,Debug)]
pub enum BlockType {
    Cobblestone,
    Dirt,
    GrassBlock,
    Gravel,
    OakPlanks,
    Sand,
    Stone,
    WaterFlow,
    WaterStill,
    LavaFlow,
    LavaStill,
}


/// Define the orientation of the block faces
#[derive(Serialize, Deserialize,Eq, Hash, PartialEq,Clone,Copy,Debug)]
pub enum Orientation {
    NONE,
    QUARTER,
    HALF,
    THREEQUARTER,
}

/// An element used to carry information on what to draw and how.
/// The convention of the chunk size is CHUNK_NORM, it is still possible to transform
/// any chunk size to the convention for modularity (an other rendering system or crates could use different chunk size).
#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq, Hash)]
pub struct SurfaceElement {
    pub blocktype: BlockType,
    /// Should be in the range, but it is not mandatory
    /// * index 0:  [0,CHUNK_NORM-1] (X)
    /// * index 1:  any              (Y)
    /// * index 2:  [0,CHUNK_NORM-1] (Z)
    pub relative_position: [i32;3],
    pub chunk: [i64;2],
    /// Tells how much block there is after this one in the first and second direction.
    /// Used for optimization in rendering system.
    /// Ex:  
    /// * face = BOTTOM -> It can extends over X or Z since Y is fixed
    /// * extends[0] is how much to extend in the X+ direction
    /// * extends[1] is how much to extend in the Z+ direction
    pub extends: [u32;2],
    pub face: Face,
    pub orientation: Orientation,
}

impl SurfaceElement {
    /// Return the coordinates of the surface element in absolute coordinates
    pub fn get_absolute_position(&self) -> [i64;3]{
        [
            (self.chunk[0] * CHUNK_SIZE as i64) + self.relative_position[0] as i64,
            self.relative_position[1] as i64,
            (self.chunk[1] * CHUNK_SIZE as i64) + self.relative_position[2] as i64,
        ]
    }
    /*/// Convert a SurfaceElement from specified chunk size to fit the convention (CHUNK_NORM)
    pub fn convert_from_chunksize(&mut self,chunk_size: usize){
        if chunk_size == CHUNK_NORM {
            return;
        }
        let absolute = relative_to_absolute(self.chunk,self.relative_position, chunk_size);
        (self.chunk,self.relative_position) = absolute_to_relative(absolute, CHUNK_NORM)
    }
    /// Convert a SurfaceElement from the CHUNK_NORM to a specified chunksize
    pub fn convert_to_chunksize(&mut self,chunk_size: usize){
        if chunk_size == CHUNK_NORM {
            return;
        }
        let absolute = relative_to_absolute(self.chunk,self.relative_position, CHUNK_NORM);
        (self.chunk,self.relative_position) = absolute_to_relative(absolute, chunk_size)
    }*/
}

/// Transform absolute coordinates to chunk and relative coordinates
fn _absolute_to_relative(absolute_coord: [i64;3], chunk_size: usize) -> ([i64;2],[i32;3]){
    let mut chunk_coord: [i64; 2] = [0, 0]; // [X, Z]
    let mut relative_coord: [i32; 3] = [0, absolute_coord[1] as i32, 0];
    let chunki32 = chunk_size as i32;
    if absolute_coord[0]<0 {
        chunk_coord[0] = (absolute_coord[0] - (chunk_size as i64-1))/chunk_size as i64;
        relative_coord[0] = (absolute_coord[0] as i32 % chunki32 + chunki32) % chunki32;
    }
    else {
        chunk_coord[0] = absolute_coord[0]/chunk_size as i64;
        relative_coord[0] = absolute_coord[0] as i32 % chunki32;
    }
    if absolute_coord[2]<0 {
        chunk_coord[1] = (absolute_coord[2] - (chunk_size as i64-1))/chunk_size as i64;
        relative_coord[2] = (absolute_coord[2] as i32 % chunki32 + chunki32) % chunki32;
    }
    else {
        chunk_coord[1] = absolute_coord[2]/ chunk_size as i64;
        relative_coord[2] = absolute_coord[2] as i32% chunki32;
    }
    
    (chunk_coord, relative_coord)
}

/// Transform relative coordinates and chunk coordinates into absolute coordinates
fn _relative_to_absolute(chunk_coord: [i64;2],relative_coord:[i32;3], chunk_size: usize) -> [i64;3]{
    [
        (chunk_coord[0] * chunk_size as i64) + relative_coord[0] as i64,
        relative_coord[1] as i64,
        (chunk_coord[1] * chunk_size as i64) + relative_coord[2] as i64,
    ]
}