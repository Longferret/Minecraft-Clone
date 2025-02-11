use crate::block::*;
use crate::renderer_structures::MyVertex;


/// Defines the input data needed to add/remove a square from the rendering system
#[derive(Eq, Hash, PartialEq,Clone,Debug)]
pub struct Square{
    pub x:i64,
    pub y:i64,
    pub z:i64,
    pub extent1: u32, // draw one more square in the direction 1 (1 is the first direction along x+,y+,z+)
    pub extent2: u32, // draw one more square in the direction 2 (2 is the second direction along x+,y+,z+)
    pub orientation: Orientation, // only implemented for top block
    pub direction: Direction,
    pub block_type: BlockTypes,
    pub is_interior: bool,
}

/// Defines a face of a cube.
#[derive(Eq, Hash, PartialEq,Clone,Debug)]
pub enum Direction{
    UP,         // Y+
    DOWN,       // Y-
    RIGHT,      // X+
    LEFT,       // X-
    FORWARD,    // Z+
    BACKWARD,   // Z-
}

/// Define the orientation of a square
#[derive(Eq, Hash, PartialEq,Clone,Copy,Debug)]
pub enum Orientation {
    NONE,
    QUARTER,
    HALF,
    THREEQUARTER,
}

/// Create a square with x fixed, do not support orientation, very ugly might need some work
pub fn get_square_xfixed(x:f32,y:f32,z:f32,half:f32,is_right:bool,extent1:f32,extent2:f32,block_type:u32) -> Vec<MyVertex>{
    let valy: [f32; 2];
    let valz: [f32; 2];
    let uvy: [f32; 2];
    let uvz: [f32; 2];
    let var: f32;
    if is_right {
        uvy = [0.,1.+extent2];
        uvz = [0.,1.+extent1];
        valy = [y+half+extent1,y-half];
        valz = [z-half,z+half+extent2];
        var = half;
    }
    else{
        uvy = [1.+extent2,0.];
        uvz = [1.+extent1,0.];
        valy = [y-half,y+half+extent1];
        valz = [z-half,z+half+extent2];
        var = -half;
    }
    let mut vertices = Vec::new();
    for i in 0..2{
        for j in 0..2{
            vertices.push(MyVertex{
                position: [x+var,valy[j],valz[i]],
                uv: [uvy[i],uvz[j]],
                block_type: block_type
            });
        }
    }
    let v4 = vertices.pop().unwrap();
    vertices.push(vertices[2].clone());
    vertices.push(vertices[1].clone());
    vertices.push(v4);

    vertices
}

/// Create a square with y fixed, very ugly might need some work
pub fn get_square_yfixed(x:f32,y:f32,z:f32,half:f32,is_top:bool,extent1:f32,extent2:f32,block_type:u32,orientation:Orientation) -> Vec<MyVertex>{
    let valx: [f32; 2];
    let valz: [f32; 2];
    let uvx: [f32; 2];
    let uvz: [f32; 2];
    let var: f32;
    let ij_inverted :bool;
    match orientation{
        Orientation::QUARTER => {
            //uvx = [1.+extent1,0.];
            uvx = [0.,1. + extent1];
            uvz = [0.,1.+extent2];
            ij_inverted = false;
        }
        Orientation::HALF => {
            uvx = [1.+extent1,0.];
            uvz = [0.,1.+extent2];
            ij_inverted = true;
        }
        Orientation::THREEQUARTER => {
            uvx = [1.+extent1,0.];
            uvz = [1.+extent2,0.];
            ij_inverted = false;
        }
        _ => {
            uvx = [0.,1.+extent1];
            uvz = [1.+extent2,0.];
            ij_inverted = true;
        }
    }

    if is_top {
        valx = [x-half,x+half+extent1];
        valz = [z-half,z+half+extent2];
        var = half;

    }
    else{
        valx = [x+half+extent1,x-half];
        valz = [z-half,z+half+extent2];
        var = -half;
    }

    let mut vertices = Vec::new();
    if ij_inverted {
        for i in 0..2{
            for j in 0..2{
                vertices.push(MyVertex{
                    position: [valx[j],y+var,valz[i]],
                    uv: [uvx[j],uvz[i]],
                    block_type: block_type
                });
            }
        }
    }
    else{
        for i in 0..2{
            for j in 0..2{
                vertices.push(MyVertex{
                    position: [valx[j],y+var,valz[i]],
                    uv: [uvx[i],uvz[j]],
                    block_type: block_type as u32
                });
            }
        }
    }
    let v4 = vertices.pop().unwrap();
    vertices.push(vertices[2].clone());
    vertices.push(vertices[1].clone());
    vertices.push(v4);

    vertices
}

/// Create a square with z fixed, do not support orientation, very ugly might need some work
pub fn get_square_zfixed(x:f32,y:f32,z:f32,half:f32,is_forward:bool,extent1:f32,extent2:f32,block_type:u32) -> Vec<MyVertex>{

    let valx: [f32; 2];
    let valy: [f32; 2];
    let uvx: [f32; 2];
    let uvy: [f32; 2];
    let var: f32;
    if is_forward {
        uvx = [0.,1.+extent1];
        uvy = [1.+extent2,0.];
        valx = [x+half+extent1,x-half];
        valy = [y-half,y+half+extent2];
        var = half;
    }
    else{
        uvx = [0.,1.+extent1];
        uvy = [1.+extent2,0.];
        valx = [x-half,x+half+extent1];
        valy = [y-half,y+half+extent2];
        var = -half;
    }
    let mut vertices = Vec::new();
    for i in 0..2{
        for j in 0..2{
            vertices.push(MyVertex{
                position: [valx[j],valy[i],z+var],
                uv: [uvx[j],uvy[i]],
                block_type: block_type
            });
        }
    }
    let v4 = vertices.pop().unwrap();
    vertices.push(vertices[2].clone());
    vertices.push(vertices[1].clone());
    vertices.push(v4);

    vertices
}



