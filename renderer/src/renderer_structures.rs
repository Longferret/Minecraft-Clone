use nalgebra_glm::{identity, TMat4};
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

/// The vertex sturcture used by shaders
/// * position, the position of the vertex
/// * uv, the position for the texture
/// * block_type, the texture to display
#[derive(BufferContents, Vertex, Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    #[format(R32_UINT)]
    pub block_type: u32,
    #[format(R32G32_SINT)]
    pub block_offset: [i32;2],
}

// The distance between player position and a block (for sorting transparent blocks later)
/*impl MyVertex {
    // Calculate the Manhattan distance from a block to the player
    pub fn manhattan_distance(&self, player_pos: [f32;3]) -> f32 {
        (self.position [0]- player_pos[0]).abs() + (self.position[1] - player_pos[1]).abs() + (self.position[2] - player_pos[2]).abs()
    }
}*/