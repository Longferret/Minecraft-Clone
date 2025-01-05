use draw_element::{BlockType,Face,SurfaceElement};

/// The differents block types 
/// It is linked to block charac, I might need to change for a macro
#[derive(Eq, Hash, PartialEq,Clone,Copy,Debug)]
pub enum TextureType {
    Cobblestone = 0,
    Dirt = 1,
    GrassBlockSide = 2,
    GrassBlockTop = 3,
    Gravel = 4,
    OakPlanks = 5,
    Sand = 6,
    Stone = 7,
    WaterStill = 8,
    WaterFlow = 9,
    LavaStill = 10,
    LavaFlow = 11, 
}

pub fn get_texturetype(surface_elem: &SurfaceElement) -> TextureType{
    let blocktype = surface_elem.blocktype;
    let face = surface_elem.face;
    match blocktype {
        BlockType::Cobblestone => {
            TextureType::Cobblestone
        }
        BlockType::Dirt => {
            TextureType::Dirt
        }
        BlockType::GrassBlock => {
            match face {
                Face::TOP => {
                    TextureType::GrassBlockTop
                }
                Face::BOTTOM => {
                    TextureType::Dirt
                }
                _ => {
                    TextureType::GrassBlockSide
                }
            }
        }
        BlockType::Gravel => {
            TextureType::Gravel
        }
        BlockType::OakPlanks => {
            TextureType::OakPlanks
        }
        BlockType::Sand => {
            TextureType::Sand
        }
        BlockType::Stone => {
            TextureType::Stone
        }
        BlockType::WaterFlow => {
            TextureType::WaterFlow
        }
        BlockType::WaterStill => {
            TextureType::WaterStill
        }
        BlockType::LavaFlow => {
            TextureType::LavaFlow
        }
        BlockType::LavaStill => {
            TextureType::LavaStill
        }
    }
}

pub fn get_texture_charac(surface_elem: &SurfaceElement) -> &'static BlockCharac{
    let texturetype = get_texturetype(surface_elem);
    return &BLOCK_CHARAC[texturetype as usize];
}

/// A structure that holds all the block characteristics
#[derive(Debug)]
pub struct BlockCharac{
    pub name: &'static str,
    pub textures: u32,
    pub animation_speed: u32, 
    pub is_transparent: bool,
}

// All the block characteristics
pub const BLOCK_CHARAC: [BlockCharac;12] = [
    BlockCharac{
        name: "cobblestone",
        textures: 1,
        animation_speed: 0,
        is_transparent: false,
    },
    BlockCharac{
        name: "dirt",
        textures: 1,
        animation_speed: 0,
        is_transparent: false,
    },
    BlockCharac{
        name: "grass_block_side",
        textures: 1,
        animation_speed: 0,
        is_transparent: false,
    },
    BlockCharac{
        name: "grass_block_top",
        textures: 1,
        animation_speed: 0,
        is_transparent: false,
    },
    BlockCharac{
        name: "gravel",
        textures: 1,
        animation_speed: 0,
        is_transparent: false,
    },
    BlockCharac{
        name: "oak_planks",
        textures: 1,
        animation_speed: 0,
        is_transparent: false,
    },
    BlockCharac{
        name: "sand",
        textures: 1,
        animation_speed: 0,
        is_transparent: false,
    },
    BlockCharac{
        name: "stone",
        textures: 1,
        animation_speed: 0,
        is_transparent: false,
    },
    BlockCharac{
        name: "water_still",
        textures: 32,
        animation_speed: 2,
        is_transparent: true,
    },
    BlockCharac{
        name: "water_flow",
        textures: 64,
        animation_speed: 1,
        is_transparent: true,
    },
    BlockCharac{
        name: "lava_still",
        textures: 38,
        animation_speed: 1,
        is_transparent: false,
    },
    BlockCharac{
        name: "lava_flow",
        textures: 32,
        animation_speed: 2,
        is_transparent: false,
    },
];