/// The differents block types 
/// It is linked to block charac, I might need to change for a macro
#[derive(Eq, Hash, PartialEq,Clone,Copy,Debug)]
pub enum BlockTypes {
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