use image::{DynamicImage, GenericImageView, Rgba};
use std::collections::HashMap;


// Load blocks from the path as 64x64 image
pub fn load_blocks_64x64(path: &str) -> (Vec<u8>,HashMap<u32, u32>) {
    let mut blocktype_to_imageindex = HashMap::new();
    let mut out_data:Vec<u8> = Vec::new();
    let mut index_charac = 0;
    let mut index_texture=0;
    for b in BLOCK_CHARAC{
        let block_name = b.name;
        let image: DynamicImage;
        let base = String::from(path);
        let result = image::open(base+"/"+block_name+".png");
        match result {
            Ok(value) => {
                image = value; // Crop image for number of texture declared
                blocktype_to_imageindex.insert(index_charac, index_texture);
            }
            Err(_e) => {
                panic!("Image {}.png not found.",block_name);
            }
        }
        // Make the block green from gray
        if block_name == "grass_block_top" {
            out_data.extend(load_green(image));
        }
        else if block_name == "lava_still"{
            let img = image.crop_imm(0, 0, 64, (b.textures+2)*32);
            out_data.extend(load_and_double(img, (b.textures+2)/2));
        }
        else if image.dimensions().0 == 64{
            out_data.extend(image.crop_imm(0, 0, 64, b.textures*64).to_rgba8().to_vec());
        }
        else{
            out_data.extend(image.crop_imm(64, 0, 64, b.textures*64).to_rgba8().to_vec());
        }
        index_charac +=1 ;
        index_texture += b.textures ;
    }
    (out_data,blocktype_to_imageindex)
}

fn load_and_double(image: DynamicImage,nbr:u32)->  Vec<u8>{
    let mut chunks = Vec::new();
    let mut out = Vec::new();
    for i in 0..nbr {
        chunks.push(image.crop_imm(0, i*64, 64, 64));
    } 
    for i in 1..nbr-1{
        chunks.push(chunks[(nbr-i-1) as usize].clone());
    }
    for c in chunks {
        out.extend(c.to_rgba8().to_vec());
    }
    out
}

fn load_green(image: DynamicImage)->  Vec<u8>{
    let mut img  = image.to_rgba8();
    for (_x, _y, pixel) in img.enumerate_pixels_mut() {
        let arr = pixel.0; // Get the RGBA values of the pixel
        let ratio = 0.58;
        // Replace gray by green (with a ratio)
        if arr[0] == arr[1] && arr[1] == arr[2] {
            let g = (arr[1] as f32)*0.9;
            let r = g*ratio;
            let b = r*ratio;
            *pixel = Rgba([r as u8, g as u8, b as u8, arr[3]]); 
        }
    }
    img.to_vec()
}
// Linked to block charac, might need to change for a macro
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
#[derive(Debug)]
pub struct BlockCharac{
    pub name: &'static str,
    pub textures: u32,
    pub animation_speed: u32, // In second/frame , MAX= 1000
}

pub const BLOCK_CHARAC: [BlockCharac;12] = [
    BlockCharac{
        name: "cobblestone",
        textures: 1,
        animation_speed: 0,
    },
    BlockCharac{
        name: "dirt",
        textures: 1,
        animation_speed: 0,
    },
    BlockCharac{
        name: "grass_block_side",
        textures: 1,
        animation_speed: 0,
    },
    BlockCharac{
        name: "grass_block_top",
        textures: 1,
        animation_speed: 0,
    },
    BlockCharac{
        name: "gravel",
        textures: 1,
        animation_speed: 0,
    },
    BlockCharac{
        name: "oak_planks",
        textures: 1,
        animation_speed: 0,
    },
    BlockCharac{
        name: "sand",
        textures: 1,
        animation_speed: 0,
    },
    BlockCharac{
        name: "stone",
        textures: 1,
        animation_speed: 0,
    },
    BlockCharac{
        name: "water_still",
        textures: 32,
        animation_speed: 2,
    },
    BlockCharac{
        name: "water_flow",
        textures: 64,
        animation_speed: 1,
    },
    BlockCharac{
        name: "lava_still",
        textures: 38,
        animation_speed: 1,
    },
    BlockCharac{
        name: "lava_flow",
        textures: 32,
        animation_speed: 1,
    },
];