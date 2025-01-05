use std::collections::HashMap;
use noise::{NoiseFn, Perlin};
use tokio::sync::mpsc;
use tokio::runtime::Runtime;
use std::fs::File;
use std::path::Path;
use std::io::{Write, Read};
use std::mem;
use rand::Rng;

use draw_element::*;

mod chunk;
use chunk::*;
pub use chunk::Block;

use super::CHUNK_SIZE;
use super::CHUNK_HEIGHT;
use super::PERLIN_FACTOR;
use super::PERLIN_MAX_HEIGHT;
use super::FOLDER_CHUNK;


/// A world structure.
pub struct World {
    /// Storage of all chunks
    chunks: HashMap<[i64;2],Chunk>,
    perlin: Perlin,
    /// The modification that will be given to the renderer
    modified_faces: Vec<(bool,SurfaceElement)>,
    // To Load/Unload chunks by other threads
    receiver_chunk_load: mpsc::Receiver<(Chunk,Vec<(bool,SurfaceElement)>)>, 
    sender_chunk_load: mpsc::Sender<(Chunk,Vec<(bool,SurfaceElement)>)>,   
    receiver_chunk_unload: mpsc::Receiver<Vec<(bool,SurfaceElement)>>,  
    sender_chunk_unload: mpsc::Sender<Vec<(bool,SurfaceElement)>>,  
    tokio_runtime: Runtime,
}

impl World {
    /// Return an empty world.
    pub fn new (seed: u32) -> Self {
        let (sender_chunk_load,receiver_chunk_load) = mpsc::channel(30);
        let (sender_chunk_unload,receiver_chunk_unload) = mpsc::channel(30);
        World {
            chunks: HashMap::new(),
            perlin: Perlin::new(seed),
            modified_faces: Vec::new(),
            receiver_chunk_load,
            sender_chunk_load,
            receiver_chunk_unload,
            sender_chunk_unload,
            tokio_runtime: Runtime::new().unwrap(),
        }
    }

    /// Return the DrawElements and clear them (ownership transfered)
    pub fn get_and_clear_visibility_modifications(&mut self) -> Vec<(bool,SurfaceElement)> {
        mem::take(&mut self.modified_faces)
    }

    /// Get a block from its relative coordinates and the chunk coordinates.
    /// It is possible to get a block outside of the chunk.
    /// Return the block, None if the chunk is not loaded or Y coordinate is out of bound.
    pub fn get_relative_block(&mut self, relative_coord:[i32;3], chunk_coord:[i64;2]) -> Option<&Block>  {
        let mut rel_coord = relative_coord;
        let mut c_coord = chunk_coord;
        // Y is out of bound -> not possible
        if relative_coord[1] > CHUNK_HEIGHT as i32 -1 || relative_coord[1] < 0 {
            return None
        }
        // X is out of bound [0,15] -> modify chunk pos
        while rel_coord[0] > CHUNK_SIZE as i32 -1{
            rel_coord[0] -= CHUNK_SIZE as i32;
            c_coord[0] += 1;
        }
        while rel_coord[0] < 0 {
            rel_coord[0] += CHUNK_SIZE as i32;
            c_coord[0] -= 1;
        }
        // Z is out of bound [0,15] -> modify chunk pos
        while rel_coord[2] > CHUNK_SIZE as i32 -1{
            rel_coord[2] -= CHUNK_SIZE as i32;
            c_coord[1] += 1;
        }
        while rel_coord[2] < 0 {
            rel_coord[2] += CHUNK_SIZE as i32;
            c_coord[1] -= 1;
        }
        match self.chunks.get(&c_coord){
            Some(chunk) => {
                chunk.get_block(&rel_coord)   
            }
            None => {
                println!("Get block - Chunk is not loaded");
                None
            }
        }
    }
    
    /// Set a block from its relative coordinates and the chunk coordinates.
    /// It is possible to set a block outside of the chunk.
    /// Return true if the block was modified, false if the chunk was not loaded or Y coordinate is out of bound
    pub fn set_relative_block(&mut self, relative_coord:[i32;3], chunk_coord:[i64;2], block: Block) -> bool {
        let mut rel_coord = relative_coord;
        let mut c_coord = chunk_coord;
        // Y is out of bound -> not possible
        if relative_coord[1] > CHUNK_HEIGHT as i32 -1 || relative_coord[1] < 0 {
            return false
        }
        // X is out of bound [0,15] -> modify chunk pos
        while rel_coord[0] > CHUNK_SIZE as i32 -1 {
            rel_coord[0] -= CHUNK_SIZE as i32;
            c_coord[0] += 1;
        }
        while rel_coord[0] < 0 {
            rel_coord[0] += CHUNK_SIZE as i32;
            c_coord[0] -= 1;
        }
        // Z is out of bound [0,15] -> modify chunk pos
        while rel_coord[2] > CHUNK_SIZE as i32 -1 {
            rel_coord[2] -= CHUNK_SIZE as i32;
            c_coord[1] += 1;
        }
        while rel_coord[2] < 0 {
            rel_coord[2] += CHUNK_SIZE as i32;
            c_coord[1] -= 1;
        }
        match self.chunks.get_mut(&c_coord){
            Some(chunk) => {
                self.modified_faces.extend(chunk.set_block(rel_coord, block));
                true
            }
            None => { 
                println!("Set block: Chunk is not loaded");
                false 
            }
        }
    }

    /// Remove a block from its relative coordinates and the chunk coordinates.
    /// It is possible to remove a block outside of the chunk.
    /// Return true if the block was deleted, false if the chunk was not loaded or Y coordinate is out of bound
    pub fn remove_relative_block(&mut self, relative_coord:[i32;3], chunk_coord:[i64;2]) -> bool {
        let mut rel_coord = relative_coord;
        let mut c_coord = chunk_coord;
        // Y is out of bound -> not possible
        if relative_coord[1] > CHUNK_HEIGHT as i32 -1 || relative_coord[1] < 0 {
            return false
        }
        // X is out of bound [0,15] -> modify chunk pos
        if rel_coord[0] > CHUNK_SIZE as i32 -1{
            rel_coord[0] -= CHUNK_SIZE as i32;
            c_coord[0] += 1;
        }
        else if rel_coord[0] < 0 {
            rel_coord[0] += CHUNK_SIZE as i32;
            c_coord[0] -= 1;
        }
        // Z is out of bound [0,15] -> modify chunk pos
        if rel_coord[2] > CHUNK_SIZE as i32 -1{
            rel_coord[2] -= CHUNK_SIZE as i32;
            c_coord[1] += 1;
        }
        else if rel_coord[2] < 0 {
            rel_coord[2] += CHUNK_SIZE as i32;
            c_coord[1] -= 1;
        }
        match self.chunks.get_mut(&c_coord){
            Some(chunk) => {
                let modified = chunk.remove_block(rel_coord);
                let ret = 
                if modified.is_empty() {
                    false
                }
                else{
                    true
                };
                self.modified_faces.extend(modified);
                ret
            }
            None => {
                println!("Remove block: Chunk is not loaded");
                false
            }
        }
    }
    
    /// Unload a chunk and saves it.
    /// The chunk is not unloaded here, a thread is created do it in parallel. 
    pub fn unload_chunk(&mut self,chunk_coord: [i64;2]){
        let sender = self.sender_chunk_unload.clone();
        match self.chunks.remove(&chunk_coord){
            Some(chunk) => {
                // Run the thread for chunk unloading
                self.tokio_runtime.spawn(async move {
                    let removed_squares = unload_chunk_async(chunk,chunk_coord).await;
                    sender.send(removed_squares).await.unwrap();
                });
            }
            None => { 
                return;
            }
        }
    }

    /// Load a chunk or generate it from perlin noise.
    /// The chunk is not loaded here, a thread is created do it in parallel. 
    pub fn load_chunk(&mut self,chunk_coord: [i64;2]){
        // The chunk is already loaded
        if self.chunks.contains_key(&chunk_coord){
            return;
        }
        let perlin = self.perlin.clone();
        let sender = self.sender_chunk_load.clone();
        // Preload the Chunk (avoid 0.5 ms delay when inserting loaded chunk)
        self.chunks.insert(chunk_coord, Chunk::new(chunk_coord));

        // Run the thread for chunk loading
        self.tokio_runtime.spawn(async move {
            let chunk = load_chunk_async(perlin, chunk_coord).await;
            sender.send((chunk.1,chunk.0)).await.unwrap();
        });
    }

    /// Non-Blocking check if a chunk has finished loading/unloading
    pub fn check_loading_chunks(&mut self){
        match self.receiver_chunk_load.try_recv() {
            Ok((chunk,added)) => {
                self.modified_faces.extend(added);        
                self.chunks.insert(chunk.coordinates, chunk);
            }
            Err (_) => { return;}
        }
        match self.receiver_chunk_unload.try_recv() {
            Ok(removed) => {
                self.modified_faces.extend(removed);
            }
            Err (_) => { return;}
        }
    } 

    /// Blocking chunk loading.
    /// It waits until all the chunks are properly loaded.
    pub fn load_chunk_and_wait(&mut self,range_x: &Vec<i64>, range_y: &Vec<i64>) {
        // Ask for chunk loading
        for x in range_x {
            for y in range_y {
                self.load_chunk([*x,*y]);
            }
        }
        // Wait for chunks to be loaded
        let mut test = 0;
        for _ in range_x {
            for _ in range_y {
                test += 1;
                let (chunk,added) = self.receiver_chunk_load.blocking_recv().unwrap();
                self.modified_faces.extend(added);        
                self.chunks.insert(chunk.coordinates, chunk);
            }
        }
        println!("TEST CHUNK: {:?}",test);
    }

}

impl Drop for World {
    /// Save all chunks when world is dropped.
    /// This can be improved to work in parrellel.
    fn drop(&mut self) {
        let all_chunks:Vec<([i64; 2], Chunk)> = self.chunks.drain().collect();
        for (coord,chunk) in all_chunks {
            self.tokio_runtime.block_on(async{
                unload_chunk_async(chunk,coord).await;
            });
        }
    }
}

/// Generate or load a chunk from memory.
/// Returns the DrawElements of the chunk generated and the chunk.
async fn load_chunk_async(perlin: Perlin ,chunk_coord: [i64;2]) -> (Vec<(bool,SurfaceElement)>,Chunk){
    let chunk_path = FOLDER_CHUNK.to_string() + "/" + "chunk_(" + &chunk_coord[0].to_string() + "_" + &chunk_coord[1].to_string() + ")";
    let mut chunk = Chunk::new(chunk_coord);
    // Chunk file exists
    if Path::new(&chunk_path).exists() && Path::new(&chunk_path).is_file(){
        // Unwrapping everything has the game must panic (no need to handle those cases)
        let mut file = File::open(chunk_path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
    
        // Deserialize the binary data back into a HashMap
        chunk = bincode::deserialize(&buffer).unwrap();
    }
    // Chunk file does not exist
    else{
        // Fill chunk thanks to perlin noise
        let absolute_x = chunk_coord[0] * CHUNK_SIZE as i64;
        let absolute_z = chunk_coord[1] * CHUNK_SIZE as i64;
        let mut a_x = absolute_x;
        let mut rng = rand::thread_rng(); // Create a random number generator
        let random_number: i32 = rng.gen_range(1..=100); // Generate a number between 1 and 100 (inclusive)
        let top_b;
        let middle_b;
        let botom_b;
        if random_number <= 5 {
            top_b = BlockType::Sand;
            middle_b = BlockType::Cobblestone;
            botom_b = BlockType::Cobblestone
        }
        else if random_number <= 15{
            top_b = BlockType::LavaStill;
            middle_b = BlockType::LavaStill;
            botom_b = BlockType::LavaStill
        }
        else if random_number <= 20{
            top_b = BlockType::Cobblestone;
            middle_b = BlockType::Cobblestone;
            botom_b = BlockType::Cobblestone
        }
        else if random_number <= 30 {
            top_b = BlockType::Gravel;
            middle_b = BlockType::Gravel;
            botom_b = BlockType::Gravel
        }
        else if random_number <= 35 {
            top_b = BlockType::OakPlanks;
            middle_b = BlockType::OakPlanks;
            botom_b = BlockType::OakPlanks
        }
        else{
            top_b = BlockType::GrassBlock;
            middle_b = BlockType::Dirt;
            botom_b = BlockType::Stone
        }
        for x in 0..CHUNK_SIZE as i32 {
            let mut a_z = absolute_z;
            for z in 0..CHUNK_SIZE as i32 {
                let value = perlin.get([a_x as f64 / PERLIN_FACTOR, a_z as f64 / PERLIN_FACTOR]);
                let perlin_y = ((value + 1.) * PERLIN_MAX_HEIGHT/2.) as i32 + 5; // from -1,1 to 0-maxheight+5                
                chunk.set_block(
                    [x,perlin_y,z], 
                    Block{
                        block_type: top_b
                    }
                );
                for y in 0..perlin_y {
                    if y < perlin_y-2 {
                        chunk.set_block(
                            [x,y,z], 
                            Block{
                                block_type: botom_b
                            }
                        );
                    }
                    else{
                        chunk.set_block(
                            [x,y,z], 
                            Block{
                                block_type: middle_b
                            }
                        );
                    }
                }
                a_z += 1;
            }
            a_x += 1;
        }
    }
    (chunk.get_visible_faces(true),chunk)
}

/// Save a chunk to memory.
/// Returns the DrawElements of the chunk saved.
async fn unload_chunk_async(chunk: Chunk,chunk_coord: [i64;2]) -> Vec<(bool,SurfaceElement)> {
    // Unwrapping everything, the thread must panic in case of error (no need to handle those cases).
    let chunk_encoded = bincode::serialize(&chunk).unwrap();
    let chunk_path  = format!("chunk_data/chunk_({}_{})", chunk_coord[0], chunk_coord[1]);
    let mut file = File::create(chunk_path).unwrap();
    file.write_all(&chunk_encoded).unwrap();
    chunk.get_visible_faces(false)
}

