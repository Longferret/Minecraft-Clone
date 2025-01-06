use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use strum::IntoEnumIterator;
use draw_element::*;

/// A block structure, that composes a chunk.
/// In the future a block could be composed of subblocks.
/// For the moment the block types are:
// * 1 stone
// * 2 dirt block
// * 3 grass block
#[derive(Copy,Clone,Serialize, Deserialize, Debug)]
pub struct Block{
    pub block_type: BlockType,
    // sub_blocks
}


/// A chunk,
/// Its coordinates are not strictly enforced since it is a Hasmap of blocks,
/// normaly its coordinates are (enforced by parent structure World):
/// * X = [0,CHUNK_SIZE-1]
/// * Y = [0,CHUNK_HEIGHT-1]
/// * Z = [0,CHUNK_SIZE-1]
#[derive(Serialize, Deserialize, Debug)]
pub struct Chunk {
    /// Coordinates of the chunk
    pub coordinates: [i64;2],
    /// Blocks of the chunk
    blocks: HashMap<[i32;3],Block>,
    /// Visible faces, faces that have no adjacent blocks
    visible_faces: HashSet<([i32;3],Face,BlockType)>,
    /// Visible rectangle, the drawElements are computed from the visible faces to 
    /// group them for rendering optimization. 
    /// It is possible to merge visible_faces and this fiels (Hashmapbut<([i32;3],Face,u8),DrawElement>) 
    /// but I find it easier to manage this way.
    /// All the elements saved here have the field as_added set to true, it is my convention.
    visible_rectangles: HashSet<SurfaceElement>
}

impl Chunk {
    /// Return an empty chunk
    pub fn new(coordinates: [i64;2]) -> Self {
        Chunk {
            coordinates,
            blocks: HashMap::new(),
            visible_faces: HashSet::new(),
            visible_rectangles: HashSet::new()
        }
    }

    /// Delete a block at the relative position specified and update visibility.
    /// Return a vector of DrawElement.
    /// If no block where present, the function return an empty vector.
    pub fn remove_block(&mut self,coord:[i32;3]) -> Vec<(bool,SurfaceElement)> {
        let mut modified_faces = Vec::new();
        // Verify that the block is present
        match self.blocks.remove(&coord){
            Some(block) => {
                // Loop over all faces, remove them and get the drawelements generated
                for face in Face::iter() {
                    modified_faces.extend(self.remove_face(coord, face, block.block_type));
                }
            }
            // The block is not present, return
            None => {return modified_faces;}
        }
        let directions = [
            ([0, 0, 1], Face::SOUTH),
            ([0, 0, -1], Face::NORTH),
            ([1, 0, 0], Face::WEST),
            ([-1, 0, 0], Face::EAST),
            ([0, 1, 0], Face::BOTTOM),
            ([0, -1, 0], Face::TOP),
        ];
        // Loop over all adjacent blocks to add face of adjacent blocks
        for &(offset, opposite_face) in &directions {
            let neighbor_coord = [
                coord[0] + offset[0],
                coord[1] + offset[1],
                coord[2] + offset[2],
            ];
            // If an adjacent block is present, add the face in that direction and get the drawelements generated
            let key;
            match self.blocks.get_mut(&neighbor_coord) {
                Some(block) => {
                    key =  Some((neighbor_coord, opposite_face, block.block_type));
                }
                None => {
                    key = None;
                }
            }
            // Trick with key used because Rust don't allow us to have a mutable reference to self (get_mut) and calling a mutable self function
            match key {
                Some(k) => { modified_faces.extend(self.add_face(k.0, k.1, k.2)); }
                None => {}
            }
        }
        modified_faces

    }

    /// Add a block at the relative position specified and update visibility.
    /// Return a vector of DrawElement.
    /// If a block was present, it is removed.
    pub fn set_block(&mut self,coord:[i32;3], new_block: Block) -> Vec<(bool,SurfaceElement)> {
        let mut modified_faces = Vec::new();

        // If a block is removed, we removed its faces and get the drawelements.
        match self.blocks.insert(coord,new_block){
            Some(block) => {
                for face in Face::iter() {
                    modified_faces.extend(self.remove_face(coord, face, block.block_type));
                }
            }
            None => {}
        }
        let directions = [
            (Face::NORTH, [0, 0, 1], Face::SOUTH),
            (Face::SOUTH, [0, 0, -1], Face::NORTH),
            (Face::EAST, [1, 0, 0], Face::WEST),
            (Face::WEST, [-1, 0, 0], Face::EAST),
            (Face::TOP, [0, 1, 0], Face::BOTTOM),
            (Face::BOTTOM, [0, -1, 0], Face::TOP),
        ];
        // Loop over all faces to remove face of adjacent blocks and add face of the new block
        for &(face, offset, opposite_face) in &directions {
            let neighbor_coord = [
                coord[0] + offset[0],
                coord[1] + offset[1],
                coord[2] + offset[2],
            ];
            let key_add;
            let key_rem;
            match self.blocks.get_mut(&neighbor_coord) {
                // If an adjacent block is present, remove the face of the adjacent block in opposite direction and get the drawelements generated
                Some(block) => {
                    key_rem = Some((neighbor_coord,opposite_face,block.block_type));
                    key_add = None;
                }
                // If an adjacent block not present, add the face of the new block in that direction and get the drawelements generated
                None => {
                    key_rem = None;
                    key_add = Some((coord,face,new_block.block_type));
                } 
            } // End match adjacent block

            // Trick with key used because Rust don't allow us to have a mutable reference to self (get_mut) and calling a mutable self function
            match key_add {
                Some(k) => { modified_faces.extend(self.add_face(k.0, k.1, k.2));}
                None => {}
            }
            match key_rem {
                Some(k) => { modified_faces.extend(self.remove_face(k.0, k.1, k.2));}
                None => {}
            }

        } // End loop over direction
        modified_faces
    }

    /// Set the face of a block as "visible".
    /// Update the visible faces and rectangles.
    /// Return a vector of DrawElement.
    fn add_face(&mut self, coords: [i32;3],face: Face, blocktype: BlockType) -> Vec<(bool,SurfaceElement)>{

        // If insertion return false -> face already there, we return
        if !self.visible_faces.insert((coords,face,blocktype)) {
            return Vec::new();
        }

        // Set the indexes we are working on
        let index;
        let extend_index;
        if face == Face::BOTTOM || face == Face::TOP { 
            index = 0;
            extend_index = 0;
        }
        else if face == Face::NORTH || face == Face::SOUTH { 
            index = 0;
            extend_index = 0;
        }
        else { //face == Face::WEST || face == Face::EAST
            index = 2;
            extend_index = 1;
        }

        // Check how much consecutive faces are present before the new face
        // prev_length represents the number of consecutive faces before the new face (in the direction of index)
        let mut prev_coords = coords;
        let mut prev_length = 0;
        loop {
            prev_coords[index] -= 1;
            if !self.visible_faces.contains(&(prev_coords,face,blocktype)){
                break;
            }
            prev_length += 1;
        }
        // Check how much consecutive faces are present after the new face
        // next_length represents the number of consecutive faces after the new face (in the direction of index)
        let mut next_coords = coords;
        let mut next_length = 0;
        loop {
            next_coords[index] += 1;
            if !self.visible_faces.contains(&(next_coords,face,blocktype)){
                break;
            }
            next_length += 1;
        }


        let mut final_length = 1;
        let mut final_position = coords;
        let mut elements_to_modify = Vec::new();

        // Remove the previous element if any
        if prev_length > 0 {
            final_position[index] -= prev_length;
            final_length += prev_length;
            let removed_postion = final_position;
            let mut extends = [0,0];
            extends[extend_index] = prev_length as u32 -1;
            let drawelement =            
            SurfaceElement {
                blocktype,
                relative_position: removed_postion,
                chunk: self.coordinates,
                extends,
                face,
                orientation: Orientation::NONE,
            };
            self.visible_rectangles.remove(&drawelement);
            elements_to_modify.push((false,drawelement));
        }

        // Remove next element if any
        if next_length > 0 {
            final_length += next_length;
            let mut removed_postion = coords;
            removed_postion[index] += 1;
            let mut extends =  [0,0];
            extends[extend_index] = next_length as u32 -1;
            let drawelement = 
            SurfaceElement {
                blocktype,
                relative_position: removed_postion,
                chunk: self.coordinates,
                extends,
                face,
                orientation: Orientation::NONE,
            };
            self.visible_rectangles.remove(&drawelement);
            elements_to_modify.push((false,drawelement));
        }

        // Add the new element
        let mut extends = [0,0];
        extends[extend_index] = final_length as u32 -1;
        let drawelement= SurfaceElement {
            blocktype,
            relative_position: final_position,
            chunk: self.coordinates,
            extends,
            face,
            orientation: Orientation::NONE,
        };
        self.visible_rectangles.insert(drawelement.clone());
        elements_to_modify.push((true,drawelement));

        elements_to_modify
    }

    /// Set the face of a block as "not visible".
    /// Update the visible faces and rectangles.
    /// Return a vector of DrawElement.
    fn remove_face(&mut self, coords: [i32;3],face: Face, blocktype: BlockType) -> Vec<(bool,SurfaceElement)>{

        // If removal returns false -> the value was not present, nothing to be removed
        if !self.visible_faces.remove(&(coords,face,blocktype)){
            return Vec::new();
        }

        // Set the index we are working on
        let index;
        let extend_index;
        if face == Face::BOTTOM || face == Face::TOP { 
            index = 0;
            extend_index = 0;
        }
        else if face == Face::NORTH || face == Face::SOUTH { 
            index = 0;
            extend_index = 0;
        }
        else { //face == Face::WEST || face == Face::EAST
            index = 2;
            extend_index = 1;
        }

        // Check how much consecutive faces are present before the new face
        // prev_length represents the number of consecutive faces before the new face (in the direction of index)
        let mut prev_coords = coords;
        let mut prev_length = 0;
        loop {
            prev_coords[index] -= 1;
            if !self.visible_faces.contains(&(prev_coords,face,blocktype)){
                break;
            }
            prev_length += 1;
        }

        // Check how much consecutive faces are present after the new face
        // next_length represents the number of consecutive faces after the new face (in the direction of index)
        let mut next_coords = coords;
        let mut next_length = 0;
        loop {
            next_coords[index] += 1;
            if !self.visible_faces.contains(&(next_coords,face,blocktype)){
                break;
            }
            next_length += 1;
        }

        let mut final_length = 1;
        let mut final_position = coords;
        let mut elements_to_modify = Vec::new();

        // Recreate a DrawElement if any face before
        if prev_length > 0 {
            final_position[index] -= prev_length;
            final_length += prev_length;
            let add_position = final_position;
            let mut extends = [0,0];
            extends[extend_index] = prev_length as u32 -1;
            let drawelement = SurfaceElement {
                blocktype,
                relative_position: add_position,
                chunk: self.coordinates,
                extends,
                face,
                orientation: Orientation::NONE,
            };
            self.visible_rectangles.insert(drawelement.clone());
            elements_to_modify.push((true,drawelement));
        }
        // Recreate a DrawElement if any face after
        if next_length > 0 {
            final_length += next_length;
            let mut add_position = coords;
            add_position[index] += 1;
            let mut extends =  [0,0];
            extends[extend_index] = next_length as u32 -1;
            let drawelement =  SurfaceElement {
                blocktype,
                relative_position: add_position,
                chunk: self.coordinates,
                extends,
                face,
                orientation: Orientation::NONE,
            };
            self.visible_rectangles.insert(drawelement.clone());
            elements_to_modify.push((true,drawelement));
        }
        // Remove the element
        let mut extends = [0,0];
        extends[extend_index] = final_length as u32 -1;
        let drawelement =  
        SurfaceElement {
            blocktype,
            relative_position: final_position,
            chunk: self.coordinates,
            extends,
            face,
            orientation: Orientation::NONE,
        };
        self.visible_rectangles.remove(&drawelement);
        elements_to_modify.push((false,drawelement));
        
        elements_to_modify
    }

    /// Return the block at the relative position specified
    pub fn get_block(&self,coord:&[i32;3]) -> Option<&Block> {
        self.blocks.get(coord)
    }

    /// Return all the DrawElements by looping on the Hashset.
    /// Used when a chunk is loaded or unloaded
    pub fn get_visible_faces(&self, as_added: bool) -> Vec<(bool,SurfaceElement)> {

        let mut elements = Vec::new();
        for drawelement in &self.visible_rectangles {
            elements.push((as_added,drawelement.clone()));
        }
        elements
    }
}