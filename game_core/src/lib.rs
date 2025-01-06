use winit::event::{KeyboardInput,MouseButton,ElementState, VirtualKeyCode};
use std::time::Instant;
use nalgebra_glm::two_pi;
use std::path::Path;
use std::fs;
use draw_element::*;

mod world;
use world::*;

/// Describe the folder where chunk are stored
const FOLDER_CHUNK:&str = "chunk_data";
const CHUNK_HEIGHT:usize = 512;
/// Used to generate the terrain, a higher value leads to smooth terrain.
const PERLIN_FACTOR:f64 = 10.;
/// Used to generate the terrain, describe the max height of a block
const PERLIN_MAX_HEIGHT:f64 = 10.;
/// Number of chunks rendered around the player
const RENDER_DISTANCE:i64 = 7;
/// Maximal distance to add, destroy blocks
const INTERACTION_RADIUS:f32 = 4.;
/// The maximal number of drawElement output by the update function
const MAX_DRAW_BATCH:usize = 200;

#[derive(Default,Copy,Clone)]
struct Entity{
    chunk_position: [i64;2],
    /// The relative position inside a chunk of the lowest point (xmin,ymin,zmin)
    relative_posititon: [f32;3],
    /// The length of the bounding box along each axis.
    bounding_box: [f32;3],
    speeds: [f32;3],
    on_ground: bool,
}
pub struct GameEngine {
    inputhandle: InputHandle,
    view_angle_horizontal: f32,
    view_angle_vertical: f32,
    mouse_sensitivity: f32,
    base_player_speed: f32,
    gravity_y: f32,
    time_last_update: Instant,
    world: World,
    /// The list of blocks that intersect the player vision 
    target_blocks: Vec<([i32;3],Face)>,
    player: Entity,
    /// Used to give DrawElement by batch and avoid lag.
    batch_draw_element: Vec<Vec<(bool,SurfaceElement)>>,
}

impl GameEngine {
    /// Create a new gameEngine.
    pub fn new(mouse_sensitivity: f32,base_player_speed: f32) -> (Self,Vec<(bool,SurfaceElement)>) {
        let relative_posititon = [0.,PERLIN_MAX_HEIGHT as f32 + 10.,0.];//[0.,PERLIN_MAX_HEIGHT as f32 +10.,0.];
        let chunk_position = [0,0];
        let mut world = World::new(0);
        if !(Path::new(FOLDER_CHUNK).exists() && Path::new(FOLDER_CHUNK).is_dir()) {
            match fs::create_dir(FOLDER_CHUNK) {
                Ok(_) => {
                    println!("Folder '{}' created to store chunk data.", FOLDER_CHUNK)
                }
                Err(e) => {
                    eprintln!("Failed to create folder '{}': {}", FOLDER_CHUNK, e);
                    panic!();
                }

            }
        } 
        // init chunks
        let range: Vec<_> = (-RENDER_DISTANCE..RENDER_DISTANCE+1).collect() ;
        world.load_chunk_and_wait(&range, &range);
        let elements_generated = world.get_and_clear_visibility_modifications();


        let player = Entity {
            chunk_position,
            relative_posititon,
            bounding_box: [0.6,1.8,0.6],
            speeds: [0.,0.,0.],
            on_ground: false,
        };
        (GameEngine {
            inputhandle: InputHandle::new(),
            view_angle_horizontal: 0.,
            view_angle_vertical: 0.,
            mouse_sensitivity,
            base_player_speed,
            gravity_y: -10., // block per second^2
            time_last_update: Instant::now(),
            world,
            target_blocks: Vec::new(),
            player,
            batch_draw_element: Vec::new(),
        },elements_generated)
    }
    
    /// Handle a key event
    pub fn key_event(&mut self,input: KeyboardInput){
        self.inputhandle.handle_key(input);
    }

    /// Handle a mouse event
    pub fn mouse_event(&mut self,action: ElementState ,button: MouseButton){
        self.inputhandle.handle_mbutton(action,button);
    }
    
    /// Handle the motion of the mouse.
    pub fn motion_event(&mut self,delta: (f64,f64)){
        self.inputhandle.handle_mouse(delta);
    }

    /// Update the world
    /// Returns the drawElements for the rendering system.
    pub fn update(&mut self) -> Vec<(bool,SurfaceElement)> {

        // Check is some chunk have loaded/unloaded
        self.world.check_loading_chunks();

        self.update_view_angle();
        self.set_obstacle_target();

        // If a mouse click is detected, place/remove a block
        if self.inputhandle.mouse_left {
            self.inputhandle.mouse_left = false;
            for &(coord,_) in &self.target_blocks {
                if self.world.remove_relative_block(coord, self.player.chunk_position) {
                    break;
                }
            }
        }
        if self.inputhandle.mouse_right {
            self.inputhandle.mouse_right = false;
            for &(coord,face) in &self.target_blocks {
                match self.world.get_relative_block(coord, self.player.chunk_position) {
                    Some(_) => {
                        let mut pos = coord;
                        match face {
                            Face::BOTTOM => { pos[1] -= 1;}
                            Face::TOP =>    { pos[1] += 1;} 
                            Face::EAST =>   { pos[0] += 1;} 
                            Face::WEST =>   { pos[0] -= 1;} 
                            Face::NORTH =>  { pos[2] += 1;} 
                            Face::SOUTH =>  { pos[2] -= 1;} 
                        }
                        if self.world.set_relative_block(
                            pos, 
                            self.player.chunk_position,
                            Block{
                                block_type: BlockType::Cobblestone
                            }) {
                            break;
                        }
                    }
                    None => {}
                }
            }
        }

        // Update speed and position
        let time_now = self.time_last_update.elapsed().as_micros() as f32 / 1000000.;
        self.update_speed(time_now);
        self.update_position(time_now);
        self.time_last_update = Instant::now();


        // Return drawElements by batches to avoid lag
        let draw_elements = self.world.get_and_clear_visibility_modifications();
        if !draw_elements.is_empty() {
            let  mut index = 0;
            let mut batch_element = Vec::new();
            for elem in draw_elements {
                if index >= MAX_DRAW_BATCH {
                    index = 0;
                    self.batch_draw_element.push(batch_element);
                    batch_element =  Vec::new();
                }
                batch_element.push(elem);
                index += 1;
            }
            self.batch_draw_element.push(batch_element);
        }
        match self.batch_draw_element.pop() {
            Some(batch_element) => {batch_element}
            None => {Vec::new()}
        }
    }

    /// Return the relative eyes position of the player
    pub fn get_eyes_position(&self) ->  ([f32;3],[i64;2]){
        /*[
            self.player.relative_posititon[0]+(self.player.bounding_box[0]/2.) + (self.player.chunk_position[0] as f32 * CHUNK_SIZE as f32),
            self.player.relative_posititon[1]+(self.player.bounding_box[1]*0.9),
            self.player.relative_posititon[2]+(self.player.bounding_box[2]/2.) + (self.player.chunk_position[1] as f32 * CHUNK_SIZE as f32)
        ]*/
        ([
            self.player.relative_posititon[0]+(self.player.bounding_box[0]/2.),
            self.player.relative_posititon[1]+(self.player.bounding_box[1]*0.9),
            self.player.relative_posititon[2]+(self.player.bounding_box[2]/2.)
        ],self.player.chunk_position)
    }


    /// Get the camera angles of the player 
    /// The first is the horizontal angle [0,2PI[
    /// The second is the vertical angle ]-PI/2,PI/2[
    pub fn get_camera_angles(&self) -> (f32,f32){
        (self.view_angle_horizontal, self.view_angle_vertical)
    }
    
    /// Update the view angle
    fn update_view_angle(&mut self){
        // update horizontal angle
        if self.inputhandle.mouse_motion_x!=0.{
            self.view_angle_horizontal += (self.inputhandle.mouse_motion_x as f32)*self.mouse_sensitivity;
            // stay in [0;2PI]
            if self.view_angle_horizontal.abs()> two_pi(){
                self.view_angle_horizontal = self.view_angle_horizontal%two_pi::<f32>();
            }
            self.inputhandle.mouse_motion_x = 0.;
        }
        // update vertical angle
        if self.inputhandle.mouse_motion_y!=0. {
            self.view_angle_vertical -= (self.inputhandle.mouse_motion_y as f32)*self.mouse_sensitivity;
            // block at half pi up and down
            if self.view_angle_vertical.abs()> 1.57{
                self.view_angle_vertical = 1.57 * self.view_angle_vertical.signum();
            }
            self.inputhandle.mouse_motion_y = 0.;
        }
    }
    
    /// Update the speed of the player
    fn update_speed(&mut self,time_elapsed: f32){
        // component forward
        let mut forward:f32;
        if self.inputhandle.z_key && !self.inputhandle.s_key {
            forward = 1.;
        }
        else if !self.inputhandle.z_key && self.inputhandle.s_key {
            forward = -1.;
        }
        else {
            forward = 0.;
        }
        // component side
        let mut side:f32;
        if self.inputhandle.d_key && !self.inputhandle.q_key {
            side = 1.;
        }
        else if !self.inputhandle.d_key && self.inputhandle.q_key {
            side = -1.;
        }
        else {
            side = 0.;
        }
        // component up
        if self.inputhandle.space_key && self.player.on_ground{
            self.player.speeds[1] = self.base_player_speed;
            self.player.on_ground = false;
        }
        // normalize flat speed (x,z)
        if forward.abs() + side.abs() > 1. {
            side *= 0.7071;
            forward *= 0.7071;
        }
        // rotate flat speed 
        let sin_angle = self.view_angle_horizontal.sin();
        let cos_angle = self.view_angle_horizontal.cos();
        self.player.speeds[0] = (side* cos_angle + forward* sin_angle)*self.base_player_speed;
        self.player.speeds[2] = (-side* sin_angle + forward* cos_angle)*self.base_player_speed;

        // Update speed y for gravity 
        self.player.speeds[1] += self.gravity_y*time_elapsed;

    }

    /// Update the postition of the player
    /// This function should be refactored.
    fn update_position(&mut self,time_elapsed: f32){
        // 1. Get next "theorical position"
        let next_x = self.player.relative_posititon[0] +  self.player.speeds[0]*time_elapsed;
        let next_y = self.player.relative_posititon[1] +  self.player.speeds[1]*time_elapsed;
        let next_z = self.player.relative_posititon[2] +  self.player.speeds[2]*time_elapsed;
        
        // 2. Get all obstacles in the zone (from play position to next position)
        // Not optimized -> get all obstacles in the cube formed by the initial and next position.
        // This do not matter for now as speed are relatively low. 
        let x_component;
        let y_component;
        let z_component;
        if self.player.speeds[0] > 0. {
            x_component = self.player.bounding_box[0];
        }
        else {
            x_component = 0.;
        }
        if self.player.speeds[1] > 0. {
            y_component = self.player.bounding_box[1];
        }
        else {
            y_component = 0.;
        }
        if self.player.speeds[2] > 0. {
            z_component = self.player.bounding_box[0];
        }
        else {
            z_component = 0.;
        }
        let from_x_component = self.player.bounding_box[0] - x_component;
        let from_y_component = self.player.bounding_box[1] - y_component;
        let from_z_component = self.player.bounding_box[2] - z_component;

        // Calculate the block of initial position and the block of next position
        let mut from_block = [f32_to_i32(self.player.relative_posititon[0]+from_x_component),f32_to_i32(self.player.relative_posititon[1]+from_y_component),f32_to_i32(self.player.relative_posititon[2]+from_z_component)];
        let mut to_block = [f32_to_i32(next_x+x_component),f32_to_i32(next_y+y_component),f32_to_i32(next_z+z_component)];

        // Put everything in the right order
        if from_block[0] > to_block[0] {
            let tmp = from_block[0];
            from_block[0] = to_block[0];
            to_block[0] = tmp;
        }
        if from_block[1] > to_block[1] {
            let tmp = from_block[1];
            from_block[1] = to_block[1];
            to_block[1] = tmp;
        }
        if from_block[2] > to_block[2] {
            let tmp = from_block[2];
            from_block[2] = to_block[2];
            to_block[2] = tmp;
        }

        // Loop on all blocks and get obstacles in each block
        let mut obstacles = Vec::new();
        for x in  from_block[0]..to_block[0]+1 {
            for y in from_block[1]..to_block[1]+1 {
                for z in from_block[2]..to_block[2]+1 {
                    obstacles.extend(self.get_obstacles_in_block([x,y,z]));
                }
            }
        }
        
        // 3. Check for collision along each face
        let mut next = [next_x,next_y,next_z];
        let mut actual = self.player.relative_posititon;

        check_collisions(&mut actual, &mut next,self.player.bounding_box,&mut self.player.speeds,time_elapsed,&mut obstacles,&mut self.player.on_ground);

        // player is out of bound in Y, move it higher
        if next[1] < 0. {
            self.player.relative_posititon[1] = PERLIN_MAX_HEIGHT as f32 + 10.;
        }
        else{
            self.player.relative_posititon = next;
            // Check if player is outside [0-15] 
            // if it is the case, change the chunk position, the relative player position and verify around chunk are loaded
            if self.player.relative_posititon [0] >= CHUNK_SIZE as f32 {
                self.player.relative_posititon [0] -= CHUNK_SIZE as f32;
                self.player.chunk_position[0] += 1;
                self.check_chunks();
            }
            else if self.player.relative_posititon [0] < 0. {
                self.player.relative_posititon [0] += CHUNK_SIZE as f32;
                self.player.chunk_position[0] -= 1;
                self.check_chunks();
            }
            if self.player.relative_posititon [2] >= CHUNK_SIZE as f32{
                self.player.relative_posititon [2] -= CHUNK_SIZE as f32;
                self.player.chunk_position[1] += 1;
                self.check_chunks();

            }
            else if self.player.relative_posititon [2] < 0. {
                self.player.relative_posititon [2] += CHUNK_SIZE as f32;
                self.player.chunk_position[1] -= 1;
                self.check_chunks();

            }
        }
    }

    /// Check in square around the player that all chunk are loaded/unloaded
    fn check_chunks(&mut self) {
        for x in -RENDER_DISTANCE-3..RENDER_DISTANCE+4 {
            for y in -RENDER_DISTANCE-3..RENDER_DISTANCE+4 {
                if x.abs() <= RENDER_DISTANCE && y.abs() <= RENDER_DISTANCE {
                    self.world.load_chunk([self.player.chunk_position[0]+x,self.player.chunk_position[1]+y]);
                }
                else if !(x.abs() <= RENDER_DISTANCE+1 && y.abs() <= RENDER_DISTANCE+1){
                    self.world.unload_chunk([self.player.chunk_position[0]+x,self.player.chunk_position[1]+y]);
                }
            }
        }
    }

    /// Set the target blocks 
    fn set_obstacle_target(&mut self){
        let x_direction = self.view_angle_horizontal.sin() * self.view_angle_vertical.cos();
        let y_direction = self.view_angle_vertical.sin();
        let z_direction = self.view_angle_horizontal.cos() * self.view_angle_vertical.cos();
        let dir = [x_direction,y_direction,z_direction];
        let view_point:[f32;3] =        [
            self.player.relative_posititon[0]+(self.player.bounding_box[0]/2.),
            self.player.relative_posititon[1]+(self.player.bounding_box[1]*0.9),
            self.player.relative_posititon[2]+(self.player.bounding_box[2]/2.)
        ];
        self.target_blocks = ray_intersect_blocks(view_point, dir, INTERACTION_RADIUS);
    }

    /// Return the obstacles lower position and bounding boxes in the block.
    /// For now, only return the blocks boundaries
    fn get_obstacles_in_block(&mut self ,coord: [i32;3]) -> Vec<([f32;3],[f32;3])>{
    let mut obstacles = Vec::new();
    match self.world.get_relative_block(coord, self.player.chunk_position){
        None => {return obstacles;}
        Some(_) => {
            obstacles.push(
                ([coord[0] as f32,coord[1] as f32,coord[2] as f32],[coord[0] as f32 + 1.,coord[1] as f32 + 1.,coord[2] as f32 + 1.])
            );
        }
    }
    obstacles
    }

}

/// Return a list of block that intersec the line drawn from origin in the normalized direction and which face it intersected.
fn ray_intersect_blocks(origin: [f32; 3],normalized_dir: [f32; 3],max_distance: f32) -> Vec<([i32; 3],Face)> {
    let mut blocks = Vec::new();
    let mut faces = [Face::BOTTOM;3];

    // Current block (integer coordinates)
    let mut block = [
        origin[0].floor() as i32,
        origin[1].floor() as i32,
        origin[2].floor() as i32,
    ];

    // Determine the step direction
    let step = [
        if normalized_dir[0] > 0.0 {faces[0]=Face::WEST; 1 } else {faces[0]=Face::EAST; -1 },
        if normalized_dir[1] > 0.0 {faces[1]=Face::BOTTOM; 1 } else {faces[1]=Face::TOP; -1 },
        if normalized_dir[2] > 0.0 {faces[2]=Face::SOUTH; 1 } else {faces[2]=Face::NORTH; -1 },
    ];

    // Calculate the initial t_max and t_delta for each axis
    let mut t_max = [
        ((block[0] as f32 + if step[0] > 0 { 1. } else { 0. }) - origin[0]) / normalized_dir[0],
        ((block[1] as f32 + if step[1] > 0 { 1. } else { 0. }) - origin[1]) / normalized_dir[1],
        ((block[2] as f32 + if step[2] > 0 { 1. } else { 0. }) - origin[2]) / normalized_dir[2],
    ];

    let t_delta = [
        (1.0 / normalized_dir[0]).abs(),
        (1.0 / normalized_dir[1]).abs(),
        (1.0 / normalized_dir[2]).abs(),
    ];

    // Traverse the grid
    let mut t = 0.0;
    let mut face = faces[0];
    while t < max_distance {
        // Add the current block to the list
        blocks.push((block,face));

        // Determine which axis to step along
        if t_max[0] < t_max[1] && t_max[0] < t_max[2] {
            t = t_max[0];
            t_max[0] += t_delta[0];
            block[0] += step[0];
            face = faces[0];
        } else if t_max[1] < t_max[2] {
            t = t_max[1];
            t_max[1] += t_delta[1];
            block[1] += step[1];
            face = faces[1];
        } else {
            t = t_max[2];
            t_max[2] += t_delta[2];
            block[2] += step[2];
            face = faces[2];
        }
    }
    blocks
}

/// Check is a collision has occured between the actual and next position with a certain hitbox.
/// This function will modify the speeds and the next position if a collision occured.
fn check_collisions(actual:&mut[f32;3],next:&mut[f32;3],hitbox:[f32;3],speeds:&mut [f32;3],time_elapsed: f32, obstacles: &mut Vec<([f32;3],[f32;3])>,on_ground:&mut bool) {
    let mut collisions = Vec::new();
    // Check collision with X face casting
    if speeds[0] > 0. {
        let from_face = [actual[0]+hitbox[0],actual[1],actual[2]];
        let to_face = [next[0]+hitbox[0],next[1],next[2]];
        match face_polygon_intersection_x(from_face,to_face,hitbox,obstacles,true){
            Some(intersec) => {
                collisions.push(intersec);
            }
            None => {}
        }
    }
    else if  speeds[0]< 0. {
        let from_face = *actual;
        let to_face = *next;
        match face_polygon_intersection_x(from_face,to_face,hitbox,obstacles,false){
            Some(intersec) => {
                collisions.push(intersec);
            }
            None => {}
        }
    }
    // Check collision with Y face casting
    if speeds[1] > 0. {
        let from_face = [actual[0],actual[1]+hitbox[1],actual[2]];
        let to_face = [next[0],next[1]+hitbox[1],next[2]];
        match face_polygon_intersection_y(from_face,to_face,hitbox,obstacles,true){
            Some(intersec) => {
                collisions.push(intersec);
            }
            None => {}
        }
    }
    else if speeds[1] < 0. {
        let from_face = [actual[0],actual[1],actual[2]];
        let to_face = [next[0],next[1],next[2]];
        match face_polygon_intersection_y(from_face,to_face,hitbox,obstacles,false){
            Some(intersec) => {
                collisions.push(intersec);
            }
            None => {}
        }
    }
    // Check collision with Z face casting
    if speeds[2] > 0. {
        let from_face = [actual[0],actual[1],actual[2]+hitbox[2]];
        let to_face = [next[0],next[1],next[2]+hitbox[2]];
        match face_polygon_intersection_z(from_face,to_face,hitbox,obstacles,true){
            Some(intersec) => {
                collisions.push(intersec);
            }
            None => {}
        }
    }
    else if speeds[2] < 0. {
        let from_face = [actual[0],actual[1],actual[2]];
        let to_face = [next[0],next[1],next[2]];
        match face_polygon_intersection_z(from_face,to_face,hitbox,obstacles,false){
            Some(intersec) => {
                collisions.push(intersec);
            }
            None => {}
        }
    }
    // If a collision was detected resolve the one that happened first thanks to the distance,
    // then make a second pass to be sure the player is no longer in collision
    if !collisions.is_empty() {
        collisions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        match collisions[0].0 {
            0 => {
                 next[0] = collisions[0].2;
                 speeds[0] = 0.;
            }
            1 => {
                next[1] = collisions[0].2;
                if speeds[1] < 0. {
                    *on_ground = true;
                }
                speeds[1] = 0.;
            }
            2 => {
                next[2] = collisions[0].2;
                speeds[2] = 0.;
                
            }
            _ => {}
        }
        check_collisions(actual, next, hitbox, speeds, time_elapsed, obstacles, on_ground);
    }
}

/// The actual faceCasting algorithm, it verifies that the polygon created by base_face and final_face intersects with obstacles
/// for Z face (NORTH or SOUTH).
/// Return:
/// * u8, always 2
/// * f32, the euclidian distance from the base_face to the obstacle (might not work for obastacle diffent than blocks)
/// * f32, the Z component of the position just before the collision occurs
/// * None, no collision occured
fn face_polygon_intersection_z(base_face:[f32;3],final_face:[f32;3],hitbox:[f32;3],obstacles: &mut Vec<([f32;3],[f32;3])>,z_positive:bool) -> Option<(u8,f32,f32)>{
    if z_positive {
        // We go in the Z positive direction
        // Sort obstacles by their min distance distance in ascending order
        obstacles.sort_by(|a, b| a.0[2].partial_cmp(&b.0[2]).unwrap());
        for obs in obstacles {
            // The plane of the obstacle is before the face plane, we ignore it 
            if obs.0[2] < base_face[2] {
                continue;
            }
            // The plane of the obstacle is in range of collision 
            if obs.0[2] < final_face[2]{
                // Check if component y,z overlaps with obstacle
                let intersect = get_point_on_line_by_z(base_face, final_face, obs.0[2]);

                let x_overlap = intersect[0] <= obs.1[0] && obs.0[0] <= (intersect[0] + hitbox[0]);
                let y_overlap = intersect[1] <= obs.1[1] && obs.0[1] <= (intersect[1] + hitbox[1]);
                
                if x_overlap && y_overlap {
                    // Calculate distance from base face to obstacle
                    let dx = intersect[0] - base_face[0];
                    let dy = intersect[1] - base_face[1];
                    let dz= intersect[2] - base_face[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    // return distance and point of impact
                    return Some((2,distance,obs.0[2]- hitbox[2] - 0.001))
                }
            }
        }
        None
    }
    else{
        // We go in the Z negative direction
        // Sort obstacles by their max distance in descending order
        obstacles.sort_by(|a, b| b.1[2].partial_cmp(&a.1[2]).unwrap());
        for obs in obstacles {
            // The plane of the obstacle is before the face plane, we ignore it 
            if obs.1[2] > base_face[2] {
                continue;
            }
            // The plane of the obstacle is in range of collision 
            if obs.1[2] > final_face[2]{
                // Check if component y,z overlaps with obstacle
                let intersect = get_point_on_line_by_z(base_face, final_face, obs.1[2]);

                let x_overlap = intersect[0] <= obs.1[0] && obs.0[0] <= (intersect[0] + hitbox[0]);
                let y_overlap = intersect[1] <= obs.1[1] && obs.0[1] <= (intersect[1] + hitbox[1]);
                
                if x_overlap && y_overlap {
                    // Calculate distance from base face to obstacle
                    let dx = intersect[0] - base_face[0];
                    let dy = intersect[1] - base_face[1];
                    let dz= intersect[2] - base_face[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    // return distance and point of impact
                    return Some((2,distance,obs.1[2]+ 0.001))
                }
            }
        }
        None
    }
}

/// The actual faceCasting algorithm, it verifies that the polygon created by base_face and final_face intersects with obstacles
/// for Y face (TOP or BOTTOM).
/// Return:
/// * u8, always 1
/// * f32, the euclidian distance from the base_face to the obstacle (might not work for obastacle diffent than blocks)
/// * f32, the Y component of the position just before the collision occurs
/// * None, no collision occured
fn face_polygon_intersection_y(base_face:[f32;3],final_face:[f32;3],hitbox:[f32;3],obstacles: &mut Vec<([f32;3],[f32;3])>,y_positive:bool) -> Option<(u8,f32,f32)>{
    if y_positive {
        // We go in the Y positive direction
        // Sort obstacles by their min distance distance in ascending order
        obstacles.sort_by(|a, b| a.0[1].partial_cmp(&b.0[1]).unwrap());
        for obs in obstacles {
            // The plane of the obstacle is before the face plane, we ignore it 
            if obs.0[1] < base_face[1] {
                continue;
            }
            // The plane of the obstacle is in range of collision 
            if obs.0[1] < final_face[1]{
                // Check if component y,z overlaps with obstacle
                let intersect = get_point_on_line_by_y(base_face, final_face, obs.0[1]);

                let x_overlap = intersect[0] <= obs.1[0] && obs.0[0] <= (intersect[0] + hitbox[0]);
                let z_overlap = intersect[2] <= obs.1[2] && obs.0[2] <= (intersect[2] + hitbox[2]);
                
                if x_overlap && z_overlap {
                    // Calculate distance from base face to obstacle
                    let dx = intersect[0] - base_face[0];
                    let dy = intersect[1] - base_face[1];
                    let dz= intersect[2] - base_face[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    // return distance and point of impact
                    return Some((1,distance,obs.0[1] - hitbox[1] - 0.001))
                }
            }
        }
        None
    }
    else{
        // We go in the Y negative direction
        // Sort obstacles by their max distance in descending order
        obstacles.sort_by(|a, b| b.1[1].partial_cmp(&a.1[1]).unwrap());
        for obs in obstacles {
            // The plane of the obstacle is before the face plane, we ignore it 
            if obs.1[1] > base_face[1] {
                continue;
            }
            // The plane of the obstacle is in range of collision 
            if obs.1[1] > final_face[1]{
                // Check if component y,z overlaps with obstacle
                let intersect = get_point_on_line_by_y(base_face, final_face, obs.1[1]);

                let x_overlap = intersect[0] <= obs.1[0] && obs.0[0] <= (intersect[0] + hitbox[0]);
                let z_overlap = intersect[2] <= obs.1[2] && obs.0[2] <= (intersect[2] + hitbox[2]);
                
                if x_overlap && z_overlap {
                    // Calculate distance from base face to obstacle
                    let dx = intersect[0] - base_face[0];
                    let dy = intersect[1] - base_face[1];
                    let dz= intersect[2] - base_face[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    // return distance and point of impact
                    return Some((1,distance,obs.1[1] + 0.001))
                }
            }
        }
        None
    }
}

/// The actual faceCasting algorithm, it verifies that the polygon created by base_face and final_face intersects with obstacles
/// for X face (WEST or EAST).
/// Return:
/// * u8, always 0
/// * f32, the euclidian distance from the base_face to the obstacle (might not work for obastacle diffent than blocks)
/// * f32, the X component of the position just before the collision occurs
/// * None, no collision occured
fn face_polygon_intersection_x(base_face:[f32;3],final_face:[f32;3],hitbox:[f32;3],obstacles: &mut Vec<([f32;3],[f32;3])>,x_positive:bool) -> Option<(u8,f32,f32)>{
    if x_positive {
        // We go in the X positive direction
        // Sort obstacles by their min distance distance in ascending order
        obstacles.sort_by(|a, b| a.0[0].partial_cmp(&b.0[0]).unwrap());
        for obs in obstacles {
            // The plane of the obstacle is before the face plane, we ignore it 
            if obs.0[0] < base_face[0] {
                continue;
            }
            // The plane of the obstacle is in range of collision 
            if obs.0[0] < final_face[0]{
                // Check if component y,z overlaps with obstacle
                let intersect = get_point_on_line_by_x(base_face, final_face, obs.0[0]);

                let y_overlap = intersect[1] <= obs.1[1] && obs.0[1] <= (intersect[1] + hitbox[1]);
                let z_overlap = intersect[2] <= obs.1[2] && obs.0[2] <= (intersect[2] + hitbox[2]);
                
                if y_overlap && z_overlap {
                    // Calculate distance from base face to obstacle
                    let dx = intersect[0] - base_face[0];
                    let dy = intersect[1] - base_face[1];
                    let dz= intersect[2] - base_face[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    // return distance and point of impact
                    return Some((0,distance,obs.0[0]- hitbox[0] - 0.001))
                }
            }
        }
        None
    }
    else{
        // We go in the X negative direction
        // Sort obstacles by their max distance in descending order
        obstacles.sort_by(|a, b| b.1[0].partial_cmp(&a.1[0]).unwrap());
        for obs in obstacles {
            // The plane of the obstacle is before the face plane, we ignore it 
            if obs.1[0] > base_face[0] {
                continue;
            }
            // The plane of the obstacle is in range of collision 
            if obs.1[0] > final_face[0]{
                // Check if component y,z overlaps with obstacle
                let intersect = get_point_on_line_by_x(base_face, final_face, obs.1[0]);

                let y_overlap = intersect[1] <= obs.1[1] && obs.0[1] <= (intersect[1] + hitbox[1]);
                let z_overlap = intersect[2] <= obs.1[2] && obs.0[2] <= (intersect[2] + hitbox[2]);
                
                if y_overlap && z_overlap {
                    // Calculate distance from base face to obstacle
                    let dx = intersect[0] - base_face[0];
                    let dy = intersect[1] - base_face[1];
                    let dz= intersect[2] - base_face[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    // return distance and point of impact
                    return Some((0,distance,obs.1[0] + 0.001))
                }
            }
        }
        None
    }
}

/// Return the point (y,z) on the line at x defined by point1 and point2
fn get_point_on_line_by_x(point1:[f32;3],point2:[f32;3],x:f32) -> [f32;3]{
    let t = (x - point1[0])/(point2[0]-point1[0]);
    let y = point1[1] + t*(point2[1]-point1[1]);
    let z = point1[2] + t*(point2[2]-point1[2]);
    [x,y,z]
}

/// Return the point (x,z) on the line at y defined by point1 and point2
fn get_point_on_line_by_y(point1:[f32;3],point2:[f32;3],y:f32) -> [f32;3]{
    let t = (y - point1[1])/(point2[1]-point1[1]);
    let x = point1[0] + t*(point2[0]-point1[0]);
    let z = point1[2] + t*(point2[2]-point1[2]);
    [x,y,z]
}

/// Return the point (x,y) on the line at y defined by point1 and point2
fn get_point_on_line_by_z(point1:[f32;3],point2:[f32;3],z:f32) -> [f32;3]{
    let t = (z - point1[2])/(point2[2]-point1[2]);
    let x = point1[0] + t*(point2[0]-point1[0]);
    let y = point1[1] + t*(point2[1]-point1[1]);
    [x,y,z]
}

/// Return the integer coordinate of a float coordinate
fn f32_to_i32(value:f32) -> i32{
    return value.floor() as i32
}

/// A structure used to store the the key/mouse information
#[derive(Default)]
struct InputHandle{
    pub mouse_motion_x: f64,
    pub mouse_motion_y: f64,
    pub z_key: bool,
    pub q_key: bool,
    pub s_key: bool,
    pub d_key: bool,
    pub space_key: bool,
    pub shift_key: bool,
    pub mouse_right: bool,
    pub mouse_left: bool,
}

impl InputHandle {
    pub fn new() -> Self {
        InputHandle::default()
    }
    /// Store mouse motion information
    pub fn handle_mouse(&mut self,delta: (f64,f64)){
        self.mouse_motion_x+=delta.0;
        self.mouse_motion_y+=delta.1;
    }
    /// Store key input information
    pub fn handle_key(&mut self,input: KeyboardInput){
        let key:VirtualKeyCode;
        match input.virtual_keycode {
            Some(value) => {
                key = value
            }
            None => {
                return;
            }
        }
        match key {
            VirtualKeyCode::Q => {
                if input.state == ElementState::Pressed {
                    self.q_key = true;
                } else {
                    self.q_key = false;
                }
            }
            VirtualKeyCode::S => {
                if input.state == ElementState::Pressed {
                    self.s_key = true;
                } else {
                    self.s_key = false;
                }
            }
            VirtualKeyCode::Z => {
                if input.state == ElementState::Pressed {
                    self.z_key = true;
                } else {
                    self.z_key = false;
                }
            }
            VirtualKeyCode::D => {
                if input.state == ElementState::Pressed {
                    self.d_key = true;
                } else {
                    self.d_key = false;
                }
            }
            VirtualKeyCode::Space => {
                if input.state == ElementState::Pressed {
                    self.space_key = true;
                } else {
                    self.space_key = false;
                }
            }
            VirtualKeyCode::LShift => {
                if input.state == ElementState::Pressed {
                    self.shift_key = true;
                } else {
                    self.shift_key = false;
                }
            }
            _ => {}
        }
    }

    /// Store mouse input information
    pub fn handle_mbutton(&mut self,action: ElementState ,button: MouseButton){
        let set;
        match action {
            ElementState::Pressed => {
                set = true;
            }
            ElementState::Released => {
                set = false;
            }
        }
        match button {
            MouseButton::Left => {
                self.mouse_left = set;
            }
            MouseButton::Right => {
                self.mouse_right = set;
            }
            _ => {
                return;
            }
        }
    }
}