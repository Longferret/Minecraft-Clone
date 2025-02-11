use winit::event::{KeyboardInput,MouseButton,ElementState, VirtualKeyCode};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use nalgebra_glm::two_pi;
use noise::{NoiseFn, Perlin};

const CHUNK_SIZE:usize = 16;
const CHUNK_HEIGHT:usize = 512;
const INTERACTION_RADIUS:f32 = 4.;
const PERLIN_FACTOR:f64 = 20.;
const PERLIN_MAX_HEIGHT:f64 = 20.;

#[derive(Default,Copy,Clone)]
struct Entity{
    chunk_position: [i64;3],
    // if relative_pos > 16 || relative pos < 0
    relative_posititon: [f32;3], // lowest point (xmin,ymin,zmin)
    bounding_box: [f32;3],
    speeds: [f32;3],
    gravity_affected: bool,
    on_ground: bool,
}
#[derive(Default,Copy,Clone)]
pub struct Block{
    block_type: u8,
    // sub_blocks
}
 
pub struct Chunk {
    data: Box<[[[Block; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>, // Z,Y,X
    visible_blocks: Vec<(u8, [i64; 3])>,
    //blocks: HashMap<[i64;3],Block>,
    //visible_blocks: HashSet<[i64;3]>
    //visible_faces ??
}

impl Chunk {

    pub fn new(perlin: &Perlin,x:i64,z:i64) -> Self {
        let mut chunk: Box<[[[Block; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]> = Box::new([[[Block::default(); CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]);
        let real_x ;
        let real_z;
        if x < 0 {                       
            real_x = x* CHUNK_SIZE as i64; // -1  -> [-15,0[ , -2 -> [-31,-15[
        }
        else {
            real_x = x* CHUNK_SIZE as i64; // 0 ->[0,16[ , 1 -> [16,32[
        }
        if z < 0 {
            real_z = z* CHUNK_SIZE as i64;
        }
        else {
            real_z = z* CHUNK_SIZE as i64;
        }
        // 0 air 
        // 1 stone
        // 2 dirt block
        // 3 grass block
        //println!("{:?} {:?}", real_x,real_z);
        // Je suis un peu con je pense
        let mut r_x = real_x;
        let mut block_added = Vec::new();
        for x in 0..CHUNK_SIZE {
            let mut r_z = real_z;
            for z in 0..CHUNK_SIZE{
                let value = perlin.get([r_x as f64 / PERLIN_FACTOR, r_z as f64 / PERLIN_FACTOR]);
                let perlin_y = ((value + 1.) * PERLIN_MAX_HEIGHT/2.) as usize; // from -1,1 to 0-maxheight
                chunk[x][perlin_y][z] = Block {block_type: 3};
                block_added.push((3,[r_x,perlin_y as i64,r_z]));
                //println!("{:?} {:?} {:?}",r_x,perlin_y,r_z);
                r_z += 1;
            }
            r_x += 1;
        }
        Chunk {
            data: chunk,
            visible_blocks: block_added,
        }
    }

    pub fn set_block() {}

    pub fn get_block() {}

    pub fn get_visible_blocks() {}
    pub fn get_list_of_blocks(&self) -> &Vec<(u8, [i64; 3])>{
        &self.visible_blocks
    }
}
// World coordinates
// X [-Inf,+Inf]
// Y [0,]
// Z [-Inf,+Inf]
pub struct World {
    pub chunks: HashMap<[i64;2],Chunk>,
    pub perlin: Perlin,
    block_removed: Vec<(u8,[i64;3])>,
    block_added: Vec<(u8,[i64;3])>,
    // player: Entity
    // mobs :Entity
}

impl World {
    pub fn new (seed: u32) -> Self {
        World {
            chunks: HashMap::new(),
            perlin: Perlin::new(seed),
            block_added: Vec::new(),
            block_removed: Vec::new(),
        }
    }

    pub fn get_recent_blocks_modifications(&self) -> (&Vec<(u8,[i64;3])>,&Vec<(u8,[i64;3])>) {
        (&self.block_added,&self.block_removed)
    }

    pub fn clear_recent_blocks_modifications(&mut self) {
        self.block_added.clear();
        self.block_removed.clear();
    }

    pub fn get_relative_block(&mut self, relative_coord:[i64;3], chunk_coord:[i64;2]) -> Option<Block>  {
        let mut rel_coord = relative_coord;
        let mut c_coord = chunk_coord;
        // Y is out of bound -> not possible
        if relative_coord[1] > CHUNK_HEIGHT as i64 -1 || relative_coord[1] < 0 {
            return None
        }
        // X is out of bound [0,15] -> modify chunk pos
        if relative_coord[0] >= CHUNK_SIZE as i64 -1 {
            rel_coord[0] -= CHUNK_SIZE as i64;
            c_coord[0] += 1;
        }
        else if relative_coord[0] < 0 {
            rel_coord[0] += CHUNK_SIZE as i64;
            c_coord[0] -= 1;
        }
        // Z is out of bound [0,15] -> modify chunk pos
        if relative_coord[2] >= CHUNK_SIZE as i64 -1 {
            rel_coord[2] -= CHUNK_SIZE as i64;
            c_coord[1] += 1;
        }
        else if relative_coord[2] < 0 {
            rel_coord[2] += CHUNK_SIZE as i64;
            c_coord[1] -= 1;
        }
        match self.chunks.get(&c_coord){
            Some(chunk) => {
                Some(chunk.data[rel_coord[0] as usize][rel_coord[1]as usize][rel_coord[2]as usize])
            }
            None => {
                println!("Block not loaded");
                self.load_chunk(c_coord);
                None
            }
        }
    }

    pub fn set_relative_block(&mut self, relative_coord:[i64;3], chunk_coord:[i64;2], block: Block){
        let mut rel_coord = relative_coord;
        let mut c_coord = chunk_coord;
        // Y is out of bound -> not possible
        if relative_coord[1] > CHUNK_HEIGHT as i64 -1 || relative_coord[1] < 0 {
            return
        }
        // X is out of bound [0,15] -> modify chunk pos
        if relative_coord[0] >= CHUNK_SIZE as i64 -1 {
            rel_coord[0] -= CHUNK_SIZE as i64;
            c_coord[0] += 1;
        }
        else if relative_coord[0] < 0 {
            rel_coord[0] += CHUNK_SIZE as i64;
            c_coord[0] -= 1;
        }
        // Z is out of bound [0,15] -> modify chunk pos
        if relative_coord[2] >= CHUNK_SIZE as i64 -1 {
            rel_coord[2] -= CHUNK_SIZE as i64;
            c_coord[1] += 1;
        }
        match self.chunks.get_mut(&c_coord){
            Some(chunk) => {
                if chunk.data[rel_coord[0] as usize][rel_coord[1]as usize][rel_coord[2] as usize].block_type == 0 {

                }

                chunk.data[rel_coord[0] as usize][rel_coord[1]as usize][rel_coord[2] as usize] = block;
            }
            None => {}
        }
    }

    pub fn get_block(&mut self, coord: [i64;3])-> Option<Block> {
        if coord[1] > CHUNK_HEIGHT as i64 -1 || coord[1] < 0 {
            return None
        }
        let mut chunk_coord: [i64; 2]  = [0,0]; // [X,Z]
        let mut relative_coord: [i64; 3]  =[0,coord[1],0];
        let i64chunksize = CHUNK_SIZE as i64;
        if coord[0]<0 {
            chunk_coord[0] = (coord[0] - (i64chunksize-1))/i64chunksize;
            relative_coord[0] = (coord[0] % i64chunksize + i64chunksize) % i64chunksize;
        }
        else {
            chunk_coord[0] = coord[0]/i64chunksize;
            relative_coord[0] = coord[0]% i64chunksize;
        }
        if coord[2]<0 {
            chunk_coord[1] = (coord[2] - (i64chunksize-1))/i64chunksize;
            relative_coord[2] = (coord[2] % i64chunksize + i64chunksize) % i64chunksize;
        }
        else {
            chunk_coord[1] = coord[2]/i64chunksize;
            relative_coord[2] = coord[2]% i64chunksize;
        }
        match self.chunks.get(&chunk_coord){
            Some(chunk) => {
                Some(chunk.data[relative_coord[0] as usize][relative_coord[1]as usize][relative_coord[2]as usize])
            }
            None => {
                println!("Block not loaded");
                self.load_chunk(chunk_coord);
                None
            }
        }
    }

    pub fn set_block(&mut self, block:Block, coord: [i64;3]){
        if coord[1] > CHUNK_HEIGHT as i64 -1 || coord[1] < 0 {
            return
        }
        let mut chunk_coord: [i64; 2]  = [0,0]; // [X,Z]
        let mut relative_coord: [i64; 3]  =[0,coord[1],0];
        let i64chunksize = CHUNK_SIZE as i64;
        if coord[0]<0 {
            chunk_coord[0] = (coord[0] - (i64chunksize-1))/i64chunksize;
            relative_coord[0] = (coord[0] % i64chunksize + i64chunksize) % i64chunksize;
        }
        else {
            chunk_coord[0] = coord[0]/i64chunksize;
            relative_coord[0] = coord[0]% i64chunksize;
        }
        if coord[2]<0 {
            chunk_coord[1] = (coord[2] - (i64chunksize-1))/i64chunksize;
            relative_coord[2] = (coord[2] % i64chunksize + i64chunksize) % i64chunksize;
        }
        else {
            chunk_coord[1] = coord[2]/i64chunksize;
            relative_coord[2] = coord[2]% i64chunksize;
        }
        match self.chunks.get_mut(&chunk_coord){
            Some(chunk) => {
                let oldblock = chunk.data[relative_coord[0] as usize][relative_coord[1]as usize][relative_coord[2] as usize];
                if block.block_type == 0 {
                    self.block_removed.push((oldblock.block_type,coord));
                }
                chunk.data[relative_coord[0] as usize][relative_coord[1]as usize][relative_coord[2] as usize] = block;
            }
            None => {}
        }
    }

    pub fn load_chunk(&mut self,coord: [i64;2]){
        match self.chunks.get(&coord){
            Some(_) => {return}
            None => {

                let chunk = Chunk::new(&self.perlin, coord[0], coord[1]);
                self.block_added.extend(chunk.get_list_of_blocks());
                self.chunks.insert(coord, chunk);
            }
        }
    }

    pub fn unload_chunk(&mut self,coord: [i64;2]){
        match self.chunks.get(&coord){
            None => {return}
            Some(chunk) => {
                self.block_removed.extend(chunk.get_list_of_blocks());
                self.chunks.remove(&coord);
            }
        }
        self.chunks.remove(&coord);
    }
}

pub struct GameEngine {
    inputhandle: InputHandle,
    player_position: [f32;3],
    view_angle_horizontal: f32,
    view_angle_vertical: f32,
    mouse_sensitivity: f32,
    base_player_speed: f32,
    speeds: [f32;3],
    gravity_y: f32,
    time_last_update: Instant,
    on_ground: bool,
    hitbox: [f32;3], // X,Y,Z (length along each axys)
    world: World,
    target_block: Option<(u8,[i64;3])>
}

impl GameEngine {

    pub fn new(mouse_sensitivity: f32,base_player_speed: f32,player_position: [f32;3]) -> Self {

        let world = World::new(0);
        //world.load_chunk([0,0]); // OK
 
        GameEngine {
            inputhandle: InputHandle::new(),
            player_position,
            view_angle_horizontal: 0.,
            view_angle_vertical: 0.,
            mouse_sensitivity,
            base_player_speed,
            speeds: [0.,0.,0.],
            gravity_y: -10., // block per second^2
            time_last_update: Instant::now(),
            on_ground: false,
            hitbox: [0.6,1.8,0.6],
            world,
            target_block: None
        }
    }

    /// Key event
    pub fn key_event(&mut self,input: KeyboardInput){
        self.inputhandle.handle_key(input);
    }

    pub fn mouse_event(&mut self,action: ElementState ,button: MouseButton){
        self.inputhandle.handle_mbutton(action,button);
    }
    /// Motion of the mouse event
    pub fn motion_event(&mut self,delta: (f64,f64)){
        self.inputhandle.handle_mouse(delta);
    }

    /// Update the world
    pub fn update(&mut self) -> (&Vec<(u8,[i64;3])>,&Vec<(u8,[i64;3])>) {

        self.world.clear_recent_blocks_modifications();

        self.update_view_angle();
        self.set_obstacle_target();

        if self.inputhandle.mouse_left {
            match self.target_block {
                Some(block)=> {
                    self.world.set_block(
                        Block{
                            block_type: 0
                        }, block.1);
                    self.target_block = None;
                    self.inputhandle.mouse_left = false;
                }
                None => {}
            }
        }
        let time_now = self.time_last_update.elapsed().as_micros() as f32 / 1000000.;
        self.update_speed(time_now);
        self.update_position(time_now);
        self.time_last_update = Instant::now();
        /*
        let mut chunk_coord:[i64;2] = [0,0];
        let i64chunksize = CHUNK_SIZE as i64;
        if self.player_position[0]<0. {
            chunk_coord[0] = (self.player_position[0]as i64 - (i64chunksize-1))/i64chunksize;
        }
        else {
            chunk_coord[0] = self.player_position[0] as i64/i64chunksize;
        }
        if self.player_position[2]<0. {
            chunk_coord[1] = (self.player_position[2]as i64 - (i64chunksize-1))/i64chunksize;
        }
        else {
            chunk_coord[1] = self.player_position[2]as i64/i64chunksize;
        }
        for x in -4..4 {
            for z  in -4..4 {
                if (x as i32).abs() > 2 || (z as i32).abs() >2 {
                    //println!("UNLOAD {:?}", [chunk_coord[0]+x,chunk_coord[1]+z]);
                    self.world.unload_chunk([chunk_coord[0]+x,chunk_coord[1]+z]);
                }
                else{
                    self.world.load_chunk([chunk_coord[0]+x,chunk_coord[1]+z]);
                }
            }
        }
        */
        self.world.get_recent_blocks_modifications()
    }

    /// Get the eyes position of the player
    /// Calculate with its hitbox
    pub fn get_eyes_position(&self) ->  [f32;3]{
        [
            self.player_position[0]+(self.hitbox[0]/2.),
            self.player_position[1]+(self.hitbox[1]*0.9),
            self.player_position[2]+(self.hitbox[2]/2.)
        ]
    }

    /// Get the camera angles of the player 
    /// The first the horizontal angle [0,2PI[
    /// The second is the vertical angle ]-PI/2,PI/2[
    pub fn get_camera_angles(&self) -> (f32,f32){
        (self.view_angle_horizontal, self.view_angle_vertical)
    }
    
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
    
    // Speed update
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
        if self.inputhandle.space_key && self.on_ground{
            self.speeds[1] = self.base_player_speed;
            self.on_ground = false;
        }
        /*else if self.inputhandle.shift_key { 
            self.speeds[1] = -self.base_player_speed;
        }
        else{
            self.speeds[1]=0.;
        }*/
        // normalize flat speed (x,z)
        if forward.abs() + side.abs() > 1. {
            side *= 0.7071;
            forward *= 0.7071;
        }
        // rotate flat speed 
        let sin_angle = self.view_angle_horizontal.sin();
        let cos_angle = self.view_angle_horizontal.cos();
        self.speeds[0] = (side* cos_angle + forward* sin_angle)*self.base_player_speed;
        self.speeds[2] = (-side* sin_angle + forward* cos_angle)*self.base_player_speed;

        // Update speed y for gravity 
        self.speeds[1] += self.gravity_y*time_elapsed;

    }

    // Update of postition
    fn update_position(&mut self,time_elapsed: f32){

        // Face casting test
        let x_component;
        let y_component;
        let z_component;
        if self.speeds[0] > 0. {
            x_component = self.hitbox[0];
        }
        else {
            x_component = 0.;
        }
        if self.speeds[1] > 0. {
            y_component = self.hitbox[1];
        }
        else {
            y_component = 0.;
        }
        if self.speeds[2] > 0. {
            z_component = self.hitbox[0];
        }
        else {
            z_component = 0.;
        }
        // Get next "theorical position"
        let next_x = self.player_position[0] +  self.speeds[0]*time_elapsed;
        let next_y = self.player_position[1] +  self.speeds[1]*time_elapsed;
        let next_z = self.player_position[2] +  self.speeds[2]*time_elapsed;
        //println!("x {:?} y {:?} z {:?}",x_component,y_component,z_component);

        // 2. Get all obstacles in the zone (from play position to next position)
        // -> must take into account the speed direction
        // -> can be optimsed further (prune useless obstacles)
        let from_x_component = self.hitbox[0] - x_component;
        let from_y_component = self.hitbox[1] - y_component;
        let from_z_component = self.hitbox[2] - z_component;
        //println!("x {:?} y {:?} z {:?}",from_x_component,from_y_component,from_z_component);

        let mut from_block = [float_to_integer_coord(self.player_position[0]+from_x_component),float_to_integer_coord(self.player_position[1]+from_y_component),float_to_integer_coord(self.player_position[2]+from_z_component)];
        let mut to_block = [float_to_integer_coord(next_x+x_component),float_to_integer_coord(next_y+y_component),float_to_integer_coord(next_z+z_component)];
        //println!("From {:?} To {:?}",from_block,to_block);
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
        // Semm to work until here
        //println!("From {:?} To {:?}",from_block,to_block);
        // Really bad must fix that shit
        let mut obstacles = Vec::new();
        for x in  from_block[0]..to_block[0]+1 {
            for y in from_block[1]..to_block[1]+1 {
                for z in from_block[2]..to_block[2]+1 {
                    obstacles.extend(get_obstacles_in_block(&mut self.world,[x,y,z]));
                }
            }

        }
        
        //3. Check for collision along each face (axys)
        let mut next = [next_x,next_y,next_z];
        let mut actual = self.player_position;

        check_collisions(&mut actual, &mut next,self.hitbox,&mut self.speeds,time_elapsed,&mut obstacles,&mut self.on_ground);

        

        if next[1] < 0. {
            self.player_position = [0.,PERLIN_MAX_HEIGHT as f32 + 1.,0.];
        }
        else{
            self.player_position = next;
        }
        
        //self.player_position = next;
    }

    // Must be changed to be able to add blocks
    fn set_obstacle_target(&mut self){
        let x_direction = self.view_angle_horizontal.sin() * self.view_angle_vertical.cos();
        let y_direction = self.view_angle_vertical.sin();
        let z_direction = self.view_angle_horizontal.cos() * self.view_angle_vertical.cos();
        let dir = [x_direction,y_direction,z_direction];
        let view_point:[f32;3] = self.get_eyes_position();
        let blocks_intersec = ray_intersect_blocks(view_point, dir, INTERACTION_RADIUS);
        //println!("Pos {:?} Target {:?}",view_point,blocks_intersec);
        for block in blocks_intersec {
            // Normaly there is no way we target a block outside
            //println!("{:?}",block);
            match self.world.get_block(block){
                Some (b) => {
                    if b.block_type != 0 {
                        self.target_block = Some((b.block_type,block)); 
                        return;
                    }
                }
                None => {}
            }
        }
        self.target_block = None; 
    }

}

fn ray_intersect_blocks(origin: [f32; 3],normalized_dir: [f32; 3],max_distance: f32) -> Vec<[i64; 3]> {
    let mut blocks = Vec::new();

    // Current block (integer coordinates)
    let mut block = [
        origin[0].floor() as i64,
        origin[1].floor() as i64,
        origin[2].floor() as i64,
    ];

    // Determine the step direction
    let step = [
        if normalized_dir[0] > 0.0 { 1 } else { -1 },
        if normalized_dir[1] > 0.0 { 1 } else { -1 },
        if normalized_dir[2] > 0.0 { 1 } else { -1 },
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
    while t < max_distance {
        // Add the current block to the list
        blocks.push(block);

        // Determine which axis to step along
        if t_max[0] < t_max[1] && t_max[0] < t_max[2] {
            t = t_max[0];
            t_max[0] += t_delta[0];
            block[0] += step[0];
        } else if t_max[1] < t_max[2] {
            t = t_max[1];
            t_max[1] += t_delta[1];
            block[1] += step[1];
        } else {
            t = t_max[2];
            t_max[2] += t_delta[2];
            block[2] += step[2];
        }
    }
    blocks
}

fn check_collisions(actual:&mut[f32;3],next:&mut[f32;3],hitbox:[f32;3],speeds:&mut [f32;3],time_elapsed: f32, obstacles: &mut Vec<([f32;3],[f32;3])>,on_ground:&mut bool) {
    let mut collisions = Vec::new();
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

// Get the point (y,z) on the line at x defined by point1 and point2
fn get_point_on_line_by_x(point1:[f32;3],point2:[f32;3],x:f32) -> [f32;3]{
    let t = (x - point1[0])/(point2[0]-point1[0]);
    let y = point1[1] + t*(point2[1]-point1[1]);
    let z = point1[2] + t*(point2[2]-point1[2]);
    [x,y,z]
}

// Get the point (x,z) on the line at y defined by point1 and point2
fn get_point_on_line_by_y(point1:[f32;3],point2:[f32;3],y:f32) -> [f32;3]{
    let t = (y - point1[1])/(point2[1]-point1[1]);
    let x = point1[0] + t*(point2[0]-point1[0]);
    let z = point1[2] + t*(point2[2]-point1[2]);
    [x,y,z]
}

// Get the point (x,y) on the line at y defined by point1 and point2
fn get_point_on_line_by_z(point1:[f32;3],point2:[f32;3],z:f32) -> [f32;3]{
    let t = (z - point1[2])/(point2[2]-point1[2]);
    let x = point1[0] + t*(point2[0]-point1[0]);
    let y = point1[1] + t*(point2[1]-point1[1]);
    [x,y,z]
}

// From the world and 3 coordinates, get the obstacles in the block
// in the form [minx,miny,minz],[maxx,maxy,maxz] --> All obstacles are boxes
// Only get the opaque block for now
fn get_obstacles_in_block(world:&mut World,coord: [i64;3]) -> Vec<([f32;3],[f32;3])>{
    let mut obstacles = Vec::new();
    let block;
    match world.get_block(coord){
        None => {return obstacles;}
        Some(b) => {block = b;}
    }

    if block.block_type != 0 {
        obstacles.push(
            ([coord[0] as f32,coord[1] as f32,coord[2] as f32],[coord[0] as f32 + 1.,coord[1] as f32 + 1.,coord[2] as f32 + 1.])
        );
    }
    obstacles
}

fn float_to_integer_coord(value:f32) -> i64{
    return value.floor() as i64
}

/// A structure used to capture the the key/mouse information and their time when pressed to 
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
    /// Update the player position and the angles of view by the information stored in self
    /// Store mouse input information
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