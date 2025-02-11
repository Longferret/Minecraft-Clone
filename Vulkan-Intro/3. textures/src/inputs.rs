use winit::event::{ElementState, VirtualKeyCode};
use winit::event::KeyboardInput;
use std::time::Instant;
use  nalgebra_glm::TVec3;
use nalgebra_glm::two_pi;
pub struct KeyInfo{
    pub time: Instant,
    pub active: bool,
}
impl Default for KeyInfo {
    fn default() -> Self {
        Self {
            time: Instant::now(),  //irrelevant time when false
            active: false, // Default value for y
        }
    }
}
#[derive(Default)]
pub struct InputHandle{
    pub mouse_motion_x: f64,
    pub mouse_motion_y: f64,
    pub z_key: KeyInfo,
    pub q_key: KeyInfo,
    pub s_key: KeyInfo,
    pub d_key: KeyInfo,
    pub space_key: KeyInfo,
    pub shift_key: KeyInfo,
}

impl InputHandle {
    // Update the player position and the angles of view by the information stored in self
    pub fn handle_update(&mut self,player_position:&mut TVec3<f32>,speed:f32,sensitivity:f32,angle_x:&mut f32,angle_y:&mut f32){
        let composant_side = angle_x.sin();
        let composant_forward = angle_x.cos();
        let factor = 1000000.;
        if self.z_key.active { // forward
            player_position[2] += speed*composant_forward*(self.z_key.time.elapsed().as_micros() as f32)/factor;
            player_position[0] += speed*composant_side*(self.z_key.time.elapsed().as_micros() as f32)/factor;
            self.z_key.time = Instant::now();
        }
        if self.s_key.active { // backward
            player_position[2] -= speed*composant_forward*(self.s_key.time.elapsed().as_micros() as f32)/factor;
            player_position[0] -= speed*composant_side*(self.s_key.time.elapsed().as_micros() as f32)/factor;
            self.s_key.time = Instant::now();
        }
        if self.q_key.active{ // left
            player_position[0] -= speed*composant_forward*(self.q_key.time.elapsed().as_micros() as f32)/factor;
            player_position[2] += speed*composant_side*(self.q_key.time.elapsed().as_micros() as f32)/factor;
            self.q_key.time = Instant::now();
    
        }
        if self.d_key.active { // right
            player_position[0] += speed*composant_forward*(self.d_key.time.elapsed().as_micros() as f32)/factor;
            player_position[2] -= speed*composant_side*(self.d_key.time.elapsed().as_micros() as f32)/factor;
            self.d_key.time = Instant::now();
        }
        if self.shift_key.active { // down
            player_position[1] -= speed*(self.shift_key.time.elapsed().as_micros() as f32)/factor;
            self.shift_key.time = Instant::now();
        }
        if self.space_key.active { // up
            player_position[1] += speed*(self.space_key.time.elapsed().as_micros() as f32)/factor;
            self.space_key.time = Instant::now();
        }
        if self.mouse_motion_x!=0. {
            *angle_x+= (self.mouse_motion_x as f32)*sensitivity;
            if (*angle_x).abs()> two_pi(){
                *angle_x = *angle_x%two_pi::<f32>();
            }
            self.mouse_motion_x = 0.;
        }
        if self.mouse_motion_y!=0. {
            *angle_y-= (self.mouse_motion_y as f32)*sensitivity;
            if (*angle_y).abs()> 1.57{
                *angle_y = 1.57 * (*angle_y).signum();
            }
            self.mouse_motion_y = 0.;
        }
    }
    // Store mouse input information
    pub fn handle_mouse(&mut self,delta: (f64,f64)){
        self.mouse_motion_x+=delta.0;
        self.mouse_motion_y+=delta.1;
    }
    // Store key input information
    pub fn handle_key(&mut self,input: KeyboardInput){
        let key = input.virtual_keycode.unwrap();
        let now = Instant::now(); 
        match key {
            VirtualKeyCode::Q => {
                if input.state == ElementState::Pressed {
                    self.q_key.active = true;
                    self.q_key.time = now;
                } else {
                    self.q_key.active = false;
                }
            }
            VirtualKeyCode::S => {
                if input.state == ElementState::Pressed {
                    self.s_key.active = true;
                    self.s_key.time = now;
                } else {
                    self.s_key.active = false;
                }
            }
            VirtualKeyCode::Z => {
                if input.state == ElementState::Pressed {
                    self.z_key.active = true;
                    self.z_key.time = now;
                } else {
                    self.z_key.active = false;
                }
            }
            VirtualKeyCode::D => {
                if input.state == ElementState::Pressed {
                    self.d_key.active = true;
                    self.d_key.time = now;
                } else {
                    self.d_key.active = false;
                }
            }
            VirtualKeyCode::Space => {
                if input.state == ElementState::Pressed {
                    self.space_key.active = true;
                    self.space_key.time = now;
                } else {
                    self.space_key.active = false;
                }
            }
            VirtualKeyCode::LShift => {
                if input.state == ElementState::Pressed {
                    self.shift_key.active = true;
                    self.shift_key.time = now;
                } else {
                    self.shift_key.active = false;
                }
            }
            _ => {}
        }
    }

}

