//! By Henry Leclipteur.
//! 
//! This code is based in my previous project.
//! See [my gitlab](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Vulkan-Intro/readme.md#project-5-animations-and-features) for the description of the project.

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use nalgebra_glm::vec3;
use std::time::Instant;
use winit::event::DeviceEvent;

pub mod renderer;
pub mod inputs;
use renderer::*;
use inputs::*;


fn main() {

    let event_loop = EventLoop::new();
    let mut renderer = Renderer::new(&event_loop,100000);
    
    // Define the structure to handle inputs
    let mut inputhandle = InputHandle::default();

    // Speed of camera and player
    let speed = 5.;   // block per second
    let camera_sensitivity = 0.0015;

    // Player characteristics
    let mut angle_x: f32 = 0.;  // Right is positive
    let mut angle_y= 0.;   // Up is positive
    let mut player_position = vec3(0.0, 2., -2.);

    // For FPS calculation
    let mut now = Instant::now();
    let mut fps = 0;

    // For a block to blink (testing)
    let mut addrm = Instant::now();
    let mut rm = 0;

    // populate the world
    dirt_floor(&mut renderer,100);
    floor(&mut renderer, BlockTypes::LavaFlow, 1, 1,2,2,Orientation::NONE);
    floor(&mut renderer, BlockTypes::WaterFlow, 1, 1,-2,2,Orientation::NONE);
    dirt_tower(&mut renderer,-2,7,3);
    tower(&mut renderer,2,7,BlockTypes::Stone,9);
    tower(&mut renderer,4,5,BlockTypes::Sand,13);
    tower(&mut renderer,-4,5,BlockTypes::Gravel,11);
    tower(&mut renderer,5,2,BlockTypes::OakPlanks,30);
    tower(&mut renderer,-5,2,BlockTypes::Cobblestone,15);
    

    // Main loop
    event_loop.run(move |event, _, control_flow| match event {

        Event::WindowEvent {event: WindowEvent::CloseRequested,..} => {
            *control_flow = ControlFlow::Exit;
        }

        Event::WindowEvent {event: WindowEvent::Resized(_),..} => {
            renderer.window_resized();
        }

        Event::DeviceEvent {event: DeviceEvent::MouseMotion{delta: d, ..},..} => {
            inputhandle.handle_mouse(d);
        }

        Event::WindowEvent {event: WindowEvent::KeyboardInput { input: inp, .. },..} => {
            inputhandle.handle_key(inp);
        }
        // All event have been handled
        Event::MainEventsCleared => {

            // FPS calculation + Vertex Buffer State
            fps +=1 ;
            if now.elapsed().as_micros() >= 1000000{
                println!("FPS: {:?},  VERTICES: {}/{}",fps,renderer.vertex_nbr,renderer.vertex_max);
                fps = 0;
                now = Instant::now();
            }

            // Wait GPU for last frame
            renderer.wait_gpu();
            
            // Add/Remove a block every X millis
            if addrm.elapsed().as_millis()>= 1500{
                if rm == 0 {
                    rm = 1;
                    addrm = Instant::now();
                    add_full_block(&mut renderer, -2, 2, 6, BlockTypes::Cobblestone);
                }
                else{
                    rm = 0;
                    addrm = Instant::now();
                    rm_full_block(&mut renderer, -2, 2, 6, BlockTypes::Cobblestone);
                }
            }
            
            // Update the player characteristics
            inputhandle.handle_update(&mut player_position,speed,camera_sensitivity,&mut angle_x,&mut angle_y);

            // Update player characteristics of the rendering system
            renderer.set_view_postition(&player_position, &angle_x, &angle_y);
            
            // Submit GPU changes
            renderer.exec_gpu();

        }
        _ => (),
    });
}


fn rm_full_block(renderer: &mut Renderer, x:i64,y:i64,z:i64,blocktype: BlockTypes){
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::UP,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::DOWN,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::RIGHT,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::LEFT,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::BACKWARD,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::FORWARD,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
}

fn add_full_block(renderer: &mut Renderer, x:i64,y:i64,z:i64,blocktype: BlockTypes){
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::UP,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::DOWN,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::RIGHT,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::LEFT,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::BACKWARD,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::FORWARD,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
}

fn dirt_tower(renderer: &mut Renderer,x:i64,z:i64, height:u32){
    let blocktype = BlockTypes::Dirt;
    renderer.add_square(&Square{
        x:x,
        y:1,
        z:z,
        extent1: height,
        extent2: 0,
        direction:Direction::LEFT,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:1,
        z:z,
        extent1: height,
        extent2: 0,
        direction:Direction::RIGHT,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:1,
        z:z,
        extent1: 0,
        extent2: height,
        direction:Direction::BACKWARD,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:1,
        z:z,
        extent1: 0,
        extent2: height,
        direction:Direction::FORWARD,
        block_type: blocktype,
        orientation: Orientation::NONE
    });

    renderer.add_square(&Square{
        x:x,
        y:(height+1)as i64,
        z:z,
        extent1: 0,
        extent2: 0,
        direction:Direction::LEFT,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:(height+1)as i64,
        z:z,
        extent1: 0,
        extent2: 0,
        direction:Direction::RIGHT,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:(height+1)as i64,
        z:z,
        extent1: 0,
        extent2: 0,
        direction:Direction::BACKWARD,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:(height+1)as i64,
        z:z,
        extent1: 0,
        extent2: 0,
        direction:Direction::FORWARD,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:(height+1)as i64,
        z:z,
        extent1: 0,
        extent2: 0,
        direction:Direction::UP,
        block_type: BlockTypes::GrassBlockTop,
        orientation: Orientation::NONE
    });
}

fn tower(renderer: &mut Renderer,x:i64,z:i64,blocktype: BlockTypes, height:u32){
    
    renderer.add_square(&Square{
        x:x,
        y:1,
        z:z,
        extent1: height,
        extent2: 0,
        direction:Direction::LEFT,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:1,
        z:z,
        extent1: height,
        extent2: 0,
        direction:Direction::RIGHT,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:1,
        z:z,
        extent1: 0,
        extent2: height,
        direction:Direction::BACKWARD,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:x,
        y:1,
        z:z,
        extent1: 0,
        extent2: height,
        direction:Direction::FORWARD,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
    
    renderer.add_square(&Square{
        x:x,
        y:(height+1)as i64,
        z:z,
        extent1: 0,
        extent2: 0,
        direction:Direction::UP,
        block_type: blocktype,
        orientation: Orientation::NONE
    });
}

fn dirt_floor(renderer: &mut Renderer,length:u32){

    renderer.add_square(&Square{
        x:-(length as i64),
        y:0,
        z:-(length as i64),
        extent1: 2*length,
        extent2: 2*length,
        direction: Direction::UP,
        block_type: BlockTypes::GrassBlockTop,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:-(length as i64),
        y:0,
        z:-(length as i64),
        extent1: 2*length,
        extent2: 2*length,
        direction: Direction::DOWN,
        block_type: BlockTypes::Dirt,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:-(length as i64),
        y:0,
        z:-(length as i64),
        extent1: 0,
        extent2: 2*length,
        direction: Direction::LEFT,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:(length as i64),
        y:0,
        z:-(length as i64),
        extent1: 0,
        extent2: 2*length,
        direction: Direction::RIGHT,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:-(length as i64),
        y:0,
        z:(length as i64),
        extent1: 2*length,
        extent2: 0,
        direction: Direction::FORWARD,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE
    });
    renderer.add_square(&Square{
        x:-(length as i64),
        y:0,
        z:-(length as i64),
        extent1: 2*length,
        extent2: 0,
        direction: Direction::BACKWARD,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE
    });

}

fn floor(renderer: &mut Renderer,blocktype: BlockTypes, height:i64, length:u32,x:i64,z:i64,orientation:Orientation){
    renderer.add_square(&Square{
        x:x-(length as i64),
        y:height,
        z:z-(length as i64),
        extent1: 2*length,
        extent2: 2*length,
        direction: Direction::UP,
        block_type: blocktype,
        orientation: orientation
    });
    renderer.add_square(&Square{
        x:x-(length as i64),
        y:height,
        z:z-(length as i64),
        extent1: 2*length,
        extent2: 2*length,
        direction: Direction::DOWN,
        block_type: blocktype,
        orientation: orientation
    });
    renderer.add_square(&Square{
        x:x-(length as i64),
        y:height,
        z:z-(length as i64),
        extent1: 0,
        extent2: 2*length,
        direction: Direction::LEFT,
        block_type: blocktype,
        orientation: orientation
    });
    renderer.add_square(&Square{
        x:x+(length as i64),
        y:height,
        z:z-(length as i64),
        extent1: 0,
        extent2: 2*length,
        direction: Direction::RIGHT,
        block_type: blocktype,
        orientation: orientation
    });
    renderer.add_square(&Square{
        x:x-(length as i64),
        y:height,
        z:z+(length as i64),
        extent1: 2*length,
        extent2: 0,
        direction: Direction::FORWARD,
        block_type: blocktype,
        orientation: orientation
    });
    renderer.add_square(&Square{
        x:x-(length as i64),
        y:height,
        z:z-(length as i64),
        extent1: 2*length,
        extent2: 0,
        direction: Direction::BACKWARD,
        block_type: blocktype,
        orientation: orientation
    });

}
