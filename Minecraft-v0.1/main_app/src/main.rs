//! By Henry Leclipteur.
//! 
//! This is the version 0.1 of my minecraft clone.
//! See [my gitlab](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Minecraft-v0.1/readme.md) for the description of the project.
use noise::Perlin;
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use std::collections::HashMap;
use std::time::Instant;
use winit::event::DeviceEvent;

use renderer::*;
use physics::*;

fn main() {

    let event_loop = EventLoop::new();
    let mut renderer = Renderer::new(&event_loop,600000,"textures");
    

    // Speed of camera and player
    let speed = 10.;   // block per second
    let camera_sensitivity = 0.0015;
    let player_position =[0., 100., 0.];


    // For FPS calculation
    let mut now = Instant::now();
    let mut fps = 0;
    // Define the structure to handle physics
    let mut game_engine;
    //let init_blocks;
    game_engine = GameEngine::new(camera_sensitivity,speed,player_position);
    //physics_to_renderer_add(&mut renderer, &init_blocks);
    // Main loop
    event_loop.run(move |event, _, control_flow| match event {

        Event::WindowEvent {event: WindowEvent::CloseRequested,..} => {
            *control_flow = ControlFlow::Exit;
        }

        Event::WindowEvent {event: WindowEvent::Resized(_),..} => {
            renderer.window_resized();
        }
        Event::DeviceEvent {event: DeviceEvent::MouseMotion{delta: d, ..},..} => {
            game_engine.motion_event(d);
        }
        Event::WindowEvent {event: WindowEvent::MouseInput {state: s, button: b, .. }, ..} => {
            game_engine.mouse_event(s,b);
        }

        Event::WindowEvent {event: WindowEvent::KeyboardInput { input: inp, .. },..} => {
            match inp.virtual_keycode {
                Some(value) => {
                    if value == VirtualKeyCode::Escape {
                        *control_flow = ControlFlow::Exit;
                    }
                }
                None => {
                    return;
                }
            }
            game_engine.key_event(inp);
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

            // Update the world
            let block_to_remove;
            let block_to_add;
            (block_to_add,block_to_remove) = game_engine.update(); 
            let lag = Instant::now();
            if !block_to_remove.is_empty() {
                physics_to_renderer_rm(&mut renderer, block_to_remove);
                //println!("{:?}",block_to_remove);
            }
            if !block_to_add.is_empty() {
                physics_to_renderer_add(&mut renderer, block_to_add);
            }
            let tt = lag.elapsed().as_millis();
            if tt > 10 {
                println!("Lag renderer: {} ms by {} vertices added & {} vertices removed",tt,block_to_add.len()*36,block_to_remove.len()*36);
            }

            // Wait GPU for last frame
            renderer.wait_gpu();

            let angles = game_engine.get_camera_angles();
            // Update player characteristics of the rendering system
            renderer.set_view_postition(&game_engine.get_eyes_position().into(),&angles.0, &angles.1);
            
            // Submit GPU changes
            renderer.exec_gpu();

        }
        _ => (),
    });
}

fn physics_to_renderer_add(renderer: &mut Renderer,blocks: &Vec<(u8,[i64;3])>){
    for b in blocks {
        match b.0 {
            1 => {
                add_full_block(renderer, b.1[0], b.1[1], b.1[2],BlockTypes::Stone);
            }
            2 => {
                add_full_block(renderer, b.1[0], b.1[1], b.1[2],BlockTypes::Dirt);
            }
            3 => {
                add_full_gblock(renderer, b.1[0], b.1[1], b.1[2]);
            }
            _ => {

            }
        }
    }
}

fn physics_to_renderer_rm(renderer: &mut Renderer,blocks: &Vec<(u8,[i64;3])>){
    for b in blocks {
        match b.0 {
            1 => {
                rm_full_block(renderer, b.1[0], b.1[1], b.1[2],BlockTypes::Stone);
            }
            2 => {
                rm_full_block(renderer, b.1[0], b.1[1], b.1[2],BlockTypes::Dirt);
            }
            3 => {
                rm_full_gblock(renderer, b.1[0], b.1[1], b.1[2]);
            }
            _ => {

            }
        }
    }
}

fn rm_full_gblock(renderer: &mut Renderer, x:i64,y:i64,z:i64){
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::UP,
        block_type: BlockTypes::GrassBlockTop,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::DOWN,
        block_type: BlockTypes::Dirt,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::RIGHT,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::LEFT,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::BACKWARD,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::FORWARD,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE,
        is_interior: false
    });
}

fn add_full_gblock(renderer: &mut Renderer, x:i64,y:i64,z:i64){
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::UP,
        block_type: BlockTypes::GrassBlockTop,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::DOWN,
        block_type: BlockTypes::Dirt,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::RIGHT,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::LEFT,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::BACKWARD,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::FORWARD,
        block_type: BlockTypes::GrassBlockSide,
        orientation: Orientation::NONE,
        is_interior: false
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
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::DOWN,
        block_type: blocktype,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::RIGHT,
        block_type: blocktype,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::LEFT,
        block_type: blocktype,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::BACKWARD,
        block_type: blocktype,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::FORWARD,
        block_type: blocktype,
        orientation: Orientation::NONE,
        is_interior: false
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
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::DOWN,
        block_type: blocktype,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::RIGHT,
        block_type: blocktype,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::LEFT,
        block_type: blocktype,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::BACKWARD,
        block_type: blocktype,
        orientation: Orientation::NONE,
        is_interior: false
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        extent1:0,
        extent2:0,
        direction:Direction::FORWARD,
        block_type: blocktype,
        orientation: Orientation::NONE,
        is_interior: false
    });
}

