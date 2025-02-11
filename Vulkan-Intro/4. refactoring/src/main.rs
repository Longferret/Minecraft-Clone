//! By Henry Leclipteur.
//! 
//! This code is based in my previous project.
//! See [my gitlab](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Vulkan-Intro/readme.md?ref_type=heads#project-4-refactoring) for the description of the project.
//!
//! This code is inspired by [this tutorial](https://github.com/taidaesal/vulkano_tutorial/tree/master).

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
    let mut renderer = Renderer::new(&event_loop,1000000);
    
    // Define the structure to handle inputs
    let mut inputhandle = InputHandle::default();



    // Speed of camera and player
    let speed = 5.;   // block per second
    let camera_sensitivity = 0.002;

    // Player characteristics
    let mut angle_x: f32 = 0.;  // Right is positive
    let mut angle_y= 0.;   // Up is positive
    let mut player_position = vec3(0.0, 2.0, -2.);

    // For FPS calculation
    let mut now = Instant::now();
    let mut fps = 0;

    // For a block to blink (testing)
    let mut addrm = Instant::now();
    let mut rm = 0;

    // populate the world
    floor(&mut renderer);
    tourdecaillou_ou_autre(&mut renderer,0,7,BlockTypes::Stone,9);
    tourdecaillou_ou_autre(&mut renderer,3,5,BlockTypes::Sand,13);
    tourdecaillou_ou_autre(&mut renderer,-3,5,BlockTypes::Gravel,11);
    tourdecaillou_ou_autre(&mut renderer,5,2,BlockTypes::OakPlanks,30);
    tourdecaillou_ou_autre(&mut renderer,-5,2,BlockTypes::Cobblestone,15);
    tourdecaca(&mut renderer,-1,0,3);

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
            renderer.wait_gpu();

            // Add/Remove a block every X millis
            if addrm.elapsed().as_millis()>= 1500{
                if rm == 0 {
                    rm = 1;
                    addrm = Instant::now();
                    add_full_block(&mut renderer, 3, 2, 3, BlockTypes::Cobblestone);
                }
                else{
                    rm = 0;
                    addrm = Instant::now();
                    rm_full_block(&mut renderer, 3, 2, 3, BlockTypes::Cobblestone);
                }
            }
            {
                // Update the player characteristics
                inputhandle.handle_update(&mut player_position,speed,camera_sensitivity,&mut angle_x,&mut angle_y);
                // Give the player characteristics to the rendering system
                renderer.set_view_postition(&player_position, &angle_x, &angle_y);
            }
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
        direction:Direction::UP,
        block_type: blocktype
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        direction:Direction::DOWN,
        block_type: blocktype
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        direction:Direction::RIGHT,
        block_type: blocktype
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        direction:Direction::LEFT,
        block_type: blocktype
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        direction:Direction::BACKWARD,
        block_type: blocktype
    });
    renderer.remove_square(&Square{
        x:x,
        y:y,
        z:z,
        direction:Direction::FORWARD,
        block_type: blocktype
    });
}

fn add_full_block(renderer: &mut Renderer, x:i64,y:i64,z:i64,blocktype: BlockTypes){
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        direction:Direction::UP,
        block_type: blocktype
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        direction:Direction::DOWN,
        block_type: blocktype
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        direction:Direction::RIGHT,
        block_type: blocktype
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        direction:Direction::LEFT,
        block_type: blocktype
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        direction:Direction::BACKWARD,
        block_type: blocktype
    });
    renderer.add_square(&Square{
        x:x,
        y:y,
        z:z,
        direction:Direction::FORWARD,
        block_type: blocktype
    });
}

fn floor(renderer: &mut Renderer){
    let length = 50;
    for i in -length..length {
        for j in -length..length {
            renderer.add_square(&Square{
                x:i,
                y:-1,
                z:j,
                direction:Direction::UP,
                block_type: BlockTypes::GrassBlockTop
            });
            renderer.add_square(&Square{
                x:i,
                y:-1,
                z:j,
                direction:Direction::DOWN,
                block_type: BlockTypes::Dirt
            });
            if j == length-1 {
                renderer.add_square(&Square{
                    x:i,
                    y:-1,
                    z:j,
                    direction:Direction::FORWARD,
                    block_type: BlockTypes::GrassBlockSide
                });
            }
            if j == -length {
                renderer.add_square(&Square{
                    x:i,
                    y:-1,
                    z:j,
                    direction:Direction::BACKWARD,
                    block_type: BlockTypes::GrassBlockSide
                });
            }
            if i == length-1 {
                renderer.add_square(&Square{
                    x:i,
                    y:-1,
                    z:j,
                    direction:Direction::RIGHT,
                    block_type: BlockTypes::GrassBlockSide
                });
            }
            if i == -length {
                renderer.add_square(&Square{
                    x:i,
                    y:-1,
                    z:j,
                    direction:Direction::LEFT,
                    block_type: BlockTypes::GrassBlockSide
                });
            }
        }
    }
}

fn tourdecaca(renderer: &mut Renderer,x:i64,z:i64, height:i64){
    let blocktype = BlockTypes::Dirt;
    for h in 0..height-1 {
        renderer.add_square(&Square{
            x:x,
            y:h,
            z:z,
            direction:Direction::LEFT,
            block_type: blocktype
        });
        renderer.add_square(&Square{
            x:x,
            y:h,
            z:z,
            direction:Direction::RIGHT,
            block_type: blocktype
        });
        renderer.add_square(&Square{
            x:x,
            y:h,
            z:z,
            direction:Direction::BACKWARD,
            block_type: blocktype
        });
        renderer.add_square(&Square{
            x:x,
            y:h,
            z:z,
            direction:Direction::FORWARD,
            block_type: blocktype
        });
    }
    renderer.add_square(&Square{
        x:x,
        y:height-1,
        z:z,
        direction:Direction::LEFT,
        block_type: BlockTypes::GrassBlockSide
    });
    renderer.add_square(&Square{
        x:x,
        y:height-1,
        z:z,
        direction:Direction::RIGHT,
        block_type: BlockTypes::GrassBlockSide
    });
    renderer.add_square(&Square{
        x:x,
        y:height-1,
        z:z,
        direction:Direction::BACKWARD,
        block_type: BlockTypes::GrassBlockSide
    });
    renderer.add_square(&Square{
        x:x,
        y:height-1,
        z:z,
        direction:Direction::FORWARD,
        block_type: BlockTypes::GrassBlockSide
    });
    renderer.add_square(&Square{
        x:x,
        y:height-1,
        z:z,
        direction:Direction::UP,
        block_type: BlockTypes::GrassBlockTop
    });
}

fn tourdecaillou_ou_autre(renderer: &mut Renderer,x:i64,z:i64,blocktype: BlockTypes, height:i64){
    
    for h in 0..height {
        renderer.add_square(&Square{
            x:x,
            y:h,
            z:z,
            direction:Direction::LEFT,
            block_type: blocktype
        });
        renderer.add_square(&Square{
            x:x,
            y:h,
            z:z,
            direction:Direction::RIGHT,
            block_type: blocktype
        });
        renderer.add_square(&Square{
            x:x,
            y:h,
            z:z,
            direction:Direction::BACKWARD,
            block_type: blocktype
        });
        renderer.add_square(&Square{
            x:x,
            y:h,
            z:z,
            direction:Direction::FORWARD,
            block_type: blocktype
        });
    }
    renderer.add_square(&Square{
        x:x,
        y:height-1,
        z:z,
        direction:Direction::UP,
        block_type: blocktype
    });
}