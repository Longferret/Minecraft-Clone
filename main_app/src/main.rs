//! By Henry Leclipteur.
//! 
//! This is the version 0.3 of my minecraft clone.
//! See [my gitlab](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Minecraft-v0.3/readme.md) for the description of the project.

use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use std::time::Instant;
use winit::event::DeviceEvent;

use draw_element::*;
use renderer::*;
use game_core::*;

const MAX_VERTICES:usize = 1_000_000;

fn main() {
    // Speed of camera and player
    let speed = 10.;   // block per second
    let camera_sensitivity = 0.0015; 

    // Initialize the rendering system 
    let event_loop = EventLoop::new();
    let mut renderer = Renderer::new(&event_loop,MAX_VERTICES,"textures");

    // Initialize the game engine
    let (mut game_core,drawelements) = GameEngine::new(camera_sensitivity,speed);

    // Put init chunks in renderer
    engine_to_renderer(&mut renderer, &drawelements);

    // For FPS calculation
    let mut now = Instant::now();
    let mut fps = 0;
    
    // Main loop
    event_loop.run(move |event, _, control_flow| match event {

        Event::WindowEvent {event: WindowEvent::CloseRequested,..} => {
            *control_flow = ControlFlow::Exit;
        }

        Event::WindowEvent {event: WindowEvent::Resized(_),..} => {
            renderer.window_resized();
        }
        Event::DeviceEvent {event: DeviceEvent::MouseMotion{delta: d, ..},..} => {
            game_core.motion_event(d);
        }
        Event::WindowEvent {event: WindowEvent::MouseInput {state: s, button: b, .. }, ..} => {
            game_core.mouse_event(s,b);
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
            game_core.key_event(inp);
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
            let lag = Instant::now();
            let draw_elements= game_core.update(); 
            let tt = lag.elapsed().as_micros() as f32 / 1000.;
            if tt > 6. {
                println!("Lag Game Core: {} ms",tt);
            }
            let lag = Instant::now();
            engine_to_renderer(&mut renderer, &draw_elements);
            let tt = lag.elapsed().as_micros() as f32 / 1000.;
            if tt > 6. {
                println!("Lag buffering changes: {} ms by {} surface modified",tt,draw_elements.len());
            }

            let angles = game_core.get_camera_angles();
            // Update player characteristics of the rendering system
            renderer.set_view_postition(game_core.get_eyes_position().into(),angles.0, angles.1);
            
            // Wait GPU for last frame
            renderer.update();
        
            let lag = Instant::now();
            // Submit GPU changes
            let tt = lag.elapsed().as_micros() as f32 / 1000.;
            if tt > 6. {
                println!("Actual renderer lag: {} ms",tt);
            }
        }
        _ => (),
    });
}

/// This function send the received SurfaceElement from the GameEngine to the rendering system
fn engine_to_renderer(renderer: &mut Renderer,draw_elements: &Vec<(bool,SurfaceElement)>){
    for (to_be_added,surface_elem) in draw_elements {
        if *to_be_added {
            renderer.add_quad(surface_elem);
        }
        else{
            renderer.remove_quad(surface_elem);
        }
    }
}