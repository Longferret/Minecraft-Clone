//! By Henry Leclipteur.
//! 
//! This code is an amelioration of the code of the "Windowing" chapter at <http://vulkano.rs>.
//! 
//! See [my gitlab](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Vulkan-Intro/readme.md?ref_type=heads#project-1-triangle-and-input) for the description of the project.

use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::event::{Event, WindowEvent,ElementState,VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

mod utils;
use utils::*;

fn main() {
    // Search for a Vulkan library
    let library = vulkano::VulkanLibrary::new().expect("No local Vulkan library/DLL");

    // Create the envent loop, this is part of the winit crate.
    // It will handle the window on screen and the inputs inside that window.
    let event_loop = EventLoop::new();

    // Ask the required extensions for a device to draw with winit.
    let required_extensions = Surface::required_extensions(&event_loop);

    // Every Application that uses Vulkano need to create an Instance,
    // this is the base of vulkano.
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    ).expect("failed to create instance");

    // Create the actual window from the winit event loop.
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    // Create the surface on which vulkano will draw (through the Vulkan API)
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    // Create a DeviceExtensions structure and only put khr_swapchain as true.
    // This is the only extension our GPU needs to draw on the surface.
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    // Get the GPU we are going to use and a queue family index.
    // The queue family index is the index of the queue family we are going to use.
    // The queue family is a set of queues that have the same properties.
    // A queue is a GPU queue that can serve to transfer data,compute or make graphical operation
    // See the select_physical_device function (utils.rs) for more information.
    let (physical_device, queue_family_index) =
        select_physical_device(&instance, &surface, &device_extensions);

    println!("Device selected: {}", physical_device.properties().device_name);
    
    let mut i = 0;
    for family in physical_device.queue_family_properties() {
        if i == queue_family_index{
            println!("Queue Family selected:\n {:#?}", family);
        }
        i = i+1
    }

    // From our Physical Device and our Queue family index, we create a logical device.
    // It serves as a communication channel to the GPU.
    // We also get all the queue of the specified family index.
    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions, // new
            ..Default::default()
        },
    )
    .expect("failed to create device");

    // Simply select first queue, we will only use one
    let queue = queues.next().unwrap();

    // A swapchain is the link between the GPU and the surface.
    // It is a collection of images (often 2= double bufferring or 3= tripe buffering).
    // Those images will be in turn presented on the surface.
    // The GPU executes what we ask it to do and draw it on an image of the swapchain.
    // images are the images of the swapchain. 
    let (mut swapchain, images) = {
        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");

        let dimensions = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device.clone(), // specify the logical device
            surface,        // specify the surface
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,      // specify the number of image in the swapchain
                image_format,                               // specify the image format (ex: B8G8R8A8_UNORM)
                image_extent: dimensions.into(),            // specify the image dimensions
                image_usage: ImageUsage::COLOR_ATTACHMENT,  // specify that we will use the swapchain to draw on images
                composite_alpha,                            // specify the composite alpha -> if there will be transparency between the swapchain images (ex: Opaque)
                ..Default::default()                        // We use default for the rest
            },
        )
        .unwrap()
    };

    // A render pass represents the discrete steps in which rendering is done.
    // it is composed of:
    // - attachments which are inputs, outputs or intermediate values.
    // - one or more subpass which describes which data will be used in this pass and why (store, load, temporary value, ..)
    // - dependencies (not specified here)
    // see get_render_pass function for more detail (in utils.rs).
    let render_pass = get_render_pass(device.clone(), swapchain.clone());

    // A framebuffer is a container that binds image to render passes.
    // It does only store metadata.
    // see get_framebuffers function for more detail (in utils.rs).
    let framebuffers = get_framebuffers(&images, render_pass.clone());

    // We define here our vertices to draw our first triangle
    // see MyVertex struture for more detail (in utils.rs).
    let mut vertex1 = MyVertex {
        position: [-0.5, -0.5],
    };
    let mut vertex2 = MyVertex {
        position: [0.0, 0.5],
    };
    let mut vertex3 = MyVertex {
        position: [0.5, -0.25],
    };

    // Define the vertex buffers, they are used to store the vertices and pass them the GPU for drawing.
    // We create as many vertex buffer as there is image in the swapchain to be able to modify the content of one vertex buffer,
    // while the other is used by the GPU for rendering.
    let mut vertex_buffers = Vec::new();
    for _i in 0..images.len() {
        // Generic allocation of memory, It is an abstraction of Vulkano.
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        vertex_buffers.push(
            // Create a new buffer
            Buffer::from_iter(
                memory_allocator, // specify how to allocate memory
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,  // specify the buffer usage
                    ..Default::default()                // set to default
                },
                AllocationCreateInfo {
                    // PREFER_DEVICE + HOST_SEQUENTIAL_WRITE -->  continuous update to GPU 
                    // - used for direct ressource acces like vertices,  image, ..
                    // PREFER_DEVICE --> only visible from GPU (hard acces to CPU)
                    // - used for texture, intermediary buffers, ..
                    // PREFER_HOST + HOST_SEQUENTIAL_WRITE --> only visible to CPU
                    // - used for staging buffers
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE 
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vec![vertex1, vertex2, vertex3], // initial data to put in the buffer
            )
            .unwrap()
        );
    }

    // Definition of the vertex (vs) and fragment shaders (fs).
    // see the vs and fs module for more details (in utils.rs).
    let vs = vs::load(device.clone()).expect("failed to create shader module");
    let fs = fs::load(device.clone()).expect("failed to create shader module");

    // Maps the NPC coordinate to real window coordinate
    // In vulkan, the Normalized Device Coordinate goes from -1 to 1 , in window coordinate it goes from 0 to width / 0 to height.
    let mut viewport = Viewport {
        offset: [0.0, 0.0],                 // starting position
        extent: window.inner_size().into(), // size of the window
        depth_range: 0.0..=1.0,             // Not used 
    };

    // It defines the entire sequence of operations to draw on the surface
    // see get_pipeline for more info (in utils.rs)
    let pipeline = get_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    // Allocator of command buffer
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    // Create the command buffers.
    // The command buffers are the actual commands that the GPU will execute, It contains all the the information to draw on the screen.
    // There are as many command buffer that there is image in the swapchain, for the GPU to able to render 2 frames at the same time for example.
    // see the get_command_buffers for more details (in utils.rs).
    let mut command_buffers = get_command_buffers(
        &command_buffer_allocator,
        &queue,
        &pipeline,
        &framebuffers,
        &vertex_buffers,
    );

    // Used as notification in the event loop
    let mut window_resized = false;
    let mut recreate_swapchain = false;

    
    let frames_in_flight = images.len();
    // A GPU fence is a synchronization primitive (=future in rust)
    // A future represents a value that might not be available yet because the computation to produce it is still in progress
    // It is used for the CPU to wait the GPU in order update the vertex buffer for example.
    // I will use the terms:
    // -->  gpu-future for the future given by the GPU (when can we resubmit the commandbuffer ?)
    // Note: it is also possible to use different command buffer but it is less efficient.
    // -->  swapchain-future for the future given by swapchain (when is the image i in the swapchain is available ?)
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    // Used as notification in the event loop
    let mut q_key = false;
    let mut z_key = false;
    let mut s_key = false;
    let mut d_key = false;

    // Speed in pixel per frame
    let move_speed = 0.01;
    
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            window_resized = true;
        }
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput{
                input : inp,
                ..},
            ..
        } => {
            let key = inp.virtual_keycode.unwrap();
            match key {
                VirtualKeyCode::Q => {
                    if inp.state == ElementState::Pressed {
                        q_key = true;
                    }
                    else{
                        q_key = false;
                    }
                }
                VirtualKeyCode::S => {
                    if inp.state == ElementState::Pressed {
                        s_key = true;
                    }
                    else{
                        s_key = false;
                    }
                }
                VirtualKeyCode::Z => {
                    if inp.state == ElementState::Pressed {
                        z_key = true;
                    }
                    else{
                        z_key = false;
                    }
                }
                VirtualKeyCode::D => {
                    if inp.state == ElementState::Pressed {
                        d_key = true;
                    }
                    else{
                        d_key = false;
                    }
                }
                _ => {}
            }
        }
        // All event have been handled
        Event::MainEventsCleared => {
            // Recreate swapchain
            if window_resized || recreate_swapchain {
                recreate_swapchain = false;

                let new_dimensions = window.inner_size();

                let (new_swapchain, new_images) = swapchain
                    .recreate(SwapchainCreateInfo {
                        image_extent: new_dimensions.into(),
                        ..swapchain.create_info()
                    })
                    .expect("failed to recreate swapchain");

                swapchain = new_swapchain;
                let new_framebuffers = get_framebuffers(&new_images, render_pass.clone());
                // Recreate the command buffer and the viewport
                if window_resized {
                    window_resized = false;

                    viewport.extent = new_dimensions.into();
                    let new_pipeline = get_pipeline(
                        device.clone(),
                        vs.clone(),
                        fs.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                    );
                    command_buffers = get_command_buffers(
                        &command_buffer_allocator,
                        &queue,
                        &new_pipeline,
                        &new_framebuffers,
                        &vertex_buffers,
                    );
                }
            }
            // Get the next image in the swapchain 
            // We get:
            // -> the image index in the swapchain
            // -> suboptimal, a bool that indicates if the swapchain works optimaly or  not optimaly
            // -> aquire_future, a swap-chain future representing the point in time when the image acquisition is complete and the image is ready for use in the rendering pipeline.
            let (image_i, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None)
                    .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

            if suboptimal {
                recreate_swapchain = true;
            }
            // Wait for the GPU-future (=fence),(normally this would be the oldest fence)
            if let Some(image_fence) = &fences[image_i as usize] {
                image_fence.wait(None).unwrap(); 
            }
            // Modify the Vertex buffer (we can acces and modify the vertex buffer since GPU has finished work of the commandbuffer)
            let result = vertex_buffers[image_i as usize].write();
            match result {
                Ok(mut value) => {
                    value[0] = vertex1;
                    value[1] = vertex2;
                    value[2] = vertex3;
                    let mut offset = 0.;
                    if q_key {
                        offset-= move_speed;
                    }
                    if d_key {
                        offset+= move_speed;
                    }
                    vertex1.position[0] += offset;
                    vertex2.position[0] += offset;
                    vertex3.position[0] += offset;
                    offset = 0.;
                    if z_key {
                        offset-= move_speed;
                    }
                    if s_key {
                        offset+= move_speed;
                    }
                    vertex1.position[1] += offset;
                    vertex2.position[1] += offset;
                    vertex3.position[1] += offset;
                    
                }
                Err(_e) => {
                    // Handle the error case
                    eprintln!("Access Refused");
                }
            } 
            
            // Get the previous GPU-future 
            // if there is none, create one
            let previous_future = match fences[previous_fence_i as usize].clone() {
                // Create a NowFuture
                None => {
                    let mut now = sync::now(device.clone());
                    now.cleanup_finished();

                    now.boxed()
                }
                // Use the existing GPU-future
                Some(fence) => fence.boxed(),
            };

            let future = previous_future
                // Combine the GPU-future of last GPU execution with the swapchain-future image acquisition (both must finish for next operation)
                .join(acquire_future) 
                // Execute the command of the command buffer for frame i on the GPU
                .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
                .unwrap()
                // Present the result to the swapchain
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                )
                // Flush the operation and get a future (to be able to wait -> Line 386)
                .then_signal_fence_and_flush();

            // Verify that the gpu-future created succeded and put it in the fence array
            fences[image_i as usize] = match future.map_err(Validated::unwrap) {
                Ok(value) => Some(Arc::new(value)),
                Err(VulkanError::OutOfDate) => {
                    recreate_swapchain = true;
                    None
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                    None
                }
            };
            // Change index (of swapchain image)
            previous_fence_i = image_i;

        }
        _ => (),
    });
}