//! By Henry Leclipteur.
//! 
//! This code is based in my previous project.
//! See [my gitlab](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Vulkan-Intro/readme.md?ref_type=heads#project-2-square-and-perspective) for the description of the project.
//!
//! This code is inspired by [this tutorial](https://github.com/taidaesal/vulkano_tutorial/tree/master).

mod utils;
mod inputs;
use utils::*;
use std::time::Instant;
use inputs::*;


fn main() {

    // Define the MVP 
    // see MVP structure for more details (in utils.rs).
    let mut mvp = MVP::new();
    mvp.view = look_at(
        &vec3(0.0, 0.0, 0.0),
        &vec3(0.0, 0.0, 0.0),
        &vec3(0.0, 1.0, 0.0),
    );

    let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let event_loop = EventLoop::new();

    let required_extensions = Surface::required_extensions(&event_loop);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("failed to create instance");

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) =
        select_physical_device(&instance, &surface, &device_extensions);

    println!(
        "Device selected: {}",
        physical_device.properties().device_name
    );

    let mut i = 0;
    for family in physical_device.queue_family_properties() {
        if i == queue_family_index {
            println!("Queue Family selected:\n {:#?}", family);
        }
        i = i + 1
    }

     let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions, 
            ..Default::default()
        },
    )
    .expect("failed to create device");
    let queue = queues.next().unwrap();

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
        let image_extent: [u32; 2] = dimensions.into();

        let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
        // Definition of the projection component (from the player to all its environment)
        mvp.projection = perspective(aspect_ratio, half_pi(), 0.01, 100.0);

        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent,
                image_usage: caps.supported_usage_flags,
                composite_alpha,
                present_mode: PresentMode::Fifo,
                ..Default::default()
            },
        )
        .unwrap()
    };
    println!("Number of images in the swapchain: {:?}", images.len());

    // Changes in render pass, addition of depth testing.
    // see get_render_pass function for more detail (in utils.rs).
    let render_pass = get_render_pass(device.clone(), swapchain.clone());
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    // Changes in framebuffer, we now allocate a depth-buffer to the GPU for it to draw overlapping triangles correclty.
    // see get_framebuffers function for more detail (in utils.rs).
    let framebuffers = get_framebuffers(&memory_allocator,&images, render_pass.clone());


    //  Definition of a lot of cubes
    let mut my_square1 = create_square(-3.,10.,-0.,1.);
    let mut trianglenbr = 0;
    for x in -100..100 {
        for z in -100..100 {
            let mut my_square2 = create_square(x as f32,0.,z as f32,1.);
            my_square1.append(&mut my_square2);
            trianglenbr += 1;
        }
    }
    for y in 1..20{
        let mut my_square2 = create_square(2.,y as f32,2.,1.);
        my_square1.append(&mut my_square2);
        trianglenbr += 1;
    }
    println!("Rendered triangles per frame :{:?}",trianglenbr);
    

    let mut vertex_buffers = Vec::new();
    // Define the new uniform buffer that we will use as inputs in all the shaders to 
    // transform the "real" cube game coordinates into 2D vulkan coordinates.
    // We create as many uniform buffer as there are images to be able to wirte in one
    // while the other is used by the GPU.
    let mut uniform_buffers = Vec::new();
    for _i in 0..images.len() {
        vertex_buffers.push(
            Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                my_square1.clone(),
            )
            .unwrap(),
        );
        // Define the MVP_DATA variable in the vertex buffer
        // from our MVP structure
        let data = vs::MVP_Data {
            model: mvp.model.into(),
            view: mvp.model.into(),
            projection: mvp.projection.into(),
        };
        uniform_buffers.push(
            Buffer::from_data(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER, // buffer that will be visible by all shader invocation
                    ..Default::default()
                },
                AllocationCreateInfo {
                    // Same as vertex buffer, we want the CPU to update it frequently (on mouse moves)
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                data,
            )
            .unwrap(),
        );
    }

    let vs = vs::load(device.clone()).expect("failed to create shader module");
    let fs = fs::load(device.clone()).expect("failed to create shader module");

    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: window.inner_size().into(),
        depth_range: 0.0..=1.0,
    };
    // Changes in pipeline, addition of depth checking
    // see get_pipeline function for more detail (in utils.rs). 
    let pipeline = get_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    // To upload the vertices to the GPU, we simply had to give to the command buffer the vertex buffer
    // but it is not the case for uniform data (data that will be used by all shaders).
    // To make the MVP data visible by all shaders we need to bind a descriptor set to the command buffer.
    // Here is the first step: the definition of the descriptor set.

    // Get the first layout we defined in the pipeline
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    // We create as many descriptor sets as there are images in the swapchain (to write while GPU render).
    let mut descriptor_sets = Vec::new();
    for i in 0..images.len() {
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        );
        descriptor_sets.push(
            // Persistent means that the DescriptorSet will be reused multiple times
            PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                layout.clone(),
                // Associate our uniform buffer (where CPU writes MVP data)
                // to the descriptor set
                [WriteDescriptorSet::buffer(0, uniform_buffers[i].clone())],
                [],
            )
            .unwrap(),
        );
    }

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());
    // Changes in the command buffer, now we give it ou descriptor sets to bind them.
    // see get_command_buffers function for more detail (in utils.rs).
    let mut command_buffers = get_command_buffers(
        &command_buffer_allocator,
        &queue,
        &pipeline,
        &framebuffers,
        &vertex_buffers,
        &descriptor_sets,
    );

    // New structure to better handle the inputs
    // see inputs.rs for more details
    let mut inputhandle = InputHandle::default();

    // keep track of the player position (in game coordinate)
    let mut player_position = vec3(0.0, 2.0, 0.1);

    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    // Speed of camera and player
    let speed = 5.;   // block per second
    let camera_sensibility = 0.002;

    // Actual angle of vision
    let mut angle_x: f32 = 0.;  // Right is positive
    let mut angle_y= 0.;   // Up is positive

    // For FPS calculation
    let mut now = Instant::now();
    let mut fps = 0;

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
        // Mouse input Event
        Event::DeviceEvent { 
            event: DeviceEvent::MouseMotion{delta: d, ..},
            ..
        } => {
            inputhandle.handle_mouse(d);
        }
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput { input: inp, .. },
            ..
        } => {
            inputhandle.handle_key(inp);
        }
        Event::MainEventsCleared => {
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
                let new_framebuffers = get_framebuffers(&memory_allocator,&new_images, render_pass.clone());

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
                        &descriptor_sets,
                    );
                }
            }

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

            if let Some(image_fence) = &fences[image_i as usize] {
                image_fence.wait(None).unwrap();
            }
            // Display FPS
            fps +=1 ;
            if now.elapsed().as_micros() >= 1000000{
                println!("FPS: {:?}",fps);
                fps = 0;
                now = Instant::now();
            }
            // Here we can modify the MVP in the uniform buffer (if the player moved or the camera moved) 
            let result = uniform_buffers[image_i as usize].write();
            match result {
                Ok(mut value) => {
                    // Update the variables: player_position, angle_x, angle_y 
                    inputhandle.handle_update(&mut player_position,speed,camera_sensibility,&mut angle_x,&mut angle_y);

                    // Change the view and eye component of MVP
                    let looking_towards = player_position + vec3(angle_x.sin()*angle_y.cos(),angle_y.sin(),angle_x.cos()*angle_y.cos());
                    mvp.view =  look_at(
                        &player_position,
                        &looking_towards,
                        &vec3(0.0, 1.0, 0.0),
                    );
                    // Put the value inside the uniform buffer
                    value.view = look_at(
                        &player_position,
                        &looking_towards,
                        &vec3(0.0, 1.0, 0.0),
                    ).into();
                }
                Err(_e) => {
                    eprintln!("Access Refused to the MVP buffer");
                }
            }
            // Modify Vertices
            let result = vertex_buffers[image_i as usize].write();
            match result {
                Ok(_value) => {
                    // Input/Change vertices
                    // Nothing here, static environment
                }
                Err(_e) => {
                    eprintln!("Access Refused to the vertex buffer");
                }
            }

            let previous_future = match fences[previous_fence_i as usize].clone() {
                None => {
                    let mut now = sync::now(device.clone());
                    now.cleanup_finished();

                    now.boxed()
                }
                Some(fence) => fence.boxed(),
            };

            let future = previous_future
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                )
                .then_signal_fence_and_flush();

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

            previous_fence_i = image_i;
        }
        _ => (),
    });
}
