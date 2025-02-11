//! By Henry Leclipteur.
//! 
//! This code is based in my previous project.
//! See [my gitlab](https://gitlab.uliege.be/Henry.Leclipteur/minecraft-rust-clone/-/blob/main/Vulkan-Intro/readme.md?ref_type=heads#project-3-textures-and-simple-culling) for the description of the project.
//!
//! This code is inspired by [this tutorial](https://github.com/taidaesal/vulkano_tutorial/tree/master).

mod utils;
mod inputs;
use utils::*;
use std::time::Instant;
use inputs::*;
use png;
use std::io::Cursor;


fn main() {

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

    let render_pass = get_render_pass(device.clone(), swapchain.clone());
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let framebuffers = get_framebuffers(&memory_allocator,&images, render_pass.clone());


    let mut my_square1 = create_square(-3.,10.,-0.,1.);
    for x in -100..100 {
        for z in -100..100 {
            for y in 1..2{
                let mut my_square2 = create_square(x as f32,-y as f32,z as f32,1.);
                my_square1.append(&mut my_square2);
            }
        }
    }
    for y in 1..20{
        let mut my_square2 = create_square(2.,y as f32,2.,1.);
        my_square1.append(&mut my_square2);
    }

    println!("Rendered triangles per frame :{:?}",my_square1.len());

    let mut vertex_buffers = Vec::new();
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
                    memory_type_filter: 
                        MemoryTypeFilter::PREFER_DEVICE |
                        MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                my_square1.clone(),
            )
            .unwrap(),
        );
        let data = vs::MVP_Data {
            model: mvp.model.into(),
            view: mvp.model.into(),
            projection: mvp.projection.into(),
        };
        uniform_buffers.push(
            Buffer::from_data(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: 
                        MemoryTypeFilter::PREFER_DEVICE |
                        MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
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
    // Changes in the get_pipeline to account for change in descriptor sets
    // see utils.rs
    let pipeline = get_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    // Texture upload start -----------------------------------------------------------
    // The goal is upload a texture to the GPU for it to render the texture at all frames.

    // The first step is to get our texture (a png file) as a string of bytes an its dimensions
    // To do so we use the png crate
    let (image_data, image_dimensions) = {
        // Creation of the basic stuff to read a png 
        let png_bytes = include_bytes!("../textures/stone.png").to_vec();
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();
        
        let image_dimensions = [info.width, info.height, 1]; // 1 is depth for 2D images
        
        // Determine the number of bytes per pixel based on the color type
        let bytes_per_pixel = match info.color_type {
            png::ColorType::Grayscale => 1,
            png::ColorType::Rgb => 3,
            png::ColorType::Indexed => 1, 
            png::ColorType::GrayscaleAlpha => 2,
            png::ColorType::Rgba => 4,
        };

        // Panic if we don't use the RGBA format (it is hardcoded later in the code)
        if bytes_per_pixel !=4 {
            panic!("The png is not RGBA color type.");
        }

        // Calculate the total size of the image data buffer
        let total_bytes = (info.width * info.height * bytes_per_pixel) as usize;
    
        let mut image_data = vec![0; total_bytes];
    
        // Read the image frame into the buffer
        reader.next_frame(&mut image_data).unwrap();
    
        (image_data, image_dimensions)
    };

    // The second step is to create the image that will be accessible by the shaders in the GPU.
    // - Here is the image information for the image
    let texture_create_info = ImageCreateInfo {
        format: Format::R8G8B8A8_UNORM, // hardcoded format RGBA
        extent: image_dimensions,       // the image dimension of our png
        // SAMPLED means that this image will be used in shader as a texture
        // TRANSFER_DST means that the image will receive data
        usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
        mip_levels: 1,                  // the image only have 1 level 
        array_layers: 1,                // some times multiple image can be store in an array, here we have 1 image
        ..Default::default()
    };
    // - Here is the allocation information for the image
    let texture_allocation_info = AllocationCreateInfo {
        // PREFER_DEVICE means that the GPU can acces it really fast, but the CPU can't acces this image directly.
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, 
        ..Default::default()
    };
    // - Here is the previous iformation to create the image
    let texture_image = Image::new(
        memory_allocator.clone(),
        texture_create_info,
        texture_allocation_info,
    ).unwrap();
    // - Here is the creation of the ImageView, it is just a wrapper around the image for the GPU to interpret the image correctly.
    let texture_imageview = ImageView::new_default(texture_image.clone()).unwrap();

    // The third step is to define a sampler.
    // A sampler is needed by the shader because they use uv coordinates to diplay the image to the triangle.
    // The sampler will give the shader the right pixel color for a UV value.
    let texture_sampler = Sampler::new(
        device.clone(),
        SamplerCreateInfo {
            // NEAREST means that the nearest pixel value will be selected (which give a pixaleted effect)
            // LINEAR means that the sampler does a linear interpolation between the pixel values
            // We choose NEAREST to create a minecraft game 
            mag_filter: Filter::Nearest,                    // Applied when the texture is enlarged
            min_filter: Filter::Nearest,                    // Applied when the texture is recuded
            
            // Not relevant for now as we only have 1 mimap level in the image
            // LINEAR = interpolation between mimaps, NEAREST = no interpolation
            mipmap_mode: SamplerMipmapMode::Linear,         // Behavior between the mipmap levels
            // Not relevant for now, 
            // REPEAT = repeat the pattern 
            address_mode: [SamplerAddressMode::Repeat; 3],  // How to handle UV values outside [0,1]
            // Not relevant for now
            // <0 prefer higher quality
            // >0 prefer lower quality
            mip_lod_bias: 0.0,                              // Preference of mimap levels
            ..Default::default()
        },
    )
    .unwrap();

    // The fourth step is to define a buffer to upload the texture from the CPU to GPU.
    let texture_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,               // we use this buffer as a transfer source
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: 
                // HOST_SEQUENTIAL_WRITE is not usefull here as we give directly data to the buffer, but if we upload more texture that's great.
                MemoryTypeFilter::HOST_SEQUENTIAL_WRITE |   // optimised for sequential write
                MemoryTypeFilter::PREFER_HOST,              // optimised for fast CPU acces (the GPU will never access this buffer)
            ..Default::default()
        },
        image_data,                                   // the image bytes from above
    )
    .unwrap();

    // The last step is to upload the data to the image visible by the GPU
    let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
    // - Here we create a command buffer 
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),                 // use the same queue as usual
        CommandBufferUsage::OneTimeSubmit,   // we will use this command buffer only once
    )
    .unwrap();
    // - Here we add the command to copy the buffer to the image visible by the GPU
    builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(texture_buffer, texture_image)).unwrap();
    let texture_cmd = builder.build().unwrap();

    // - Here we start the execution of the command buffer
    let future = sync::now(device.clone())
    .then_execute(queue.clone(), texture_cmd)
    .unwrap()
    .then_signal_fence_and_flush() // same as signal fence, and then flush
    .unwrap();

    // - Here we wait the GPU to finish its work to go on
    future.wait(None).unwrap();  // None is an optional timeout

    // We are not done yet, we must change our descriptors sets to assign the image to a new binding (1 here) 
    // for the shaders to be able to read the texture.

    // Texture upload end -----------------------------------------------------------

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let mut descriptor_sets = Vec::new();
    for i in 0..images.len() {
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        );
        descriptor_sets.push(
            PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, uniform_buffers[i].clone()),
                    // Adding a descriptor for the texture with the ImageView and the Sampler
                    WriteDescriptorSet::image_view_sampler(1, texture_imageview.clone(), texture_sampler.clone())
                    ],
                [],
            )
            .unwrap(),
        );
    }

    let mut command_buffers = get_command_buffers(
        &command_buffer_allocator,
        &queue,
        &pipeline,
        &framebuffers,
        &vertex_buffers,
        &descriptor_sets,
    );

    let mut inputhandle = InputHandle::default();
    let mut player_position = vec3(0.0, 0.0, -5.0);

    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    let speed = 5.; 
    let camera_sensibility = 0.002;

    let mut angle_x: f32 = 0.; 
    let mut angle_y=0.;    

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

            fps +=1 ;
            if now.elapsed().as_micros() >= 1000000{
                println!("FPS: {:?}",fps);
                fps = 0;
                now = Instant::now();
            }
            // Modify MVP 
            let result = uniform_buffers[image_i as usize].write();
            match result {
                Ok(mut value) => {
                    inputhandle.handle_update(&mut player_position,speed,camera_sensibility,&mut angle_x,&mut angle_y);
                    let looking_towards = player_position + vec3(angle_x.sin()*angle_y.cos(),angle_y.sin(),angle_x.cos()*angle_y.cos());
                    mvp.view =  look_at(
                        &player_position,
                        &looking_towards,
                        &vec3(0.0, 1.0, 0.0),
                    );

                    value.view =mvp.view.into();
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
