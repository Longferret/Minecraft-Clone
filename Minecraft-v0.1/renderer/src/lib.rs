
// Self made modules
mod block;                // Module for block and block charac definition
use block::*;
mod renderer_structures;  // Module for structures definition (vertex,MVP)
use renderer_structures::*;
mod renderer_utils;       // Module for helper functions definition
use renderer_utils::*;
mod texture;              // Module for texture loading
use texture::*;
mod quads;                // Module for square definition (draw square API)
use quads::*;

// Public structure
pub use block::BlockTypes;
pub use quads::{Square,Direction,Orientation};   

// Other modules
use block_vshader::UNI_data;
use std::collections::HashMap;
use winit::window::Fullscreen;
use nalgebra_glm::{translate, identity, TMat4};


/// Defines a square that should be updated by the rendering system
#[derive(Clone,Copy,Debug)]
struct DynamicSquare {
    time: u32,
    speed: u32,
    vertex_index: usize,
    texture_index: u32,
    texture_count: u32,
}

/// Defines an action the rendering system will have to take for each image in the swapchain.
enum ActionElement{
    ADD(Square,usize),
    REMOVE(usize),
    RESIZE,
    UPDATE(DynamicSquare)
}

/// The rendering system in one structure.
pub struct Renderer {
    // Components of the rendering system
    // - instance
    window: Arc<Window>,
    // - surface
    // - physical device
    device: Arc<Device>,
    // - queues
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    // - images: Vec<Arc<Image>>,
    render_pass: Arc<RenderPass>,
    // - frame buffers
    mvp_model: TMat4<f32>,
    vertex_buffers: Vec<Subbuffer<[MyVertex]>>,
    interface_buffers: Vec<Subbuffer<[MyVertex]>>,
    uniform_buffers: Vec<Subbuffer<UNI_data>>,
    viewport: Viewport,
    block_vshader: Arc<ShaderModule>,
    block_fshader: Arc<ShaderModule>,
    interface_vshader: Arc<ShaderModule>,
    interface_fshader: Arc<ShaderModule>,
    // - pipeline: 
    descriptor_sets: Vec<Arc<PersistentDescriptorSet>>,
    descriptor_sets_interface: Vec<Arc<PersistentDescriptorSet>>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer<StandardCommandBufferAllocator>>>,

    // Variable to draw at each frames
    window_resized: bool,
    recreate_swapchain: bool,
    // - frames_in_flight:
    fences: Vec<Option<Arc<FenceSignalFuture<PresentFuture<CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>>>>>>,
    previous_fence_i: u32,
    image_i: u32,
    acquirefuture: Option<SwapchainAcquireFuture>,

    // Variables to handle vertex buffer
    free_indexes_opaque: Vec<usize>,
    free_indexes_transparent: Vec<usize>,
    first_transparent_index: usize,
    square_to_vertex_index: HashMap<Square,usize>,
    action_per_image: Vec<Vec<ActionElement>>,

    // Debug variables
    pub vertex_nbr: u64,
    pub vertex_max: u64,

    // Variable for texture modification
    blocktype_to_imageindex: HashMap<u32,u32>,
    time: Instant,
    square_to_dynamic_squares: HashMap<Square,DynamicSquare>
}

impl Renderer {
    /// # Arguments:
    /// * event_loop, a reference to the winit event loop
    /// * vertexbuff_capacity, the maximal capacity of the vertex buffer (exceeding capacity will drop vertices).
    /// 
    /// # Description:
    /// Build the rendering system
    /// 
    /// # Return:
    /// * The rendering system
    pub fn new(event_loop: &EventLoop<()>,vertexbuff_capacity: usize, texture_path: &str) -> Self{

        // ----------- 1. Create the basics (Instance,Window,Surface,Device,Queues) -----------
        let library = vulkano::VulkanLibrary::new().expect("No local Vulkan library/DLL");
        let required_extensions = Surface::required_extensions(&event_loop);

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        ).expect("failed to create instance");

        let window = Arc::new(WindowBuilder::new()
            .with_title("Minecraft Clone")
            //.with_fullscreen(Some(Fullscreen::Borderless(None)))
            .build(&event_loop).unwrap());

        window.set_cursor_visible(false);
        window.set_cursor_grab(winit::window::CursorGrabMode::Confined).unwrap();
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = select_physical_device(&instance, &surface, &device_extensions);

        let sample_counts = physical_device.properties().framebuffer_color_sample_counts & physical_device.properties().framebuffer_depth_sample_counts;
        println!("Supported sample counts: {:?}", sample_counts);
        let aa = physical_device.properties().max_sampler_anisotropy;
        println!("Maximal anisotropy: {:?}", aa);
        let aa = physical_device.properties().max_image_array_layers;
        println!("Maximal array layers: {:?}", aa);

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions, 
                enabled_features: Features{
                    sampler_anisotropy: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("failed to create device");

        // ----------- 2. Create the swapchain  -----------
        let (swapchain, images) = {
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
                device.clone(), 
                surface,       
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count,  
                    //min_image_count: 3,      
                    image_format,                             
                    image_extent: dimensions.into(),            
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST, 
                    composite_alpha, 
                    present_mode: PresentMode::Immediate,
                    //present_mode: PresentMode::Fifo,                          
                    ..Default::default()     
                },
            )
            .unwrap()
        };

        // ----------- 3. Define the renderpass and framebuffer for pipeline -----------
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let render_pass = get_render_pass(device.clone(), swapchain.clone());
        let framebuffers = get_framebuffers(device.clone(),&images, render_pass.clone());
        



        // ----------- 4. Create the MVP matrix for pipeline  -----------
        let mut mvp = MVP::new();
        mvp.view = look_at(
        &vec3(0.0, 0.0, 0.0),
        &vec3(0.0, 0.0, 0.0),
        &vec3(0.0, 1.0, 0.0));

        let dimensions = window.inner_size();
        let image_extent: [u32; 2] = dimensions.into();
        let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
        mvp.projection = perspective(aspect_ratio, half_pi(), 0.01, 10000.0); 


        // ----------- 5. Create the uniform, vertex and interface buffer for pipeline  -----------
        // Max capacity of the vertex buffer, MUST BE A MULTIPLE OF 6 (1 square = 6 vertices)
        let capacity = vertexbuff_capacity - (vertexbuff_capacity%6);
        let mut free_indexes_opaque =  Vec::new();
        let mut free_indexes_transparent = Vec::new();
        let mut first_transparent_index = capacity*2/3;
        first_transparent_index = first_transparent_index - (first_transparent_index%6);
        let square_transparent_capacity = (capacity-first_transparent_index)/6;
        let square_opaque_capacity = capacity/6 - square_transparent_capacity;
        println!("Max opaque squares: {:?}",square_opaque_capacity);
        println!("Max transparent squares: {:?}",square_transparent_capacity);

        let mut vertices: Vec<MyVertex> = Vec::new();
        for i in 0..capacity{
            if i % 6 == 0 {
                if i<first_transparent_index{
                    free_indexes_opaque.push(i);
                }
                else{
                    free_indexes_transparent.push(i);
                }
            }
            vertices.push(MyVertex{
                position: [0.0,0.0,0.0],
                uv: [0.0,0.0],
                block_type: 0
            });
        }

        // Only cursor
        let interface_vertices = get_square_zfixed(0., 0., -10.0, 0.3, true, 0., 0., 0);

        let mut vertex_buffers = Vec::new();
        let mut interface_buffers = Vec::new();
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
                    vertices.clone(),
                )
                .unwrap()
            );
            interface_buffers.push(
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
                    interface_vertices.clone(),
                )
                .unwrap()
            );
            let data = block_vshader::UNI_data {
                model: mvp.model.into(),
                view: mvp.projection.into(),
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
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    data,
                )
                .unwrap(),
            );
        }
        // ----------- 6. Load the shaders for pipeline -----------
        let block_vshader = block_vshader::load(device.clone()).expect("failed to create shader module");
        let block_fshader = block_fshader::load(device.clone()).expect("failed to create shader module");
        let interface_vshader = interface_vshader::load(device.clone()).expect("failed to create shader module");
        let interface_fshader = interface_fshader::load(device.clone()).expect("failed to create shader module");
        
        // ----------- 7. Create the viewport and the pipelines  -----------
        let viewport = Viewport {
            offset: [0.0, 0.0],                
            extent: window.inner_size().into(), 
            depth_range: 0.0..=1.0,            
        };

        
        let subpass_opaque = Subpass::from(render_pass.clone(), 0).unwrap();
        let pipeline_opaque = get_pipeline(
            device.clone(),
            block_vshader.clone(),
            block_fshader.clone(),
            subpass_opaque,
            viewport.clone(),
            true,
            false,
            true,
        );

        let subpass_transparent = Subpass::from(render_pass.clone(), 1).unwrap();
        let pipeline_transparent = get_pipeline(
            device.clone(),
            block_vshader.clone(),
            block_fshader.clone(),
            subpass_transparent,
            viewport.clone(),
            true,
            true,
            false,
        );

        let subpass_interface = Subpass::from(render_pass.clone(), 2).unwrap();
        let pipeline_interface = get_interface_pipeline(
            device.clone(),
            interface_vshader.clone(),
            interface_fshader.clone(),
            subpass_interface,
            viewport.clone(),
        );
        //let pipeline_interface = pipeline_opaque.clone();
        
        // ----------- 8. Import all textures  -----------
        // Textures of blocks
        let (image_data,blocktype_to_imageindex) = load_blocks_64x64(texture_path);
        let image_nbr = (image_data.len() as u32)/(4*64*64);

        println!("Block textures count (64x64) : {:?}",image_nbr);

        let image_dimensions = [64,64,1];

        let texture_create_info = ImageCreateInfo {
            format: Format::R8G8B8A8_UNORM, 
            extent: image_dimensions,  
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            mip_levels: 1, 
            array_layers: image_nbr,  // Define an array image
            ..Default::default()
        };

        let texture_allocation_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, 
            ..Default::default()
        };

        let texture_image = Image::new(
            memory_allocator.clone(),
            texture_create_info,
            texture_allocation_info,
        ).unwrap();

        let texture_imageview = ImageView::new_default(texture_image.clone()).unwrap();
        let texture_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,   // Mandatory nearest (for minecraft like )          
                min_filter: Filter::Linear,             
                mipmap_mode: SamplerMipmapMode::Linear, 
                address_mode: [SamplerAddressMode::Repeat; 3], 
                mip_lod_bias: 0.0, 
                anisotropy: Some(physical_device.properties().max_sampler_anisotropy), // Set max anisotropy
                ..Default::default()
            },
        )
        .unwrap();

        let texture_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC, 
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: 
                    MemoryTypeFilter::HOST_SEQUENTIAL_WRITE |
                    MemoryTypeFilter::PREFER_HOST, 
                ..Default::default()
            },
            image_data,
        )
        .unwrap();

        // Texture of cursor
        let cursor_data = load_cursor(texture_path);
        let image_dimensions = [60,60,1];
        
        let cursor_create_info = ImageCreateInfo {
            format: Format::R8G8B8A8_UNORM, 
            extent: image_dimensions,  
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            mip_levels: 1, 
            array_layers: 1,
            ..Default::default()
        };

        let cursor_allocation_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, 
            ..Default::default()
        };

        let cursor_image = Image::new(
            memory_allocator.clone(),
            cursor_create_info,
            cursor_allocation_info,
        ).unwrap();

        let cursor_imageview = ImageView::new_default(cursor_image.clone()).unwrap();

        let cursor_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC, 
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: 
                    MemoryTypeFilter::HOST_SEQUENTIAL_WRITE |
                    MemoryTypeFilter::PREFER_HOST, 
                ..Default::default()
            },
            cursor_data,
        )
        .unwrap();

        // Upload data
        
        let queue = queues.next().unwrap();

        let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),   
            CommandBufferUsage::OneTimeSubmit, 
        )
        .unwrap();

        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(texture_buffer, texture_image)).unwrap()
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(cursor_buffer, cursor_image)).unwrap();
        let texture_cmd = builder.build().unwrap();

        let future = sync::now(device.clone())
        .then_execute(queue.clone(), texture_cmd)
        .unwrap()
        .then_signal_fence_and_flush() 
        .unwrap();

        future.wait(None).unwrap(); 

        // ----------- 9. Create the descriptor sets for block pipelines (textures and MVP)  -----------
        // Normally opaque and transparent pipelines have the same bindings
        let layout = pipeline_opaque.layout().set_layouts().get(0).unwrap();
        let layout_interface = pipeline_interface.layout().set_layouts().get(0).unwrap();
        let mut descriptor_sets = Vec::new();
        let mut descriptor_sets_interface = Vec::new();
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
                        WriteDescriptorSet::image_view_sampler(1, texture_imageview.clone(), texture_sampler.clone())
                    ],
                    [],
                )
                .unwrap(),
            );
            descriptor_sets_interface.push(
                PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    layout_interface.clone(),
                    [
                        WriteDescriptorSet::buffer(0, uniform_buffers[i].clone()),
                        WriteDescriptorSet::image_view_sampler(1, cursor_imageview.clone(), texture_sampler.clone())
                    ],
                    [],
                )
                .unwrap(),
            );
        }
        // ----------- 10. Create the reusable command buffers  -----------
        let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let command_buffers = get_command_buffers(
            &command_buffer_allocator,
            &queue,
            &pipeline_opaque,
            &pipeline_transparent,
            &pipeline_interface,
            &framebuffers,
            &vertex_buffers,
            &interface_buffers,
            &descriptor_sets,
            &descriptor_sets_interface,
            first_transparent_index
        );
        let window_resized = false;
        let recreate_swapchain = false;
        let frames_in_flight = images.len();
        let fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let previous_fence_i = 0;
        let image_i =  0;
        let mut action_per_image: Vec<Vec<ActionElement>> = Vec::new();
        for _i in 0..fences.len(){
            action_per_image.push(Vec::new());
        }

        Renderer {
            // instance
            window,
            // surface
            // physical device
            device,
            // queues
            queue,
            swapchain,
            // images,
            render_pass,
            // frame buffers
            mvp_model: mvp.model,
            vertex_buffers,
            interface_buffers,
            uniform_buffers,
            block_vshader,
            block_fshader,
            interface_vshader,
            interface_fshader,
            viewport,
            // pipeline: 
            descriptor_sets,
            descriptor_sets_interface,
            command_buffers,
            window_resized,
            recreate_swapchain,
            //frames_in_flight,
            fences,
            previous_fence_i,
            image_i,
            acquirefuture: None,

            free_indexes_opaque,
            free_indexes_transparent,
            first_transparent_index,
            square_to_vertex_index: HashMap::new(),
            action_per_image,

            vertex_nbr: 0,
            vertex_max: capacity as u64,

            blocktype_to_imageindex,
            time: Instant::now(),
            square_to_dynamic_squares: HashMap::new(),
        }
    }

    /// # Arguments:
    /// * The rendering system
    /// 
    /// # Description:
    /// This function waits the swapchain to get and Image and waits that the GPU finished computation for that image.
    /// 
    /// It also recreate a swapchain if needed (suboptimal or window resizing)
    /// 
    /// This function must be called before the other renderer functions in the main loop.
    pub fn wait_gpu (&mut self){
        // Recreate swapchain
        if self.window_resized || self.recreate_swapchain {
            self.recreate_swapchain = false;
            let new_dimensions = self.window.inner_size();

            let (new_swapchain, new_images) = self.swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: new_dimensions.into(),
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            self.swapchain = new_swapchain;
            let new_framebuffers = get_framebuffers(self.device.clone(),&new_images, self.render_pass.clone());
            // Recreate the command buffer and the viewport
            if self.window_resized {
                self.window_resized = false;

                self.viewport.extent = new_dimensions.into();
                let subpass_opaque = Subpass::from(self.render_pass.clone(), 0).unwrap();
                let new_pipeline_opaque = get_pipeline(
                    self.device.clone(),
                    self.block_vshader.clone(),
                    self.block_fshader.clone(),
                    subpass_opaque,
                    self.viewport.clone(),
                    true,
                    false,
                    true,
                );

                let subpass_transparent = Subpass::from(self.render_pass.clone(), 1).unwrap();
                let new_pipeline_transparent = get_pipeline(
                    self.device.clone(),
                    self.block_vshader.clone(),
                    self.block_fshader.clone(),
                    subpass_transparent,
                    self.viewport.clone(),
                    true,
                    true,
                    false,
                );

                let subpass_interface = Subpass::from(self.render_pass.clone(), 2).unwrap();
                let new_pipeline_interface = get_interface_pipeline(
                    self.device.clone(),
                    self.interface_vshader.clone(),
                    self.interface_fshader.clone(),
                    subpass_interface,
                    self.viewport.clone(),
                );

                //let new_pipeline_interface = new_pipeline_opaque.clone();

                let command_buffer_allocator =
                StandardCommandBufferAllocator::new(self.device.clone(), Default::default());            
                self.command_buffers = get_command_buffers(
                    &command_buffer_allocator,
                    &self.queue,
                    &new_pipeline_opaque,
                    &new_pipeline_transparent,
                    &new_pipeline_interface,
                    &new_framebuffers,
                    &self.vertex_buffers,
                    &self.interface_buffers,
                    &self.descriptor_sets,
                    &self.descriptor_sets_interface,
                    self.first_transparent_index
                );
            }
        }
        self.update_dynamic_squares();
        let (image_i, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(self.swapchain.clone(), None)
                    .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

            if suboptimal {
                self.recreate_swapchain = true;
            }
            
            // Wait for the GPU-future (=fence),(normally this would be the oldest fence)
            if let Some(image_fence) = &self.fences[image_i as usize] {
                image_fence.wait(None).unwrap(); 
            }

            self.acquirefuture = Some(acquire_future);
            self.image_i = image_i;
            // Modify the Vertex buffer (we can acces and modify the vertex buffer since GPU has finished work of the commandbuffer)
    }
    
    /// # Arguments:
    /// * The rendering system
    /// 
    /// # Description:
    /// This function execute the command buffer to draw the next image.
    /// 
    /// This function must be called after the other renderer functions  in the main loop.
    pub fn exec_gpu(&mut self){
        
        self.update_frame_before_execution();

        let previous_future = match self.fences[self.previous_fence_i as usize].clone() {
            // Create a NowFuture
            None => {
                let mut now = sync::now(self.device.clone());
                now.cleanup_finished();

                now.boxed()
            }
            // Use the existing GPU-future
            Some(fence) => fence.boxed(),
        };
        let future = previous_future
                // Combine the GPU-future of last GPU execution with the swapchain-future image acquisition (both must finish for next operation)
                .join(self.acquirefuture.take().unwrap()) 
                // Execute the command of the command buffer for frame i on the GPU
                .then_execute(self.queue.clone(), self.command_buffers[self.image_i as usize].clone())
                .unwrap()
                // Present the result to the swapchain
                .then_swapchain_present(
                    self.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), self.image_i),
                )
                // Flush the operation and get a future (to be able to wait -> Line 386)
                .then_signal_fence_and_flush();

            

            // Verify that the gpu-future created succeded and put it in the fence array
            self.fences[self.image_i as usize] = match future.map_err(Validated::unwrap) {
                Ok(value) => Some(Arc::new(value)),
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    None
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                    None
                }
            };
            // Change index (of swapchain image)
            self.previous_fence_i = self.image_i;
    }
    
    /// # Arguments:
    /// * The rendering system
    /// * player_position, a vector indicating the eyes position of the player.
    /// * angle_x, the angle of view along the y axis (from left=negative to right=positive)
    /// * angle_y, the angle of view along the x axis (from bottom=negative to top=positive)
    /// 
    /// # Description:
    /// This function modifies the MVP matrix to change the view of the player.
    /// 
    /// This function must be called in between wait_gpu and exec_gpu
    pub fn set_view_postition(&mut self,player_position: & TVec3<f32>,angle_x:&f32, angle_y:&f32){
        let result = self.uniform_buffers[self.image_i as usize].write();
        match result {
            Ok(mut value) => {
                let looking_towards = vec3(angle_x.sin()*angle_y.cos(),angle_y.sin(),angle_x.cos()*angle_y.cos());
                value.view = look_at(
                    &vec3(0.0, 0.0, 0.0),
                    &looking_towards,
                    &vec3(0.0, 1.0, 0.0),
                ).into();
                self.mvp_model = translate(&identity(), &-*player_position);
                value.model = self.mvp_model.into();
            }
            Err(_e) => {
                eprintln!("Unexpected Access Refused to the MVP buffer");
            }
        }
    }

    /// # Arguments:
    /// * The rendering system
    /// 
    /// # Description:
    /// Notify the rendering system that the window was resized.
    /// 
    /// This function can be called at anytime
    pub fn window_resized(&mut self){
        self.window_resized = true;
        // Not writting in the vertex buffer here, just advetising to all image of the changes to do
        for i in 0..self.fences.len(){
            self.action_per_image[i].push(ActionElement::RESIZE);
        }
    }

    /// # Arguments:
    /// * The rendering system
    /// * square, a structure specifying the charateristics of the square to draw
    /// 
    /// # Description:
    /// Notify the rendering system to add a square at the specified position.
    /// 
    /// This function is better called in between wait_gpu and exec_gpu for better performance 
    /// If the square already exists or there is no place in the vertex buffer, nothing happens
    pub fn add_square(&mut self,square: &Square){
        // Verify that the block is not already in the hashmap
        match self.square_to_vertex_index.get(square){
            Some(_val) => {return;}
            None => {}
        }
        let result;
        // Get an available index to store the 6 vertices
        if BLOCK_CHARAC[square.block_type as usize].is_transparent {
            result = self.free_indexes_transparent.pop();
        }
        else{
            result = self.free_indexes_opaque.pop();
        }
        //let result = self.free_indexes_opaque.pop();
        let vertex_index;
        match result {
            Some(value) => {
                vertex_index = value;
                self.vertex_nbr += 6;
            }
            None => {
                return;
            }
        }
        // Not writting in the vertex buffer here, just advetising to all image of the changes to do
        for i in 0..self.fences.len(){
            self.action_per_image[i].push(ActionElement::ADD(square.clone(),vertex_index));
        }
        // Add element to hashmap (to be able to retreive its index in the vertex buffer)
        self.square_to_vertex_index.insert(square.clone(), vertex_index);

        // Add the square to the dynamic squares if needed
        let block_charac = &BLOCK_CHARAC[square.block_type as usize];
        if block_charac.animation_speed > 0{
            let texture_index = self.blocktype_to_imageindex.get(&(square.block_type as u32)).unwrap();
            let texture_count = block_charac.textures;
            let dynamic_square = DynamicSquare{
                time: 0,
                speed: BLOCK_CHARAC[square.block_type as usize].animation_speed,
                vertex_index: vertex_index,
                texture_index: *texture_index,
                texture_count: texture_count
            };
            self.square_to_dynamic_squares.insert(square.clone(),dynamic_square);
        }

    }

    /// # Arguments:
    /// * The rendering system
    /// * square, a structure specifying the charateristics of the square to remove
    /// 
    /// # Description:
    /// Notify the rendering system to remove a square at the specified position.
    /// 
    /// This function is better called in between wait_gpu and exec_gpu for better performance 
    /// If the square don't exist in the vertex buffer, nothing happens
    pub fn remove_square(&mut self,square: &Square){

        // Remove the square from the dynamic squares
        let _ =  &mut self.square_to_dynamic_squares.remove(square); 

        // Verify that the block is in the hashmap
        let index;
        let sq_to_ind = &mut self.square_to_vertex_index;
        match sq_to_ind.get(square){
            Some(val) => {
                index = val.clone();
                self.vertex_nbr -= 6;
            }
            None => {return;}
        }
        sq_to_ind.remove(square);

         // Not writting in the vertex buffer here, just advetising to all image of the changes to do
        for i in 0..self.fences.len(){
            self.action_per_image[i].push(ActionElement::REMOVE(index));
        }
        self.free_indexes_opaque.push(index);

    }


    /// Check if dynamic squares must be updated
    /// It is done before waiting for the GPU
    fn update_dynamic_squares(&mut self){
        // Check time for dynamic squares
        if self.time.elapsed().as_millis() > 50 {
            self.time = Instant::now();
            for (_,dynamic_square) in  self.square_to_dynamic_squares.iter_mut() {
                // No update yet
                if dynamic_square.time < dynamic_square.speed{
                    dynamic_square.time += 1;
                }
                else{
                    dynamic_square.time = 0;
                    for i in 0..self.fences.len(){
                        self.action_per_image[i].push(ActionElement::UPDATE(*dynamic_square));
                    }
                } 
            } 
        }
    }
   
    /// Execute the buffered modifications for the current image
    /// * Modify vertex buffer (ADD/REMOVE vertices, CHANGE index of texture)
    /// * Modify MVP matrix (recalculate projection component)
    /// * Update the dynmic squares texture
    fn update_frame_before_execution(&mut self){
        // Execute buffered actions
        for actionelem in &self.action_per_image[self.image_i as usize]{
            match &actionelem {
                ActionElement::ADD(square,index)=> {
                    self.add_square_to_vertex_buffer(square,*index);
                }
                ActionElement::REMOVE(index)=> {
                    self.remove_square_to_vertex_buffer(*index);
                }
                ActionElement::RESIZE => {
                    self.reset_projection_to_uniform_buffer();
                }
                ActionElement::UPDATE(dynamic_square) => {
                    self.update_dynamic_texture(*dynamic_square);
                }
            }
        }
        self.action_per_image[self.image_i as usize].clear();
    
    }

    /// Update the square texture
    fn update_dynamic_texture(&self,dynamic_square: DynamicSquare) {
        let vertex_buffer = self.vertex_buffers[self.image_i as usize].write();
        match vertex_buffer {
            Ok(mut value) => {
                let texture_index = dynamic_square.texture_index;
                let vertex_index = dynamic_square.vertex_index;
                let texture_count= dynamic_square.texture_count;
                for i in 0..6 {
                    value[vertex_index+i].block_type = texture_index + ((value[vertex_index+i].block_type - texture_index +1))%texture_count;
                    
                }
            }
            Err(_e) => {
                eprintln!("Unexpected Acces Refused to Vertex Buffer");
                return;
            }
        }
    }

    /// Add a square to the specified position for the current vertex buffer.
    fn add_square_to_vertex_buffer(&self, square: &Square,index:usize){

        let vertex_buffer = self.vertex_buffers[self.image_i as usize].write();
        match vertex_buffer {
            Ok(mut value) => {
                let half;
                if square.is_interior {half = 0.503;}
                else{half = 0.5;}
                let extent1 = square.extent1 as f32;
                let extent2 = square.extent2 as f32;
                let x = square.x as f32 +0.5;
                let y = square.y as f32 +0.5;
                let z = square.z as f32 +0.5;
                let vertices: Vec<MyVertex>;
                let  block_type = self.blocktype_to_imageindex.get(&(square.block_type as u32)).unwrap();
                match square.direction {
                    Direction::UP =>{
                       vertices = get_square_yfixed(x, y, z, half,true,extent1,extent2,*block_type,square.orientation);
                    }
                    Direction::DOWN =>{
                        vertices = get_square_yfixed(x, y, z, half,false,extent1,extent2,*block_type,square.orientation);
                    }
                    Direction::LEFT =>{
                        vertices = get_square_xfixed(x, y, z, half,false,extent1,extent2,*block_type);
                    }
                    Direction::RIGHT =>{
                        vertices = get_square_xfixed(x, y, z, half,true,extent1,extent2,*block_type);
                    }
                    Direction::FORWARD =>{
                        vertices = get_square_zfixed(x, y, z, half,true,extent1,extent2,*block_type);
                    }
                    Direction::BACKWARD =>{
                        vertices = get_square_zfixed(x, y, z, half,false,extent1,extent2,*block_type);
                    }
                }
                for i in 0..6 {
                    value[index+i] = vertices[i];
                }
            }
            Err(_e) => {
                eprintln!("Unexpected Acces Refused to Vertex Buffer");
                return;
            }
        }

    }

    /// Remove a square to the specified position for the current vertex buffer.
    fn remove_square_to_vertex_buffer(&self, index: usize){
        let vertex_buffer = self.vertex_buffers[self.image_i as usize].write();
        match vertex_buffer {
            Ok(mut value) => {
                for i in 0..6 {
                    value[index + i]= MyVertex {
                        position: [0.0,0.0,0.0],
                        uv: [0.,0.],
                        block_type: 0
                    };
                }
            }
            Err(_e) => {
                eprintln!("Unexpected Acces Refused to Vertex Buffer");
                return;
            }
        }

    }

    /// Recalculate the projection component of the MVP for the current uniform buffer (when window is resized).
    fn reset_projection_to_uniform_buffer(&self){
        let result = self.uniform_buffers[self.image_i as usize].write();
        match result {
            Ok(mut value) => {
                let dimensions = self.window.inner_size();
                let image_extent: [u32; 2] = dimensions.into();
                let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
                // Definition of the projection component (from the player to all its environment)
                value.projection = perspective(aspect_ratio, half_pi(), 0.01, 10000.0).into();
            }
            Err(_e) => {
                eprintln!("Unexpected Access Refused to the MVP buffer");
            }
        }
    }   

}