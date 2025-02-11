// Module for texture definition (BlockType) and texture upload
mod texture;
pub use texture::BlockTypes;
use texture::*;

// Module for helper function and definition of MVP,Vertex and shaders
mod renderer_utils;
use renderer_utils::*;
use vertexshader::MVP_Data;
use std::collections::HashMap;

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
    vertex_buffers: Vec<Subbuffer<[MyVertex]>>,
    uniform_buffers: Vec<Subbuffer<MVP_Data>>,
    viewport: Viewport,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    // - pipeline: 
    descriptor_sets: Vec<Arc<PersistentDescriptorSet>>,
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
    free_indexes: Vec<usize>,
    square_to_vertex_index: HashMap<Square,usize>,
    action_per_image: Vec<Vec<ActionElement>>,

    // Debug variables
    pub vertex_nbr: u64,
    pub vertex_max: u64,

    // Variable for texture modification
    blocktype_to_imageindex: HashMap<u32,u32>,
    time: Instant,
    square_to_dynamic_squares: HashMap<Square,DynamicSquare> // vertex_buffer index, texture index, time
}

impl Renderer {
    /// # Arguments:
    /// * event_loop, a reference to the winit event loop
    /// * vertexbuff_capacity, the maximal capacity of the vertex buffer (exceeding capacity will drop vertices).
    /// 
    /// # Description:
    /// Component characteristics:
    /// * 1 render pass, 1 pipeline, 1 queue used for everything
    /// * X vertex buffer for position, uv and blocktype (location = 0,1,2 respectively)
    /// * X uniform buffer for MVP (set 0 binding 0) 
    /// * 1 array of images for textures (set 0 binding 1), defined in texture.rs
    /// * Asynchronous GPU execution
    /// 
    /// Features:
    /// * Anisotropy MAX (x16) , MSAA x4
    /// * Depth testing
    /// * Face Culling
    /// 
    /// # Return:
    /// * The rendering system
    pub fn new(event_loop: &EventLoop<()>,vertexbuff_capacity: usize) -> Self{

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

        let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
        window.set_title("Minecraft Clone");
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


        let queue = queues.next().unwrap();

        let mut mvp = MVP::new();
        mvp.view = look_at(
        &vec3(0.0, 0.0, 0.0),
        &vec3(0.0, 0.0, 0.0),
        &vec3(0.0, 1.0, 0.0));

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
                    image_extent: dimensions.into(),            
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST, 
                    composite_alpha, 
                    present_mode: PresentMode::Fifo,                         
                    ..Default::default()     
                },
            )
            .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let render_pass = get_render_pass(device.clone(), swapchain.clone());
        let framebuffers = get_framebuffers(device.clone(),&images, render_pass.clone());

        // Max capacity of the vertex buffer, MUST BE A MULTIPLE OF 6 (1 square = 6 vertices)
        let capacity = vertexbuff_capacity - (vertexbuff_capacity%6);
        let mut free_indexes =  Vec::new();
        let mut vertices: Vec<MyVertex> = Vec::new();
        for i in 0..capacity{
            if i % 6 == 0 {
                free_indexes.push(i);
            }
            vertices.push(MyVertex{
                position: [0.0,0.0,0.0],
                uv: [0.0,0.0],
                block_type: 0
            });
        }

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
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE 
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    vertices.clone(),
                )
                .unwrap()
            );
            let data = vertexshader::MVP_Data {
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

        let vs = vertexshader::load(device.clone()).expect("failed to create shader module");
        let fs = fragshader::load(device.clone()).expect("failed to create shader module");
    
        let viewport = Viewport {
            offset: [0.0, 0.0],                
            extent: window.inner_size().into(), 
            depth_range: 0.0..=1.0,            
        };

        let pipeline = get_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        );

        
        // Texture upload start -----------------------------------------------------------

        // Get the bytes of all textures
        let (image_data,blocktype_to_imageindex) = load_blocks_64x64("textures");
        let image_nbr = (image_data.len() as u32)/(4*64*64);

        println!("Total texture count (64x64) : {:?}",image_nbr);

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
                mag_filter: Filter::Nearest,             
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

        let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),   
            CommandBufferUsage::OneTimeSubmit, 
        )
        .unwrap();

        builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(texture_buffer, texture_image)).unwrap();
        let texture_cmd = builder.build().unwrap();

        let future = sync::now(device.clone())
        .then_execute(queue.clone(), texture_cmd)
        .unwrap()
        .then_signal_fence_and_flush() 
        .unwrap();

        future.wait(None).unwrap(); 
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
                        WriteDescriptorSet::image_view_sampler(1, texture_imageview.clone(), texture_sampler.clone())
                    ],
                    [],
                )
                .unwrap(),
            );
        }

        let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let command_buffers = get_command_buffers(
            &command_buffer_allocator,
            &queue,
            &pipeline,
            &framebuffers,
            &vertex_buffers,
            &descriptor_sets,
        );
        let window_resized = false;
        let recreate_swapchain = false;
        let frames_in_flight = images.len();
        let fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let previous_fence_i = 0;
        let image_i =  0;
        let mut action_per_image: Vec<Vec<ActionElement>> = Vec::new();
        action_per_image.push(Vec::new());
        action_per_image.push(Vec::new());


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
            vertex_buffers,
            uniform_buffers,
            vs,
            fs,
            viewport,
            // pipeline: 
            descriptor_sets,
            command_buffers,
            window_resized,
            recreate_swapchain,
            //frames_in_flight,
            fences,
            previous_fence_i,
            image_i,
            acquirefuture: None,

            free_indexes,
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
    /// This function must be should called before the other renderer functions in the main loop.
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
                let new_pipeline = get_pipeline(
                    self.device.clone(),
                    self.vs.clone(),
                    self.fs.clone(),
                    self.render_pass.clone(),
                    self.viewport.clone(),
                );
                let command_buffer_allocator =
                StandardCommandBufferAllocator::new(self.device.clone(), Default::default());            
                self.command_buffers = get_command_buffers(
                    &command_buffer_allocator,
                    &self.queue,
                    &new_pipeline,
                    &new_framebuffers,
                    &self.vertex_buffers,
                    &self.descriptor_sets
                );
            }
        }
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
    /// This function should be called after the other renderer functions  in the main loop.
    pub fn exec_gpu(&mut self){
        self.update_frame();
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
                let looking_towards = *player_position + vec3(angle_x.sin()*angle_y.cos(),angle_y.sin(),angle_x.cos()*angle_y.cos());
                value.view = look_at(
                    &player_position,
                    &looking_towards,
                    &vec3(0.0, 1.0, 0.0),
                ).into();
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
        // Get an available index to store the 6 vertices
        let result = self.free_indexes.pop();
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
        self.free_indexes.push(index);

    }

    /// Execute the buffered modifications for the current image and check time for dynamic squares
    fn update_frame(&mut self){
        // Check time for dynamic squares
        if self.time.elapsed().as_millis() > 50 {
            self.time = Instant::now();
            for (_,dynamic_square) in  self.square_to_dynamic_squares.iter_mut() {
                //println!("Dynamic{}",dynamic_square.time);
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
                let half = 0.5;
                let extent1 = square.extent1 as f32;
                let extent2 = square.extent2 as f32;
                let x = square.x as f32;
                let y = square.y as f32;
                let z = square.z as f32;
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
                eprintln!("Unexpected  Acces Refused to Vertex Buffer");
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
                value.projection = perspective(aspect_ratio, half_pi(), 0.01, 100.0).into();
            }
            Err(_e) => {
                eprintln!("Unexpected Access Refused to the MVP buffer");
            }
        }
    }   

}


/// Defines the input data data needed to draw the face of cube
#[derive(Eq, Hash, PartialEq,Clone,Debug)]
pub struct Square{
    pub x:i64,
    pub y:i64,
    pub z:i64,
    pub extent1: u32, // draw one more cube in the direction 1 (1 is the first direction along x+,y+,z+)
    pub extent2: u32, // draw one more cube in the direction 2 (2 is the second direction along x+,y+,z+)
    pub orientation: Orientation, // only implemented for top block
    pub direction: Direction,
    pub block_type: BlockTypes,
}

/// Defines a face of a cube.
#[derive(Eq, Hash, PartialEq,Clone,Debug)]
pub enum Direction{
    UP,         // Y+
    DOWN,       // Y-
    RIGHT,      // X+
    LEFT,       // X-
    FORWARD,    // Z+
    BACKWARD,   // Z-
}

/// Define the orientation of a square
#[derive(Eq, Hash, PartialEq,Clone,Copy,Debug)]
pub enum Orientation {
    NONE,
    QUARTER,
    HALF,
    THREEQUARTER,
}

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


/// Create a square with x fixed, do not support orientation
fn get_square_xfixed(x:f32,y:f32,z:f32,half:f32,is_right:bool,extent1:f32,extent2:f32,block_type:u32) -> Vec<MyVertex>{
    let valy: [f32; 2];
    let valz: [f32; 2];
    let uvy: [f32; 2];
    let uvz: [f32; 2];
    let var: f32;
    if is_right {
        uvy = [0.,1.+extent2];
        uvz = [0.,1.+extent1];
        valy = [y+half+extent1,y-half];
        valz = [z-half,z+half+extent2];
        var = half;
    }
    else{
        uvy = [1.+extent2,0.];
        uvz = [1.+extent1,0.];
        valy = [y-half,y+half+extent1];
        valz = [z-half,z+half+extent2];
        var = -half;
    }
    let mut vertices = Vec::new();
    for i in 0..2{
        for j in 0..2{
            vertices.push(MyVertex{
                position: [x+var,valy[j],valz[i]],
                uv: [uvy[i],uvz[j]],
                block_type: block_type
            });
        }
    }
    let v4 = vertices.pop().unwrap();
    vertices.push(vertices[2].clone());
    vertices.push(vertices[1].clone());
    vertices.push(v4);

    vertices
}

/// Create a square with y fixed
fn get_square_yfixed(x:f32,y:f32,z:f32,half:f32,is_top:bool,extent1:f32,extent2:f32,block_type:u32,orientation:Orientation) -> Vec<MyVertex>{
    let valx: [f32; 2];
    let valz: [f32; 2];
    let uvx: [f32; 2];
    let uvz: [f32; 2];
    let var: f32;
    let ij_inverted :bool;
    match orientation{
        Orientation::QUARTER => {
            //uvx = [1.+extent1,0.];
            uvx = [0.,1. + extent1];
            uvz = [0.,1.+extent2];
            ij_inverted = false;
        }
        Orientation::HALF => {
            uvx = [1.+extent1,0.];
            uvz = [0.,1.+extent2];
            ij_inverted = true;
        }
        Orientation::THREEQUARTER => {
            uvx = [1.+extent1,0.];
            uvz = [1.+extent2,0.];
            ij_inverted = false;
        }
        _ => {
            uvx = [0.,1.+extent1];
            uvz = [1.+extent2,0.];
            ij_inverted = true;
        }
    }

    if is_top {
        valx = [x-half,x+half+extent1];
        valz = [z-half,z+half+extent2];
        var = half;

    }
    else{
        valx = [x+half+extent1,x-half];
        valz = [z-half,z+half+extent2];
        var = -half;
    }

    let mut vertices = Vec::new();
    if ij_inverted {
        for i in 0..2{
            for j in 0..2{
                vertices.push(MyVertex{
                    position: [valx[j],y+var,valz[i]],
                    uv: [uvx[j],uvz[i]],
                    block_type: block_type
                });
            }
        }
    }
    else{
        for i in 0..2{
            for j in 0..2{
                vertices.push(MyVertex{
                    position: [valx[j],y+var,valz[i]],
                    uv: [uvx[i],uvz[j]],
                    block_type: block_type as u32
                });
            }
        }
    }
    let v4 = vertices.pop().unwrap();
    vertices.push(vertices[2].clone());
    vertices.push(vertices[1].clone());
    vertices.push(v4);

    vertices
}

/// Create a square with z fixed, do not support orientation
fn get_square_zfixed(x:f32,y:f32,z:f32,half:f32,is_forward:bool,extent1:f32,extent2:f32,block_type:u32) -> Vec<MyVertex>{

    let valx: [f32; 2];
    let valy: [f32; 2];
    let uvx: [f32; 2];
    let uvy: [f32; 2];
    let var: f32;
    if is_forward {
        uvx = [0.,1.+extent1];
        uvy = [1.+extent2,0.];
        valx = [x+half+extent1,x-half];
        valy = [y-half,y+half+extent2];
        var = half;
    }
    else{
        uvx = [0.,1.+extent1];
        uvy = [1.+extent2,0.];
        valx = [x-half,x+half+extent1];
        valy = [y-half,y+half+extent2];
        var = -half;
    }
    let mut vertices = Vec::new();
    for i in 0..2{
        for j in 0..2{
            vertices.push(MyVertex{
                position: [valx[j],valy[i],z+var],
                uv: [uvx[j],uvy[i]],
                block_type: block_type
            });
        }
    }
    let v4 = vertices.pop().unwrap();
    vertices.push(vertices[2].clone());
    vertices.push(vertices[1].clone());
    vertices.push(v4);

    vertices
}
