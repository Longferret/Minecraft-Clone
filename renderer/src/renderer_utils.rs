// All those import are very ugly
pub use image::GenericImageView;
pub use nalgebra_glm::TVec3;
pub use nalgebra_glm::{half_pi, look_at, perspective, vec3};
pub use std::collections::BTreeMap;
pub use std::sync::Arc;
use std::time::Instant;
pub use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
pub use vulkano::buffer::Subbuffer;
pub use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
pub use vulkano::command_buffer::CommandBufferExecFuture;
pub use vulkano::command_buffer::CopyBufferToImageInfo;
pub use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents,
};
pub use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
pub use vulkano::descriptor_set::layout::DescriptorSetLayoutBinding;
pub use vulkano::descriptor_set::layout::DescriptorSetLayoutCreateInfo;
pub use vulkano::descriptor_set::layout::DescriptorType;
pub use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
pub use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
pub use vulkano::device::Features;
pub use vulkano::device::*;
pub use vulkano::format::Format;
pub use vulkano::image::sampler::{
    Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
};
pub use vulkano::image::sys::ImageCreateInfo;
pub use vulkano::image::view::ImageView;
pub use vulkano::image::Image;
pub use vulkano::image::ImageUsage;
pub use vulkano::image::SampleCount;
pub use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
pub use vulkano::memory::allocator::suballocator::FreeListAllocator;
pub use vulkano::memory::allocator::GenericMemoryAllocator;
pub use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
};
pub use vulkano::pipeline::graphics::color_blend::AttachmentBlend;
pub use vulkano::pipeline::graphics::color_blend::ColorComponents;
pub use vulkano::pipeline::graphics::color_blend::{BlendFactor, BlendOp};
pub use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
pub use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
pub use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
pub use vulkano::pipeline::graphics::multisample::MultisampleState;
pub use vulkano::pipeline::graphics::rasterization::CullMode;
pub use vulkano::pipeline::graphics::rasterization::RasterizationState;
pub use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
pub use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
pub use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
pub use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
pub use vulkano::pipeline::PipelineBindPoint;
pub use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
pub use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
pub use vulkano::shader::ShaderModule;
pub use vulkano::shader::ShaderStages;
pub use vulkano::swapchain::PresentFuture;
pub use vulkano::swapchain::PresentMode;
pub use vulkano::swapchain::SwapchainAcquireFuture;
pub use vulkano::swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
pub use vulkano::sync::future::FenceSignalFuture;
pub use vulkano::sync::future::JoinFuture;
pub use vulkano::sync::{self, GpuFuture};
pub use vulkano::{Validated, VulkanError};
pub use winit::event_loop::EventLoop;
pub use winit::window::Window;
pub use winit::window::WindowBuilder;
pub use vulkano::pipeline::graphics::depth_stencil::CompareOp;



use crate::renderer_structures::*;

// ------------ Shaders ------------

/// Vertex block shader
pub mod block_vshader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/block.vert"
    }
}

/// Fragment block shader
pub mod block_fshader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/block.frag"
    }
}

/// Vertex interface shader
pub mod interface_vshader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/interface.vert"
    }
}

/// Vertex interface shader
pub mod interface_fshader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/interface.frag"
    }
}

// ------------ Renderer Helper Functions  ------------
/// Get the as many command buffer than there are frambuffer, with 3 subpasses, one for each pipeline
pub fn get_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    pipeline_opaque: &Arc<GraphicsPipeline>,
    pipeline_transparent: &Arc<GraphicsPipeline>,
    pipeline_interface: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Vec<Subbuffer<[MyVertex]>>,
    interface_buffers: &Vec<Subbuffer<[MyVertex]>>,
    sets: &Vec<Arc<PersistentDescriptorSet>>,
    interface_sets: &Vec<Arc<PersistentDescriptorSet>>,
    first_transparent_index: usize,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    let mut out_buffers = Vec::new();
    let mut index = 0;
    let transparent_index = first_transparent_index as u32;
    for framebuffer in framebuffers {
        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();
        let tt = Instant::now(); 
        builder
            // First render pass for opaque objects
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        None,
                        Some([0.0, 0.68, 1.0, 1.0].into()),
                        Some(1.0.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .bind_pipeline_graphics(pipeline_opaque.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline_opaque.layout().clone(),
                0,
                sets[index].clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer[index].clone())
            .unwrap()
            .draw(transparent_index, 1, 0, 0)
            .unwrap()
            // Second render pass for transparent objects
            .next_subpass(Default::default(),Default::default())
            .unwrap()
            .bind_pipeline_graphics(pipeline_transparent.clone())
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer[index].clone())
            .unwrap()
            .draw((vertex_buffer[index].len() as u32) -transparent_index, 1, transparent_index, 0)
            .unwrap()
            // Third render pass for interface objects
            .next_subpass(Default::default(),Default::default())
            .unwrap()
            .bind_pipeline_graphics(pipeline_interface.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline_interface.layout().clone(),
                0,
                interface_sets[index].clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, interface_buffers[index].clone())
            .unwrap()
            .draw(interface_buffers[index].len() as u32, 1, 0, 0)
            .unwrap()
            .end_render_pass(Default::default())
            .unwrap();
        println!("BINDING {:?}: {:?}",index,tt.elapsed().as_micros()as f32/1000.);
        let tt = Instant::now(); 
        out_buffers.push(builder.build().unwrap());
        println!("BUILDING {:?}: {:?}",index,tt.elapsed().as_micros()as f32/1000.);
        index += 1;
        
    }
    out_buffers
}

/// Get the render pass (3 subpasses)
pub fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::ordered_passes_renderpass!(
        device,
        attachments: {
            final_color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: DontCare,
                store_op: Store,
            },
            multi_opaque : {
                format: swapchain.image_format(),
                samples: SampleCount::Sample4,
                load_op: Clear,
                store_op: Store,
            },
            depth: {
                format: Format::D32_SFLOAT,
                //D32_SFLOAT
                samples:SampleCount::Sample4,
                load_op: Clear,
                store_op: Store,
            },
        },
        passes: [ 
            // Pass 1: Opaque objects with MSAA and depth writing
            { 
                color: [multi_opaque], 
                depth_stencil: {depth},
                input : []
            },
            // Pass 2: Transparent objects and resolve MSAA into final_color
            { 
                color: [multi_opaque], 
                color_resolve: [final_color],
                depth_stencil: {depth},
                input : []
            },
            // Pass 3: Static interfaces (cursor,menus, health bar,..) 
            { 
                color: [final_color], 
                depth_stencil: {},
                input : []
            },
            ]
    )
    .unwrap()
}

/// This function create an imageView that can be used for depth testing.
pub fn get_depth_image(
    allocator: &Arc<GenericMemoryAllocator<FreeListAllocator>>,
    extentt: [u32; 3],
    ) -> Arc<ImageView> {
    // Specify Image information
    let image_info = ImageCreateInfo {
        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT, // use it for depth testing
        format: Format::D32_SFLOAT,                  // big depth format
        extent: extentt,                             // dimensions of the images (same as swapchain img)
        samples: SampleCount::Sample4,               // Number of sample 4x = MSAAx4
        ..Default::default()
    };
    // Create Image
    let image_alloc_info = AllocationCreateInfo::default();
    let depth_image = Image::new(allocator.clone(), image_info, image_alloc_info).unwrap();

    // Create ImageView
    let depth_buffer = ImageView::new_default(depth_image.clone()).unwrap();

    depth_buffer
}

/// This function returns an imageview with 4 samples that has the characteritisc of the Image. (for MSAAx4)
pub fn get_sampled_images(
    allocator: &Arc<GenericMemoryAllocator<FreeListAllocator>>,
    image: &Arc<Image>,
    ) -> Arc<ImageView> {
    // Specify Image information
    let image_info = ImageCreateInfo {
        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
        format: image.format(),
        extent: image.extent(),
        samples: SampleCount::Sample4,
        ..Default::default()
    };
    // Create Image
    let image_alloc_info = AllocationCreateInfo::default();
    let sampled_img = Image::new(allocator.clone(), image_info, image_alloc_info).unwrap();

    // Create ImageView
    let sampled_imgview = ImageView::new_default(sampled_img.clone()).unwrap();

    sampled_imgview
}


pub fn get_framebuffers(
    device: Arc<Device>,
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    ) -> Vec<Arc<Framebuffer>> {
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

            let depth_buffer = get_depth_image(&memory_allocator, image.extent());
            let multiview = get_sampled_images(&memory_allocator, image);

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view,multiview,depth_buffer],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

/// Get the pipeline (used for opaque and transparent blocks).
pub fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    subpass: Subpass,
    viewport: Viewport,
    msaa4_enabled: bool,
    blending_enabled: bool,
    depth_writes_enabled: bool,
    ) -> Arc<GraphicsPipeline> {
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

    let vertex_input_state = MyVertex::per_vertex()
        .definition(&vs.info().input_interface)
        .unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let mut descriptor_type0 =
        DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer);
    let mut descriptor_type1 =
        DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler);

    descriptor_type0.stages = ShaderStages::VERTEX; // Will be used in vertex shader
    descriptor_type1.stages = ShaderStages::FRAGMENT; // Will be used in fragment shader

    let descriptor_set_bindings = DescriptorSetLayoutCreateInfo {
        bindings: BTreeMap::from([(0, descriptor_type0), (1, descriptor_type1)]),
        ..Default::default()
    };

    // Create the descriptor of pipeline
    let descriptor_set_pipeline = PipelineDescriptorSetLayoutCreateInfo {
        flags: Default::default(),                  // No flags
        set_layouts: vec![descriptor_set_bindings], // Pass the descriptor bindings definitions
        push_constant_ranges: vec![],               // a small, fixed-size data block that can be passed to shaders
    };

    let layout = PipelineLayout::new(
        device.clone(),
        descriptor_set_pipeline
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    // Blending or none
    let blend_attachment;
    if blending_enabled {
        let attachmentblend = AttachmentBlend {
            src_color_blend_factor: BlendFactor::SrcAlpha,
            dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
            alpha_blend_op: BlendOp::Add,
        };
        blend_attachment = ColorBlendAttachmentState {
            blend: Some(attachmentblend),             // Enable blending
            color_write_mask: ColorComponents::all(), // Write all RGBA components
            color_write_enable: true,
        };
    }
    else {
        blend_attachment = ColorBlendAttachmentState::default();
    }

    // MSAA x4 or none
    let multisample_state;
    if msaa4_enabled {
        multisample_state = MultisampleState {
            rasterization_samples: SampleCount::Sample4, // Enable 4x MSAA
            ..MultisampleState::default()
        };
    }
    else{
        multisample_state = MultisampleState::default();
    }

    // Depth writes or not
    let depth_stencil_state;
    if depth_writes_enabled {
        depth_stencil_state = DepthStencilState{
            depth: Some(DepthState{
                write_enable: true,
                compare_op: CompareOp::LessOrEqual,
            }),
            ..Default::default()
        };
    }
    else {
        depth_stencil_state = DepthStencilState{
            depth: Some(DepthState{
                write_enable: false,
                compare_op: CompareOp::LessOrEqual,
            }),
            ..Default::default()
        };
    }

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            // Adding depth testing
            depth_stencil_state: Some(depth_stencil_state),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                ..Default::default()
            }),
            multisample_state: Some(multisample_state),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                blend_attachment
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

/// Get a pipeline for the interfaces (cursor)
pub fn get_interface_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    subpass: Subpass,
    viewport: Viewport,
    ) -> Arc<GraphicsPipeline> {
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

    let vertex_input_state = MyVertex::per_vertex()
        .definition(&vs.info().input_interface)
        .unwrap();

    let stages = [
    PipelineShaderStageCreateInfo::new(vs),
    PipelineShaderStageCreateInfo::new(fs),
    ];

    let mut descriptor_type0 =
        DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer);
    let mut descriptor_type1 =
        DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler);

    descriptor_type0.stages = ShaderStages::VERTEX; // Will be used in vertex shader
    descriptor_type1.stages = ShaderStages::FRAGMENT; // Will be used in fragment shader

    let descriptor_set_bindings = DescriptorSetLayoutCreateInfo {
        bindings: BTreeMap::from([(0, descriptor_type0), (1, descriptor_type1)]),
        ..Default::default()
    };


    // Create the descriptor of pipeline
    let descriptor_set_pipeline = PipelineDescriptorSetLayoutCreateInfo {
        flags: Default::default(),                  // No flags
        set_layouts: vec![descriptor_set_bindings], // Pass the descriptor bindings definitions
        push_constant_ranges: vec![], // a small, fixed-size data block that can be passed to shaders
    };

    let layout = PipelineLayout::new(
        device.clone(),
        descriptor_set_pipeline
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    // Blending or none

    let attachmentblend = AttachmentBlend {
        src_color_blend_factor: BlendFactor::SrcAlpha,
        dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
        color_blend_op: BlendOp::Add,
        src_alpha_blend_factor: BlendFactor::One,
        dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
        alpha_blend_op: BlendOp::Add,
    };
    let blend_attachment = ColorBlendAttachmentState {
        blend: Some(attachmentblend),             // Enable blending
        color_write_mask: ColorComponents::all(), // Write all RGBA components
        color_write_enable: true,
    };

    let multisample_state = MultisampleState::default();


    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                ..Default::default()
            }),
            multisample_state: Some(multisample_state),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                blend_attachment
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}


pub fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices")
        .filter(|p| p.supported_extensions().contains(device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("no device available")
}
