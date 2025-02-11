pub use std::sync::Arc;
pub use vulkano::buffer::{BufferContents, Subbuffer};
pub use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
pub use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents,
};
pub use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
pub use vulkano::image::view::ImageView;
pub use vulkano::image::Image;
pub use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
pub use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
pub use vulkano::pipeline::graphics::multisample::MultisampleState;
pub use vulkano::pipeline::graphics::rasterization::RasterizationState;
pub use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
pub use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
pub use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
pub use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
pub use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo};
pub use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
pub use vulkano::shader::ShaderModule;
pub use vulkano::pipeline::PipelineBindPoint;
pub use vulkano::format::Format;
pub use vulkano::pipeline::graphics::depth_stencil::{DepthStencilState,DepthState};
pub use vulkano::image::ImageUsage;
pub use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
pub use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
pub use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
pub use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
pub use vulkano::device::*;
pub use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
pub use vulkano::swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
pub use vulkano::sync::future::FenceSignalFuture;
pub use vulkano::sync::{self, GpuFuture};
pub use vulkano::{Validated, VulkanError};
pub use winit::event::{Event, WindowEvent};
pub use winit::event_loop::{ControlFlow, EventLoop};
pub use winit::window::WindowBuilder;
pub use vulkano::memory::allocator::GenericMemoryAllocator;
pub use vulkano::memory::allocator::suballocator::FreeListAllocator;
pub use vulkano::image::sys::ImageCreateInfo;
pub use winit::event::DeviceEvent;
pub use nalgebra_glm::{
    half_pi, identity, look_at,perspective, vec3, TMat4
};
pub use vulkano::swapchain::PresentMode;
pub use vulkano::pipeline::graphics::rasterization::CullMode;
pub use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};
pub use vulkano::command_buffer::CopyBufferToImageInfo;
pub use vulkano::descriptor_set::layout::DescriptorSetLayoutCreateInfo;
pub use vulkano::descriptor_set::layout::DescriptorSetLayoutBinding;
pub use std::collections::BTreeMap;
pub use vulkano::descriptor_set::layout::DescriptorType;
pub use vulkano::shader::ShaderStages;
// MVP structure
#[derive(Debug, Clone)]
pub struct MVP {
    pub model: TMat4<f32>,
    pub view: TMat4<f32>,
    pub projection: TMat4<f32>,
}

impl MVP {
    pub fn new() -> MVP {
        MVP {
            model: identity(),
            view: identity(),
            projection: identity(),
        }
    }
}
// Vertex structure
// We don't have colors anymore but uv coordinates.
#[derive(BufferContents, Vertex,Debug,Clone,Copy,Default)]
#[repr(C)]
pub struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv:[f32;2],
}

// vertex shader code
// we now pass the coordinates of vertex in the fragment shader
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec2 uv;

            layout(location = 0) out vec2 tex_coords;

            layout(set = 0, binding = 0) uniform MVP_Data {
                mat4 model;
                mat4 view;
                mat4 projection;
            } uniforms;

            void main() {
                mat4 worldview = uniforms.view * uniforms.model;
                vec4 pos = uniforms.projection * worldview * vec4(position, 1.0);
                gl_Position = vec4(-pos.x,-pos.y,pos.z,pos.w);
                tex_coords = uv;
            }
        ",
    }
}
// fragment shader code
// we now use the uv coordinates we got from the vertex shader 
// and the sampler data in the binding 1
pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460
            layout(location = 0) in vec2 tex_coords;
            layout(location = 0) out vec4 f_color;

            layout(set = 0, binding = 1) uniform sampler2D tex;


            void main() {
                f_color = texture(tex, tex_coords);
            }
        ",
    }
}
// The UV value are hardcoded, from testing 
//U = 0.0, V = 0.0: Bottom-left corner of the texture.
//U = 1.0, V = 0.0: Bottom-right corner of the texture.
//U = 1.0, V = 1.0: Top-right corner of the texture.
//U = 0.0, V = 1.0: Top-left corner of the texture
pub fn create_square(x:f32,y:f32,z:f32,edge:f32)->Vec<MyVertex>{
    let half = edge/2.;
    vec![
        // back face
        MyVertex {
            position: [x-half, y-half, z+half],
            uv: [1.,1.]
        },
        MyVertex {
            position: [x-half, y+half, z+half],
            uv: [1.,0.]
        },
        MyVertex {
            position: [x+half, y+half, z+half],
            uv: [0.,0.]
        },
        MyVertex {
            position: [x-half, y-half, z+half],
            uv: [1.,1.]
        },
        MyVertex {
            position: [x+half, y+half, z+half],
            uv: [0.,0.]
        },
        MyVertex {
            position: [x+half, y-half, z+half],
            uv: [0.,1.]
        },
        // front face
        MyVertex {
            position: [x+half, y+half, z-half],
            uv: [1.,0.]
        },
        MyVertex {
            position: [x-half, y+half, z-half],
            uv: [0.,0.]
        },        
        MyVertex {
            position: [x+half, y-half, z-half],
            uv: [1.,1.]
        },
        MyVertex {
            position: [x+half, y-half, z-half],
            uv: [1.,1.]
        },
        MyVertex {
            position: [x-half, y+half, z-half],
            uv: [0.,0.]
        },
        MyVertex {
            position: [x-half, y-half, z-half],
            uv: [0.,1.]
        },
        // top face
        MyVertex {
            position: [x+half, y+half, z+half],
            uv: [1.,0.]
        },
        MyVertex {
            position: [x-half, y+half, z+half],
            uv: [0.,0.]
        },
        MyVertex {
            position: [x-half, y+half, z-half],
            uv: [0.,1.]
        },

        MyVertex {
            position: [x+half, y+half, z+half],
            uv: [1.,0.]
        },
        MyVertex {
            position: [x-half, y+half, z-half],
            uv: [0.,1.]
        },
        MyVertex {
            position: [x+half, y+half, z-half],
            uv: [1.,1.]
        },
        // bottom face
        MyVertex {
            position: [x-half, y-half, z+half],
            uv: [0.,0.]
        },
        MyVertex {
            position: [x+half, y-half, z+half],
            uv: [1.,0.]
        },
        MyVertex {
            position: [x+half, y-half, z-half],
            uv: [1.,1.]
        },
        MyVertex {
            position: [x-half, y-half, z+half],
            uv: [0.,0.]
        },
        MyVertex {
            position: [x+half, y-half, z-half],
            uv: [1.,1.]
        },
        MyVertex {
            position: [x-half, y-half, z-half],
            uv: [0.,1.]
        },
        // left face
        MyVertex {
            position: [x-half, y-half, z-half],
            uv: [1.,1.]
        },
        MyVertex {
            position: [x-half, y+half, z-half],  
            uv: [1.,0.]
        },
        MyVertex {
            position: [x-half, y+half, z+half],
            uv: [0.,0.]
        },
        MyVertex {
            position: [x-half, y-half, z-half],
            uv: [1.,1.]
        },
        MyVertex {
            position: [x-half, y+half, z+half],
            uv: [0.,0.]
        },
        MyVertex {
            position: [x-half, y-half, z+half],
            uv: [0.,1.]
        },
        // right face
        MyVertex {
            position: [x+half, y-half, z+half],  
            uv: [1.,1.]
        },
        MyVertex {
            position: [x+half, y+half, z+half],
            uv: [1.,0.]
        },
        MyVertex {
            position: [x+half, y+half, z-half], 
            uv: [0.,0.]
        },
        MyVertex {
            position: [x+half, y-half, z+half], 
            uv: [1.,1.]
        },
        MyVertex {
            position: [x+half, y+half, z-half], 
            uv: [0.,0.]
        },
        MyVertex {
            position: [x+half, y-half, z-half],
            uv: [0.,1.]
        },
    ]
}

pub fn get_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Vec<Subbuffer<[MyVertex]>>,
    sets: &Vec<Arc<PersistentDescriptorSet>>) -> Vec<Arc<PrimaryAutoCommandBuffer>> {

    let mut out_buffers = Vec::new();
    let mut index = 0;
    for framebuffer in framebuffers{
        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.68, 1.0, 1.0].into()), Some(1.0.into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .bind_pipeline_graphics(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics, 
                pipeline.layout().clone(), 
                0, 
                sets[index].clone())
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer[index].clone())
            .unwrap()
            .draw(vertex_buffer[index].len() as u32, 1, 0, 0)
            .unwrap()
            .end_render_pass(Default::default())
            .unwrap();

        out_buffers.push(builder.build().unwrap());
        index += 1;
    }
    out_buffers
}

pub fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,) -> (Arc<PhysicalDevice>, u32) {
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

pub fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                format: swapchain.image_format(), 
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            depth: {
                format: Format::D16_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {depth},
        },
    )
    .unwrap()
}

pub fn get_depth_image(allocator: &Arc<GenericMemoryAllocator<FreeListAllocator>>,extentt:  [u32; 3]) -> Arc<ImageView>{ 
    let image_info = ImageCreateInfo {
        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
        format: Format::D16_UNORM,
        extent: extentt,
        ..Default::default()
    };

    let image_alloc_info = AllocationCreateInfo::default();
    let depth_image = Image::new(allocator.clone(),image_info,image_alloc_info).unwrap();


    let depth_buffer= ImageView::new_default(depth_image.clone()).unwrap();

    depth_buffer
}

pub fn get_framebuffers(allocator: &Arc<GenericMemoryAllocator<FreeListAllocator>>,images: &[Arc<Image>], render_pass: Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            let depth_buffer = get_depth_image(allocator, image.extent());
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view,depth_buffer],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

pub fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,) -> Arc<GraphicsPipeline> {
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

    let vertex_input_state = MyVertex::per_vertex()
        .definition(&vs.info().input_interface)
        .unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];


    // ----- ADDED FOR TEXTURES  -----
    // Before we used a default piepeline layout for vertices with 1 binding.
    // Now we need to define that layout with all the bindings.
    // We have 2 bindings, the first for the vertex positions and the second for the texture.

    // Descriptor for vertex
    let mut descriptor_type0 = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer);
    // Descriptor for texture
    let mut descriptor_type1 = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler);

    descriptor_type0.stages = ShaderStages::VERTEX;     // Will be used in vertex shader
    descriptor_type1.stages = ShaderStages::FRAGMENT;   // Will be used in fragment shader

    // Define the bindings of each descriptors
    let descriptor_set_bindings = DescriptorSetLayoutCreateInfo {
        bindings: BTreeMap::from([(0,descriptor_type0),(1,descriptor_type1)]),
        ..Default::default()
    };

    // Create the descriptor of pipeline 
    let descriptor_set_pipeline = PipelineDescriptorSetLayoutCreateInfo {
        flags: Default::default(),                  // No flags
        set_layouts: vec![descriptor_set_bindings], // Pass the descriptor bindings definitions
        push_constant_ranges: vec![],               // a small, fixed-size data block that can be passed to shaders
    };
    // And finally Define the pipeline layout
    let layout = PipelineLayout::new(
        device.clone(),
        descriptor_set_pipeline
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    ).unwrap();

    // ----- END ADDED FOR TEXTURES  -----

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

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
            depth_stencil_state: Some({
                let mut a = DepthStencilState::default();
                a.depth = Some(DepthState::simple());
                a
            }),
            rasterization_state: Some(RasterizationState{
                cull_mode: CullMode::Back,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}











