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

// MVP structure
// It is used to transform 3D game coordinates into clip space
#[derive(Debug, Clone)]
pub struct MVP {
    pub model: TMat4<f32>, // If the models moves, it is not used here as we share the same MVP tp all shaders.
    pub view: TMat4<f32>,  // Where is the player and where does it looks ?
    pub projection: TMat4<f32>, // What is the angle of vision ?
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
#[derive(BufferContents, Vertex,Debug,Clone,Copy,Default)]
#[repr(C)]
pub struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color:[f32;3],
}

// vertex shader code
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        // We now extract the MVP_data from the set we created and use it for perspective.
        src: r"
            #version 460
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 color;

            layout(location = 0) out vec3 out_color;

            layout(set = 0, binding = 0) uniform MVP_Data {
                mat4 model;
                mat4 view;
                mat4 projection;
            } uniforms;

            void main() {
                mat4 worldview = uniforms.view * uniforms.model;
                vec4 pos = uniforms.projection * worldview * vec4(position, 1.0);
                gl_Position = vec4(-pos.x,-pos.y,pos.z,pos.w);
                out_color = color;
            }
        ",
    }
}
// fragment shader code
pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460
            layout(location = 0) in vec3 in_color;

            layout(location = 0) out vec4 f_color;


            void main() {
                f_color = vec4(in_color, 1);
            }
        ",
    }
}


// Create a square from a postion and its edge length
// It returns a vector of vertices.
pub fn create_square(x:f32,y:f32,z:f32,edge:f32)->Vec<MyVertex>{
    let red = [1.,0.,0.];
    let green = [0.,1.,0.];
    let blue = [0.,0.,1.];
    let half = edge/2.;
    vec![
        // back face
        MyVertex {
            position: [x-half, y-half, z+half],
            color: red
        },
        MyVertex {
            position: [x-half, y+half, z+half],
            color: red
        },
        MyVertex {
            position: [x+half, y+half, z+half],
            color: red
        },
        MyVertex {
            position: [x-half, y-half, z+half],
            color: red
        },
        MyVertex {
            position: [x+half, y+half, z+half],
            color: red
        },
        MyVertex {
            position: [x+half, y-half, z+half],
            color: red
        },
        // front face
        MyVertex {
            position: [x+half, y+half, z-half],
            color: red
        },
        MyVertex {
            position: [x+half, y-half, z-half],
            color: blue
        },
        MyVertex {
            position: [x-half, y+half, z-half],
            color: green
        },
        MyVertex {
            position: [x+half, y-half, z-half],
            color: blue
        },
        MyVertex {
            position: [x-half, y+half, z-half],
            color: green
        },
        MyVertex {
            position: [x-half, y-half, z-half],
            color: red
        },
        // bottom face
        MyVertex {
            position: [x-half, y-half, z+half],
            color: blue
        },
        MyVertex {
            position: [x+half, y-half, z+half],
            color: blue
        },
        MyVertex {
            position: [x+half, y-half, z-half],
            color: blue
        },
        MyVertex {
            position: [x-half, y-half, z+half],
            color: blue
        },
        MyVertex {
            position: [x+half, y-half, z-half],
            color: blue
        },
        MyVertex {
            position: [x-half, y-half, z-half],
            color: blue
        },
        // top face
        MyVertex {
            position: [x+half, y+half, z+half],
            color: blue
        },
        MyVertex {
            position: [x-half, y+half, z+half],
            color: blue
        },
        MyVertex {
            position: [x-half, y+half, z-half],
            color: blue
        },
        MyVertex {
            position: [x+half, y+half, z+half],
            color: blue
        },
        MyVertex {
            position: [x-half, y+half, z-half],
            color: blue
        },
        MyVertex {
            position: [x+half, y+half, z-half],
            color: blue
        },
        // left face
        MyVertex {
            position: [x-half, y-half, z-half],
            color: green
        },
        MyVertex {
            position: [x-half, y+half, z-half],
            color: green
        },
        MyVertex {
            position: [x-half, y+half, z+half],
            color: green
        },
        MyVertex {
            position: [x-half, y-half, z-half],
            color: green
        },
        MyVertex {
            position: [x-half, y+half, z+half],
            color: green
        },
        MyVertex {
            position: [x-half, y-half, z+half],
            color: green
        },
        // right face
        MyVertex {
            position: [x+half, y-half, z+half],
            color: green
        },
        MyVertex {
            position: [x+half, y+half, z+half],
            color: green
        },
        MyVertex {
            position: [x+half, y+half, z-half],
            color: green
        },
        MyVertex {
            position: [x+half, y-half, z+half],
            color: green
        },
        MyVertex {
            position: [x+half, y+half, z-half],
            color: green
        },
        MyVertex {
            position: [x+half, y-half, z-half],
            color: green
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
                    // We now give a value for depth testing 
                    // Triangle further than this value will be discraded (in vulkan coordinates).
                    // No triangle will be discarded for us as the MVP takes care to put z values in [-1,1]
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
            // We now binds the descriptor set for the uniform MVP data
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
            // We add an attachment to the render pass for depth testing.
            depth: {
                format: Format::D16_UNORM, // small efficient format for depth value
                samples: 1,
                load_op: Clear,     // we set the depth to a value (max depth)
                store_op: DontCare, // no need to store the depth after the opration
            },
        },
        pass: {
            color: [color],
            depth_stencil: {depth}, // pass the depth attachment to depth stencil (for the depth test to have the buffer)
        },
    )
    .unwrap()
}

// This function create an imageView that can be used for depth testing.
pub fn get_depth_image(allocator: &Arc<GenericMemoryAllocator<FreeListAllocator>>,extentt:  [u32; 3]) -> Arc<ImageView>{ 
    // Specify Image information
    let image_info = ImageCreateInfo {
        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,    // use it for depth testing
        format: Format::D16_UNORM,                      // small depth format
        extent: extentt,                                // dimensions of the images (same as swapchain img)
        ..Default::default() 
    };
    // Create Image
    let image_alloc_info = AllocationCreateInfo::default();
    let depth_image = Image::new(allocator.clone(),image_info,image_alloc_info).unwrap();

    // Create ImageView
    let depth_buffer= ImageView::new_default(depth_image.clone()).unwrap();

    depth_buffer
}

pub fn get_framebuffers(allocator: &Arc<GenericMemoryAllocator<FreeListAllocator>>,images: &[Arc<Image>], render_pass: Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            // Define an imageView for the depth image.
            // I call it a buffer, it is one for an image.
            // See the get_depth_image function (utils.rs) for more information.
            let depth_buffer = get_depth_image(allocator, image.extent());
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view,depth_buffer], // provide the depth image attachment
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

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

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
            // Adding depth testing
            depth_stencil_state: Some({
                let mut a = DepthStencilState::default();
                a.depth = Some(DepthState::simple());
                a
            }),
            rasterization_state: Some(RasterizationState::default()),
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











