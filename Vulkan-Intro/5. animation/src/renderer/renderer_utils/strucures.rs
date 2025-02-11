/// The MVP structure for first personn perspective
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
/// The vertex sturcture used by shaders
/// * position, the position of the vertex
/// * uv, the position for the texture
/// * block_type, the texture to display
#[derive(BufferContents, Vertex,Debug,Clone,Copy,Default)]
#[repr(C)]
pub struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv:[f32;2],
    #[format(R32_UINT)]
    pub block_type: u32,
}

/// Vertex shader code
/// Change here, we do not need to calculate uniforms.view * uniforms.model because the model is not moving
pub mod vertexshader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec2 uv;
            layout(location = 2) in uint block_type;

            layout(location = 0) out vec2 tex_coords;
            layout(location = 1) out uint block_type2;

            layout(set = 0, binding = 0) uniform MVP_Data {
                mat4 model;
                mat4 view;
                mat4 projection;
            } uniforms;

            void main() {
                //mat4 worldview = uniforms.view * uniforms.model;
                //vec4 pos = uniforms.projection * worldview * vec4(position, 1.0);
                vec4 pos = uniforms.projection * uniforms.view * vec4(position, 1.0);
                gl_Position = vec4(-pos.x,-pos.y,pos.z,pos.w);
                tex_coords = uv;
                block_type2 = block_type;
            }
        ",
    }
}

/// Fragment shader code
pub mod fragshader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460
            layout(location = 0) in vec2 tex_coords;
            layout(location = 1) in flat uint block_type2;
            layout(location = 0) out vec4 f_color;

            layout(set = 0, binding = 1) uniform sampler2DArray tex;

            void main() {
                f_color = texture(tex, vec3(tex_coords, float(block_type2)));
            }
        ",
    }
}