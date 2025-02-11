#version 460
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in uint block_type;

layout(location = 0) out vec2 tex_coords;
layout(location = 1) out uint block_type2;

layout(set = 0, binding = 0) uniform UNI_data {
    mat4 model;
    mat4 view;
    mat4 projection;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.model;
    vec4 pos = uniforms.projection * worldview * vec4(position, 1.0);
    gl_Position = vec4(-pos.x,-pos.y,pos.z,pos.w);
    tex_coords = uv;
    block_type2 = block_type;
}