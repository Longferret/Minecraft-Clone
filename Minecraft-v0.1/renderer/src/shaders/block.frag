#version 460
layout(location = 0) in vec2 tex_coords;
layout(location = 1) in flat uint block_type2;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2DArray tex;

void main() {
    f_color = texture(tex, vec3(tex_coords, float(block_type2)));
}