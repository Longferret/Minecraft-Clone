#version 460
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in uint block_type;
layout(location = 3) in ivec2 chunk_coord;

layout(location = 0) out vec2 tex_coords;
layout(location = 1) out uint block_type2;

layout(set = 0, binding = 0) uniform UNI_data {
    mat4 mvp;
    int player_chunk_x;
    int player_chunk_z;
} uniforms;

void main() {
    // move the vertex from relative position inside a chunk to relative position to the player
    int offsx = chunk_coord.x - uniforms.player_chunk_x;
    int offsz = chunk_coord.y - uniforms.player_chunk_z;
    vec3 position_from_player = vec3(position.x+float(16*offsx),position.y,position.z+float(16*offsz));

    // mvp transformation
    vec4 mvp_pos = uniforms.mvp * vec4(position_from_player, 1.0);
    gl_Position = vec4(-mvp_pos.x,-mvp_pos.y,mvp_pos.z,mvp_pos.w);
    tex_coords = uv;
    block_type2 = block_type;
}