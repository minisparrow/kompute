#version 450

layout (local_size_x = {tile_size}, local_size_y = {tile_size}) in;

layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
layout (set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};

layout (constant_id = 0) const float tensor_size_f = 0;

shared float sub_tensor_1[{tile_size}][{tile_size}];
shared float sub_tensor_2[{tile_size}][{tile_size}];

void main()
{{
    uint row = gl_LocalInvocationID.x; // 0 .. tile_size
    uint col = gl_LocalInvocationID.y; // 0 .. tile_size
    // gl_WorkGroupID : 0 .. tensor_size / tile_size
    uint globalRow = {tile_size} * gl_WorkGroupID.x + row;
    uint globalCol = {tile_size} * gl_WorkGroupID.y + col;

    uint tensor_size = uint(tensor_size_f);
    float acc = 0.0;
    uint numTiles = tensor_size / {tile_size};
    for(uint t = 0u; t < numTiles; t++)
    {{
        uint tiledRow = ({tile_size} * t) + row;
        uint tiledCol = ({tile_size} * t) + col;
        sub_tensor_1[col][row] = in_tensor_1[(tiledCol * tensor_size) + globalRow];
        sub_tensor_2[col][row] = in_tensor_2[(globalCol * tensor_size) + tiledRow];

        memoryBarrierShared();
        barrier();

        for(uint k = 0u; k < {tile_size}; k++)
            acc += sub_tensor_1[k][row] * sub_tensor_2[col][k];

        barrier();
    }}
    out_tensor[tensor_size * globalCol + globalRow] = acc;
 }}
