#version 450

layout (local_size_x = 1, local_size_y = 1) in;

layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 { float in_tensor_1[]; };
layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 { float in_tensor_2[]; };
layout (set = 0, binding = 2) writeonly buffer buf_out_tensor { float out_tensor[]; };

layout (constant_id = 0) const float tensor_size_f = 0;


void main()
{
    uint globalRow = gl_GlobalInvocationID.x;
    uint globalCol = gl_GlobalInvocationID.y;
    uint tensor_size = uint(tensor_size_f);
    float acc = 0.0;
    for(uint k = 0u; k < tensor_size; k++)
        acc += in_tensor_1[(k * tensor_size) + globalRow] * in_tensor_2[(globalCol * tensor_size) + k];
    out_tensor[(globalCol * tensor_size) + globalRow] = acc;
}
