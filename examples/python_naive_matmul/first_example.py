import kp
import numpy as np


def main():
    mgr = kp.Manager()

    tensor_size = 4
    tensor_shape = [tensor_size, tensor_size]
    tensor_in_1 = mgr.tensor(np.triu(np.ones(tensor_shape)))
    tensor_in_2 = mgr.tensor(np.triu(np.ones(tensor_shape)))
    tensor_out = mgr.tensor(np.zeros(tensor_shape))

    print(f'Input tensors:\n'
          f'{tensor_in_1.data().reshape(tensor_shape)}\n'
          f'{tensor_in_2.data().reshape(tensor_shape)}\n')

    params = [tensor_in_1, tensor_in_2, tensor_out]

    with open('simple_gemm.glsl', 'r') as f:
        str_shader = f.read()
    matmul_shader = kp.Shader.compile_source(str_shader)


    algo = mgr.algorithm(
        params,  # params
        matmul_shader,  # spirv
        (*tensor_shape, 1),  # workgroup
        [float(tensor_size)],  # spec_consts
        [])  # push_consts

    (mgr.sequence()
     .record(kp.OpTensorSyncDevice(params))
     .record(kp.OpAlgoDispatch(algo))
     .record(kp.OpTensorSyncLocal(params))
     .eval())

    print(f'Output :\n{tensor_out.data().reshape(tensor_shape)}')


if __name__ == '__main__':
    main()
