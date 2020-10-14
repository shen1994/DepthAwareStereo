#include <TH/TH.h>

int depthconv_forward(THFloatTensor *input, THFloatTensor *offset,
                        THFloatTensor *output)
{
  //
  return 1;
}

int depthconv_backward(THFloatTensor *grad_output, THFloatTensor *grad_input,
                         THFloatTensor *grad_offset)
{
  //
  return 1;
}

//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
//     m.def("forward", &depthconv_forward, "depthconv forward");
//    m.def("backward", &depthconv_backward, "depthconv backward");
//}
