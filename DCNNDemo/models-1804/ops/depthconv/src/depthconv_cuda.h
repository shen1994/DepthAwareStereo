
int depthconv_forward_cuda(torch::Tensor input,
                             torch::Tensor input_depth,
                             torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                             torch::Tensor columns, torch::Tensor ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationH, int dilationW);

int depthconv_backward_input_cuda(
    torch::Tensor input, torch::Tensor input_depth, torch::Tensor gradOutput,
    torch::Tensor gradInput, torch::Tensor weight,
    torch::Tensor columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW);

int depthconv_backward_parameters_cuda(
    torch::Tensor input, torch::Tensor input_depth, torch::Tensor gradOutput,
    torch::Tensor gradWeight, torch::Tensor gradBias,
    torch::Tensor columns, torch::Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW,
    float scale);
