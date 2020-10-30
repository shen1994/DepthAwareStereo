
int depthavgpooling_forward_cuda(torch::Tensor input,
           torch::Tensor input_depth,
           torch::Tensor output,
           torch::Tensor depthweightcount,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) ;

int depthavgpooling_backward_input_cuda(
           torch::Tensor input,
           torch::Tensor input_depth,
           torch::Tensor depthweightcount,
           torch::Tensor gradOutput,
           torch::Tensor gradInput,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) ;
