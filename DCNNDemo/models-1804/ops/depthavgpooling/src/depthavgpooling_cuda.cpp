#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "depthavgpooling_cuda_kernel.h"

void shape_check(torch::Tensor input, torch::Tensor input_depth, torch::Tensor *depthweightcount, torch::Tensor *gradOutput,
  int kH, int kW, int dH, int dW, int padH, int padW) {

  AT_CHECK(kW > 0 && kH > 0,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  AT_CHECK(dW > 0 && dH > 0,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);

  int ndim = input.ndimension();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  AT_CHECK(ndim == 3 || ndim == 4,
             "3D or 4D input tensor expected but got: %d",
             ndim);

  long nInputPlane = input.size(dimh-1);
  long nInputRows = input.size(dimh);
  long nInputCols = input.size(dimw);
  long nOutputRows, nOutputCols;
  long nOutputPlane = nInputPlane;


/////////check depth map shape /////////

  int ndim_depth = input_depth.ndimension();
  int dimf_depth = 0;
  int dimh_depth = 1;
  int dimw_depth = 2;

  if (ndim_depth == 4) {
    dimf_depth++;
    dimh_depth++;
    dimw_depth++;
  }

  AT_CHECK(ndim_depth == 3 || ndim_depth == 4,
             "3D input depth tensor expected but got: %s", ndim);

  long inputHeight_depth = input_depth.size(dimh_depth);
  long inputWidth_depth = input_depth.size(dimw_depth);

  AT_CHECK(input_depth.size(1) == 1,
             "input depth should have only 1 channel",
             nInputPlane, input.size(1));

  AT_CHECK((nInputRows == inputHeight_depth && nInputCols == inputWidth_depth),
             "input image and input depth should be the same size, but got: weightcount(%d,%d), depth(%d,%d)",
             nInputRows, inputHeight_depth, nInputCols, inputWidth_depth);

  if (depthweightcount != NULL){
      AT_CHECK(depthweightcount->size(1) == 1,
                 "input depth should have only 1 channel",
                 nInputPlane, input.size(1));

      AT_CHECK((inputHeight_depth == depthweightcount->size(2) && inputWidth_depth == depthweightcount->size(3)),
                 "input depth and input depthweightcount should be the same size, but got: weightcount(%d,%d), depth(%d,%d)",
                 depthweightcount->size(dimh_depth), depthweightcount->size(dimw_depth), inputHeight_depth, inputWidth_depth);
  }
//////////////////////////////////////////

    nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

  if (nOutputCols < 1 || nOutputRows < 1)
    AT_CHECK("Given input size: (%dx%dx%d). "
            "Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,nInputRows,nInputCols,nInputPlane,nOutputRows,nOutputCols);

  if (gradOutput != NULL) {

    AT_CHECK(gradOutput->size(dimf) == nOutputPlane,
               "invalid number of gradOutput planes, expected: %d, but got: %d",
               nOutputPlane, gradOutput->size(dimf));

    AT_CHECK((gradOutput->size(dimh)== nOutputRows &&
                gradOutput->size(dimw) == nOutputCols), "invalid size of gradOutput, expected height: %d width: %d , but got height: %d width: %d", nOutputRows, nOutputCols,
               gradOutput->size(dimh), gradOutput->size(dimw));
  }
  }


int depthavgpooling_forward_cuda(torch::Tensor input,
           torch::Tensor input_depth,
           torch::Tensor output,
           torch::Tensor depthweightcount,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {

  shape_check(input, input_depth, NULL, NULL, kH, kW, dH, dW, padH, padW);

  input = input.contiguous();
  input_depth = input_depth.contiguous();

  int batch = 1;
  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input.ndimension() == 3) {
    nInputCols = input.size(2);
    nInputRows = input.size(1);
    nInputPlane = input.size(0);
    batchSize = 1;
    batch = 0;
    input = input.resize_({1, input.size(0), input.size(1), input.size(2)});
    input_depth = input_depth.resize_({1, input_depth.size(0), input_depth.size(1), input_depth.size(2)});
  }
  else
  {
    nInputCols = input.size(3);
    nInputRows = input.size(2);
    nInputPlane = input.size(1);
    batchSize = input.size(0);
  }

  nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
  nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;

  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

  output = output.resize_({batchSize, nInputPlane, nOutputRows, nOutputCols});
  depthweightcount = depthweightcount.resize_({batchSize, 1, nInputRows, nInputCols});

  for (int elt = 0; elt < batchSize; elt++) {

    int count = output[elt].size(0) * output[elt].size(1) * output[elt].size(2);

    AvePoolForward(at::cuda::getCurrentCUDAStream(),
        count, input[elt].data<float>(), input_depth[elt].data<float>(),
        nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW, output[elt].data<float>(), depthweightcount[elt].data<float>());

    // THCudaCheck(cudaGetLastError());
  }

  if(batch == 0){
    output = output.resize_({nInputPlane, nOutputRows, nOutputCols});
    input = input.resize_({nInputPlane, nInputRows, nInputCols});
  }
}

int depthavgpooling_backward_input_cuda(
           torch::Tensor input,
           torch::Tensor input_depth,
           torch::Tensor depthweightcount,
           torch::Tensor gradOutput,
           torch::Tensor gradInput,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {

  shape_check(input, input_depth, &depthweightcount, &gradOutput, kH, kW, dH, dW, padH, padW);

  input = input.contiguous();
  input_depth = input_depth.contiguous();
  gradOutput = gradOutput.contiguous();
  depthweightcount = depthweightcount.contiguous();

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;
  int dimCol = 2;
  int dimRow = 1;

  int batch = 1;
  if (input.ndimension() == 3) {
    nInputPlane = input.size(0);
    batchSize = 1;
    batch = 0;
    input = input.resize_({1, input.size(0), input.size(1),input.size(2)});
    gradOutput = gradOutput.resize_({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }
  else
  {
    dimCol = 3;
    dimRow = 2;
    nInputPlane = input.size(1);
    batchSize = input.size(0);
  }
  nInputCols = input.size(dimCol);
  nInputRows = input.size(dimRow);

  nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
  nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

  AT_CHECK((input_depth.size(0) == batchSize), 3, "invalid batch size of input depth");

  at::resize_as_(gradInput, input);

  for (int elt = 0; elt < batchSize; elt++) {

    int count = gradInput[elt].size(0) * gradInput[elt].size(1) * gradInput[elt].size(2);

    AvePoolBackward
      (at::cuda::getCurrentCUDAStream(), count,
        gradOutput[elt].data<float>(), input_depth[elt].data<float>(), depthweightcount[elt].data<float>(),
        nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW,
        gradInput[elt].data<float>());

    // THCudaCheck(cudaGetLastError());
  }

  if (batch == 0) {
    gradOutput = gradOutput.resize_({nInputPlane, nOutputRows, nOutputCols});
    input = input.resize_({nInputPlane, nInputRows, nInputCols});
    input_depth = input_depth.resize_({1, nInputRows, nInputCols});
    gradInput = gradInput.resize_({nInputPlane, nInputRows,nInputCols});
  }
}

// bind to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depthavgpooling_forward_cuda", &depthavgpooling_forward_cuda, "depthavgpooling forward cuda");
  m.def("depthavgpooling_backward_input_cuda", &depthavgpooling_backward_input_cuda, "depthavgpooling backward input cuda");
}
