#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <torch/extension.h>
#include "depthconv_cuda_kernel.h"

void shape_check(torch::Tensor input, torch::Tensor input_depth,
                 torch::Tensor *gradOutput, torch::Tensor *weight, torch::Tensor *bias, int kH, int kW,
                 int dH, int dW, int padH, int padW, int dilationH,
                 int dilationW) {

  AT_CHECK(weight->ndimension() == 4,
             "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
             "but got: %s",
             weight->ndimension());

  AT_CHECK(weight->is_contiguous(),
             "weight tensor has to be contiguous");

  AT_CHECK(kW > 0 && kH > 0,
             "kernel size should be greater than zero, but got kH: %d kW: %d",
             kH, kW);

  AT_CHECK((weight->size(2) == kH && weight->size(3) == kW),
             "kernel size should be consistent with weight, but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,
             kW, weight->size(2), weight->size(3));

  AT_CHECK(dW > 0 && dH > 0,
             "stride should be greater than zero, but got dH: %d dW: %d", dH,
             dW);

  AT_CHECK(dilationW > 0 && dilationH > 0,
             "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
             dilationH, dilationW);

  //////////// check bias //////////////////

  if (bias != NULL) {
    AT_CHECK(bias->is_contiguous(),
             "bias tensor has to be contiguous");
    AT_CHECK(bias->ndimension()==1,
             "Need bias of dimension %d but got %d", 1, bias->ndimension());
    AT_CHECK(bias->size(0) == weight->size(0),
             "Need bias of size %d but got %d", weight->size(0), bias->size(0));
  }
//////////////////////////////////////////

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
             "3D or 4D input tensor expected but got: %s", ndim);

  long nInputPlane = weight->size(1);
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);
  long nOutputPlane = weight->size(0);
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  if (outputWidth < 1 || outputHeight < 1)
    AT_ERROR("Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
        outputWidth);

  AT_CHECK((inputHeight >= kH && inputWidth >= kW),
             "input image is smaller than kernel");

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

  AT_CHECK((inputHeight == inputHeight_depth && inputWidth == inputWidth_depth),
             "input image and input depth should be the same size");
//////////////////////////////////////////

  if (gradOutput != NULL) {
    AT_CHECK(gradOutput->size(dimf) == nOutputPlane,
               "invalid number of gradOutput planes, expected: %d, but got: %d",
               nOutputPlane, gradOutput->size(dimf));

    AT_CHECK((gradOutput->size(dimh) == outputHeight &&
                gradOutput->size(dimw) == outputWidth),
                "invalid size of gradOutput, expected height: %d width: %d , but got height: %d width: %d", outputHeight, outputWidth,
                gradOutput->size(dimh), gradOutput->size(dimw));
  }
}

/** 
 *@brief: relize sgemm
**/

static cublasOperation_t _cublasOpFromChar(char op) {
  switch (op) {
    case 'n':
    case 'N':
      return CUBLAS_OP_N;
    case 't':
    case 'T':
      return CUBLAS_OP_T;
    case 'c':
    case 'C':
      return CUBLAS_OP_C;
  }
  AT_ERROR(
      "_cublasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
}

static void _cublasAdjustLdLevel3(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc) {
  bool transa_ = ((transa == 't') || (transa == 'T'));
  bool transb_ = ((transb == 't') || (transb == 'T'));

  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).
  if (n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if (transa_) {
    if (m <= 1)
      *lda = std::max<int64_t>(k, 1);
  } else {
    if (k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if (transb_) {
    if (k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  } else {
    if (n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }
}

template <typename Dtype>
inline void gemm(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::gemm: not implemented for ", typeid(Dtype).name());
}
template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float));

template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasSetStream(handle, stream);
  cublasSgemm(handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}

/**
 *@brief: relize sgemv
**/
static void _cublasAdjustLdLevel2(int64_t m, int64_t n, int64_t* lda) {
  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).

  // Q: Why does Level3 check trans but this doesn't?
  // A: In level 2, the sizes (m, n) specify the size of A
  // (independent of trans value). In level 3. the sizes (m, n, k)
  // specify the sizes of op(A), op(B) where op depend on trans
  // values.
  if (n <= 1)
    *lda = std::max<int64_t>(m, 1);
}

template <typename Dtype>
inline void gemv(CUDABLAS_GEMV_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::gemv: not implemented for ", typeid(Dtype).name());
}

template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float));

template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = _cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  cublasSetStream(handle, stream);
  cublasSgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
}

int depthconv_forward_cuda(torch::Tensor input, torch::Tensor input_depth, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                             torch::Tensor columns, torch::Tensor ones, int kW, int kH, int dW, int dH, int padW, int padH,
                             int dilationH, int dilationW) {

  shape_check(input, input_depth, NULL, &weight, &bias, kH, kW, dH, dW, padH, padW,
              dilationH, dilationW);

  input = input.contiguous();
  input_depth = input_depth.contiguous();
  weight = weight.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.resize_({1, input.size(0), input.size(1), input.size(2)});
    input_depth = input_depth.resize_({1, input_depth.size(0), input_depth.size(1), input_depth.size(2)});
  }

  int64_t batchSize = input.size(0);
  int64_t nInputPlane = input.size(1);
  int64_t inputHeight = input.size(2);
  int64_t inputWidth = input.size(3);

  int64_t nOutputPlane = weight.size(0);

  int64_t outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  bias = bias.is_contiguous() ? bias.contiguous() : bias;
  output = output.resize_({batchSize, nOutputPlane, outputHeight, outputWidth});
  columns = columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});
  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
    ones = ones.resize_({outputHeight, outputWidth});
    ones = at::fill_(ones, 1);
  }

  for (int elt = 0; elt < batchSize; elt++) {

    // Do bias first
     int64_t m_ = nOutputPlane;
     int64_t n_ = outputHeight * outputWidth;
     int64_t k_ = 1;

     if (bias.sizes().size() != 0) {
       gemm<float>(at::cuda::getCurrentCUDAStream(), \
                        't', 'n', n_, m_, k_, 1.0f, \
                        ones.data<float>(), k_, \
                        bias.data<float>(), k_, 0.0f, \
                        output[elt].data<float>(), n_);
     } else {
       output[elt] = at::zeros_like(output[elt]);
     }

    depthconv_im2col(at::cuda::getCurrentCUDAStream(), 
        input[elt].data<float>(), input_depth[elt].data<float>(), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, columns.data<float>());

    int64_t m = nOutputPlane;
    int64_t n = columns.size(1);
    int64_t k = nInputPlane * kH * kW;

    gemm<float>(at::cuda::getCurrentCUDAStream(), 'n', 'n', n, m, k, 1.0f,
                     columns.data<float>(), n,
                     weight.data<float>(), k, 1.0f,
                     output[elt].data<float>(), n);
  }

  if (batch == 0) {
    output = output.resize_({nOutputPlane, outputHeight, outputWidth});
    input = input.resize_({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}

int depthconv_backward_input_cuda(
    torch::Tensor input, torch::Tensor input_depth, torch::Tensor gradOutput,
    torch::Tensor gradInput, torch::Tensor weight, torch::Tensor columns, 
    int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW) {

  shape_check(input, input_depth, &gradOutput, &weight, NULL, kH, kW, dH, dW, padH,
              padW, dilationH, dilationW);

  input = input.contiguous();
  input_depth = input_depth.contiguous();
  gradOutput = gradOutput.contiguous();
  weight = weight.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.resize_({1, input.size(0), input.size(1), input.size(2)});
    gradOutput = gradOutput.resize_({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  AT_CHECK((input_depth.size(0) == batchSize), "invalid batch size of input depth");

  gradInput = gradInput.resize_({batchSize, nInputPlane, inputHeight, inputWidth});
  columns = columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});

  for (int elt = 0; elt < batchSize; elt++) {

    long m = nInputPlane * kW * kH;
    long n = columns.size(1);
    long k = nOutputPlane;

    gemm<float>(at::cuda::getCurrentCUDAStream(), 'n', 't', n, m, k, 1.0f,
                     gradOutput.data<float>(), n,
                     weight.data<float>(), m, 0.0f,
                     columns.data<float>(), n);

    depthconv_col2im(at::cuda::getCurrentCUDAStream(), columns.data<float>(),
        input_depth[elt].data<float>(), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, gradInput[elt].data<float>());
  }

  if (batch == 0) {
    gradOutput = gradOutput.resize_({nOutputPlane, outputHeight, outputWidth});
    input = input.resize_({nInputPlane, inputHeight, inputWidth});
    input_depth = input_depth.resize_({1, inputHeight, inputWidth});
    gradInput = gradInput.resize_({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}

int depthconv_backward_parameters_cuda(
    torch::Tensor input, torch::Tensor input_depth, torch::Tensor gradOutput,
    torch::Tensor gradWeight, torch::Tensor gradBias,
    torch::Tensor columns, torch::Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW,
    float scale) {

  shape_check(input, input_depth, &gradOutput, &gradWeight, &gradBias, kH, kW, dH, dW,
              padH, padW, dilationH, dilationW);

  input = input.contiguous();
  input_depth = input_depth.contiguous();
  gradOutput = gradOutput.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.resize_({1, input.size(0), input.size(1), input.size(2)});
    gradOutput = gradOutput.resize_({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = gradWeight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;


  // Define a buffer of ones, for bias accumulation
  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
    ones = ones.resize_({outputHeight, outputWidth});
    at::fill_(ones, 1);
  }

  columns = columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});

  for (int elt = 0; elt < batchSize; elt++) {

    depthconv_im2col(at::cuda::getCurrentCUDAStream(), input[elt].data<float>(),
        input_depth[elt].data<float>(), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, columns.data<float>());

    long m = nOutputPlane;
    long n = nInputPlane * kW * kH;
    long k = columns.size(1);

    gemm<float>(at::cuda::getCurrentCUDAStream(), 't', 'n', n, m, k, scale,
                     columns.data<float>(), k,
                     gradOutput.data<float>(), k, 1.0f,
                     gradWeight.data<float>(), n);

    // Do Bias:
    // M,N,K are dims of matrix A and B
    long m_ = nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    if(gradBias.sizes().size() != 0){
        gemv<float>(at::cuda::getCurrentCUDAStream(), 't', k_, m_, scale,
                         gradOutput.data<float>(), k_,
                         ones.data<float>(), 1, 1.0f,
                         gradBias.data<float>(), 1);
    }
  }

  if (batch == 0) {
    gradOutput = gradOutput.resize_({nOutputPlane, outputHeight, outputWidth});
    input = input.resize_({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}

// bind to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depthconv_forward_cuda", &depthconv_forward_cuda, "depthconv forward cuda");
  m.def("depthconv_backward_input_cuda", &depthconv_backward_input_cuda, "depthconv backward input cuda");
  m.def("depthconv_backward_parameters_cuda", &depthconv_backward_parameters_cuda, "depthconv backward parameters cuda");
}
