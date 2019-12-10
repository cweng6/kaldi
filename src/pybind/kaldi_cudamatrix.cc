#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-matrix.h"


namespace py = pybind11;
using namespace kaldi;

torch::Tensor tensor_from_kaldi_cudamatrix(CuMatrix<float> &in) {
  return torch::from_blob(in.Data(),
                          {in.NumRows(), in.NumCols()},
                          {in.Stride(), 1}, at::device(at::kCUDA).dtype(at::kFloat));  
}


PYBIND11_MODULE(kaldi_cudamatrix, m) {
  m.doc() = "python wrapped kaldi cudamatrix module";

  //enumerations types
  py::enum_<MatrixResizeType>(m, "MatrixResizeType")
    .value("kSetZero", MatrixResizeType::kSetZero)
    .value("kUndefined", MatrixResizeType::kUndefined)
    .value("kCopyData", MatrixResizeType::kCopyData)
    .export_values();

  py::enum_<MatrixStrideType>(m, "MatrixStrideType")
    .value("kDefaultStride", MatrixStrideType::kDefaultStride)
    .value("kStrideEqualNumCols", MatrixStrideType::kStrideEqualNumCols)
    .export_values();

  //CuDevice
  py::class_<CuDevice>(m, "CuDevice")
    .def_static("instantiate", &CuDevice::Instantiate, 
                py::return_value_policy::reference)
    .def("select_gpu_id", &CuDevice::SelectGpuId);   

  //CuMatrix
  py::class_<CuMatrix<float>>(m, "CuMatrixFloat", py::buffer_protocol())
    .def(py::init<>())
    .def(py::init<const MatrixIndexT, const MatrixIndexT,
                  MatrixResizeType, MatrixStrideType>(), 
         py::arg("r"), py::arg("c"), 
         py::arg("resize_type")=MatrixResizeType::kSetZero,
         py::arg("stride_type")=MatrixStrideType::kDefaultStride)
    .def("num_rows", &CuMatrix<float>::NumRows) 
    .def("num_cols", &CuMatrix<float>::NumCols)
    .def("stride", &CuMatrix<float>::Stride)
    .def("__call__", (float (CuMatrix<float>::*)(MatrixIndexT, MatrixIndexT) const)\
                      &CuMatrix<float>::operator(), py::is_operator())
    .def_buffer([](CuMatrix<float> &mat) -> py::buffer_info {
      return py::buffer_info(
        mat.Data(), /*raw pointer*/
        sizeof(float),/*size of one scalar*/
        py::format_descriptor<float>::format(),
        2, /*number of dimensions*/
        {mat.NumRows(), mat.NumCols()} , /*buffer dimension*/
        {sizeof(float) * mat.Stride(), sizeof(float)} /*strides in bytes*/
      );
    }
    );
  //torch_from_kaldi_cudamatrix
  m.def("tensor_from_kaldi_cudamatrix", &tensor_from_kaldi_cudamatrix);
    
}
