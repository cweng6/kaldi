#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "matrix/kaldi-matrix.h"


namespace py = pybind11;
using namespace kaldi;

//template<typename Real> 
//void DeclareMatrixClass(py::module &m, const std::string &type_str) {
//  using Class = Matrix<Real>; 
//  std::string pyclass_name = std::string("Matrix") + type_str;
//  py::class_<Class>(m, pyclass_name.c_str())
//    .def("num_rows", &Class::NumRows) 
//    .def("num_cols", &Class::NumCols)
//    .def("size_in_bytes", &Class::SizeInBytes);
//}


PYBIND11_MODULE(kaldi_matrix, m) {
  m.doc() = "python wrapped kaldi matrix module";
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

  py::class_<Matrix<float>>(m, "MatrixFloat", py::buffer_protocol())
    .def(py::init<>())
    .def(py::init<const MatrixIndexT, const MatrixIndexT,
                  MatrixResizeType, MatrixStrideType>(), 
         py::arg("r"), py::arg("c"), 
         py::arg("resize_type")=MatrixResizeType::kSetZero,
         py::arg("stride_type")=MatrixStrideType::kDefaultStride)
    .def(py::init<const Matrix<float>&>())
    .def("num_rows", &Matrix<float>::NumRows) 
    .def("num_cols", &Matrix<float>::NumCols)
    .def("size_in_bytes", &Matrix<float>::SizeInBytes)
    .def("swap", (void (Matrix<float>::*)(Matrix<float>*)) &Matrix<float>::Swap)
    .def("__call__", (float& (Matrix<float>::*)(MatrixIndexT, MatrixIndexT))\
                      &Matrix<float>::operator(), py::is_operator())
    .def_buffer([](Matrix<float> &mat) -> py::buffer_info {
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
}


//DeclareMatrixClass<float>(m, "Float"); 

//     .def(py::init<>())
       
