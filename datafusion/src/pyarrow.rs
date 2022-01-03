// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! This library demonstrates a minimal usage of Rust's C data interface to pass
//! arrays from and to Python.

use std::convert::{From, TryFrom};
use std::sync::Arc;

use pyo3::ffi::Py_uintptr_t;
use pyo3::import_exception;
use pyo3::prelude::*;

use crate::array::{make_array, Array, ArrayData, ArrayRef};
use crate::datatypes::{DataType, Field, Schema};
use crate::error::ArrowError;
use crate::ffi;
use crate::ffi::FFI_ArrowSchema;
use crate::record_batch::RecordBatch;
use pyo3::exceptions::{PyException, PyNotImplementedError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::PyNativeType;

use crate::arrow::array::ArrayData;
use crate::arrow::pyarrow::PyArrowConvert;
use crate::error::DataFusionError;
use crate::scalar::ScalarValue;

impl From<DataFusionError> for PyErr {
    fn from(err: DataFusionError) -> PyErr {
        PyException::new_err(err.to_string())
    }
}

impl PyArrowConvert for ScalarValue {
    fn from_pyarrow(value: &PyAny) -> PyResult<Self> {
        let py = value.py();
        let typ = value.getattr("type")?;
        let val = value.call_method0("as_py")?;

        // construct pyarrow array from the python value and pyarrow type
        let factory = py.import("pyarrow")?.getattr("array")?;
        let args = PyList::new(py, &[val]);
        let array = factory.call1((args, typ))?;

        // convert the pyarrow array to rust array using C data interface
        let array = array.extract::<ArrayData>()?;
        let scalar = ScalarValue::try_from_array(&array.into(), 0)?;

        Ok(scalar)
    }

    fn to_pyarrow(&self, _py: Python) -> PyResult<PyObject> {
        Err(PyNotImplementedError::new_err("Not implemented"))
    }
}

impl<'source> FromPyObject<'source> for ScalarValue {
    fn extract(value: &'source PyAny) -> PyResult<Self> {
        Self::from_pyarrow(value)
    }
}

impl<'a> IntoPy<PyObject> for ScalarValue {
    fn into_py(self, py: Python) -> PyObject {
        self.to_pyarrow(py).unwrap()
    }
}
