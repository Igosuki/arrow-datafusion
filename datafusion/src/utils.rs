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

use arrow::datatypes::{DataType, Field};

use crate::error::{DataFusionError, Result};
use crate::scalar::ScalarValue;

/// Returns the first field named `name` from the fields of a [`DataType::Struct`] or [`DataType::Union`].
/// # Error
/// Errors if
/// * the `data_type` is not a Struct or Union,
/// * there is no field named `name`
pub fn get_field<'a>(data_type: &'a DataType, name: &str) -> Result<&'a Field> {
    match data_type {
        DataType::Struct(fields) | DataType::Union(fields) => {
            let maybe_field = fields.iter().find(|x| x.name() == name);
            if let Some(field) = maybe_field {
                Ok(field)
            } else {
                Err(DataFusionError::Plan(format!(
                    "The `Struct` has no field named \"{}\"",
                    name
                )))
            }
        }
        _ => Err(DataFusionError::Plan(
            "The expression to get a field is only valid for `Struct` or 'Union'"
                .to_string(),
        )),
    }
}

/// Returns the a field access indexed by `name` from a [`DataType::List`] or [`DataType::Dictionnary`].
/// # Error
/// Errors if
/// * the `data_type` is not a Struct or,
/// * there is no field key is not of the required index type
pub fn get_indexed_field<'a>(
    data_type: &'a DataType,
    key: &ScalarValue,
) -> Result<Field> {
    match (data_type, key) {
        (DataType::Dictionary(ref kt, ref vt), ScalarValue::Utf8(Some(k))) => {
            match kt.as_ref() {
                DataType::Utf8 => Ok(Field::new(&k, *vt.clone(), true)),
                _ => Err(DataFusionError::Plan(format!("The key for a dictionary has to be an utf8 string, was : \"{}\"", key))),
            }
        },
        (DataType::Dictionary(_, _), _) => {
            Err(DataFusionError::Plan(
                "Only uf8 is valid as an indexed field in a dictionary"
                    .to_string(),
            ))
        }
        (DataType::List(lt), ScalarValue::Int64(Some(i))) => {
            Ok(Field::new(&i.to_string(), lt.data_type().clone(), false))
        }
        (DataType::List(_), _) => {
            Err(DataFusionError::Plan(
                "Only ints are valid as an indexed field in a list"
                    .to_string(),
            ))
        }
        _ => Err(DataFusionError::Plan(
            "The expression to get an indexed field is only valid for `List` or 'Dictionary'"
                .to_string(),
        )),
    }
}
