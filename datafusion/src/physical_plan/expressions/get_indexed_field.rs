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

//! get field of a `ListArray`

use std::convert::TryInto;
use std::{any::Any, sync::Arc};

use arrow::{
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};

use crate::arrow::array::Array;
use crate::field_util::StructArrayExt;
use crate::scalar::ScalarValue;
use crate::{
    error::DataFusionError,
    error::Result,
    field_util::get_indexed_field as get_data_type_field,
    physical_plan::{ColumnarValue, PhysicalExpr},
};
use arrow::array::{ListArray, StructArray};
use arrow::compute::concatenate::concatenate;
use std::fmt::Debug;

/// expression to get a field of a struct array.
#[derive(Debug)]
pub struct GetIndexedFieldExpr {
    arg: Arc<dyn PhysicalExpr>,
    key: ScalarValue,
}

impl GetIndexedFieldExpr {
    /// Create new get field expression
    pub fn new(arg: Arc<dyn PhysicalExpr>, key: ScalarValue) -> Self {
        Self { arg, key }
    }

    /// Get the input expression
    pub fn arg(&self) -> &Arc<dyn PhysicalExpr> {
        &self.arg
    }
}

impl std::fmt::Display for GetIndexedFieldExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}).[{}]", self.arg, self.key)
    }
}

impl PhysicalExpr for GetIndexedFieldExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        let data_type = self.arg.data_type(input_schema)?;
        get_data_type_field(&data_type, &self.key).map(|f| f.data_type().clone())
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        let data_type = self.arg.data_type(input_schema)?;
        get_data_type_field(&data_type, &self.key).map(|f| f.is_nullable())
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let arg = self.arg.evaluate(batch)?;
        match arg {
            ColumnarValue::Array(array) => match (array.data_type(), &self.key) {
                (DataType::List(_) | DataType::Struct(_), _) if self.key.is_null() => {
                    let scalar_null: ScalarValue = array.data_type().try_into()?;
                    Ok(ColumnarValue::Scalar(scalar_null))
                }
                (DataType::List(_), ScalarValue::Int64(Some(i))) => {
                    let as_list_array =
                        array.as_any().downcast_ref::<ListArray<i32>>().unwrap();
                    if as_list_array.is_empty() {
                        let scalar_null: ScalarValue = array.data_type().try_into()?;
                        return Ok(ColumnarValue::Scalar(scalar_null))
                    }
                    let sliced_array: Vec<Box<dyn Array>> = as_list_array
                        .iter()
                        .filter_map(|o| o.map(|list| list.slice(*i as usize, 1)))
                        .collect();
                    let vec = sliced_array.iter().map(|a| a.as_ref()).collect::<Vec<&dyn Array>>();
                    let iter = concatenate(&vec).unwrap();
                    Ok(ColumnarValue::Array(iter.into()))
                }
                (DataType::Struct(_), ScalarValue::Utf8(Some(k))) => {
                    let as_struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
                    match as_struct_array.column_by_name(k) {
                        None => Err(DataFusionError::Execution(format!("get indexed field {} not found in struct", k))),
                        Some(col) => Ok(ColumnarValue::Array(col.clone()))
                    }
                }
                (dt, key) => Err(DataFusionError::NotImplemented(format!("get indexed field is only possible on lists with int64 indexes. Tried {:?} with {} index", dt, key))),
            },
            ColumnarValue::Scalar(_) => Err(DataFusionError::NotImplemented(
                "field access is not yet implemented for scalar values".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::physical_plan::expressions::{col, lit};
    use arrow::array::{
        Int64Array, Int64Vec, MutableArray, MutableListArray, MutableUtf8Array,
        StructArray, TryPush,
    };
    use arrow::{array::Utf8Array, datatypes::Field};

    fn get_indexed_field_test(
        list_of_lists: Vec<Vec<Option<&str>>>,
        index: i64,
        expected: Vec<Option<&str>>,
    ) -> Result<()> {
        let schema = list_schema("l");
        //let list_col: Utf8Array<i32> = Utf8Array::from_slice(&list_of_lists);
        let mut list_col: MutableListArray<i32, MutableUtf8Array<i32>> =
            MutableListArray::with_capacity(list_of_lists.len());
        for list in list_of_lists {
            list_col.try_push(Some(list)).unwrap();
        }
        let expr = col("l", &schema).unwrap();
        let batch = RecordBatch::try_new(Arc::new(schema), vec![list_col.as_arc()])?;
        let key = ScalarValue::Int64(Some(index));
        let expr = Arc::new(GetIndexedFieldExpr::new(expr, key));
        let result = expr.evaluate(&batch)?.into_array(batch.num_rows());
        let result = result
            .as_any()
            .downcast_ref::<Utf8Array<i32>>()
            .expect("failed to downcast to StringArray");
        let expected = &Utf8Array::from(expected);
        assert_eq!(expected, result);
        Ok(())
    }

    fn list_schema(col: &str) -> Schema {
        Schema::new(vec![Field::new(
            col,
            DataType::List(Box::new(Field::new("item", DataType::Utf8, true))),
            true,
        )])
    }

    #[test]
    fn get_indexed_field_list() -> Result<()> {
        let list_of_lists = vec![
            vec![Some("a"), Some("b"), None],
            vec![None, Some("c"), Some("d")],
            vec![Some("e"), None, Some("f")],
        ];
        let expected_list = vec![
            vec![Some("a"), None, Some("e")],
            vec![Some("b"), Some("c"), None],
            vec![None, Some("d"), Some("f")],
        ];

        for (i, expected) in expected_list.into_iter().enumerate() {
            get_indexed_field_test(list_of_lists.clone(), i as i64, expected)?;
        }
        Ok(())
    }

    #[test]
    fn get_indexed_field_empty_list() -> Result<()> {
        let schema = list_schema("l");
        let expr = col("l", &schema).unwrap();
        let a = ListArray::<i32>::new_empty(DataType::Utf8);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a)])?;
        let key = ScalarValue::Int64(Some(0));
        let expr = Arc::new(GetIndexedFieldExpr::new(expr, key));
        let result = expr.evaluate(&batch)?.into_array(batch.num_rows());
        assert!(result.is_empty());
        Ok(())
    }

    fn get_indexed_field_test_failure(
        schema: Schema,
        expr: Arc<dyn PhysicalExpr>,
        key: ScalarValue,
        expected: &str,
    ) -> Result<()> {
        let builder = MutableUtf8Array::<i32>::with_capacity(3);
        let mut lb =
            MutableListArray::<i32, MutableUtf8Array<i32>>::new_with_capacity(builder, 0);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![lb.as_arc()])?;
        let expr = Arc::new(GetIndexedFieldExpr::new(expr, key));
        let r = expr.evaluate(&batch).map(|_| ());
        assert!(r.is_err());
        assert_eq!(format!("{}", r.unwrap_err()), expected);
        Ok(())
    }

    #[test]
    fn get_indexed_field_invalid_scalar() -> Result<()> {
        let schema = list_schema("l");
        let expr = lit(ScalarValue::Utf8(Some("a".to_string())));
        get_indexed_field_test_failure(schema, expr,  ScalarValue::Int64(Some(0)), "This feature is not implemented: field access is not yet implemented for scalar values")
    }

    #[test]
    fn get_indexed_field_invalid_list_index() -> Result<()> {
        let schema = list_schema("l");
        let expr = col("l", &schema).unwrap();
        get_indexed_field_test_failure(schema, expr,  ScalarValue::Int8(Some(0)), "This feature is not implemented: get indexed field is only possible on lists with int64 indexes. Tried List(Field { name: \"item\", data_type: Utf8, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: None }) with 0 index")
    }

    fn build_struct(
        fields: Vec<Field>,
        list_of_tuples: Vec<(Option<i64>, Vec<Option<&str>>)>,
    ) -> StructArray {
        let mut foo_builder = Int64Vec::with_capacity(list_of_tuples.len());
        let mut str_builder =
            MutableListArray::<i32, MutableUtf8Array<i32>>::with_capacity(
                list_of_tuples.len(),
            );
        for (int_value, list_value) in list_of_tuples {
            match int_value {
                None => foo_builder.push_null(),
                v => foo_builder.try_push(v).unwrap(),
            }
            str_builder.try_push(Some(list_value)).unwrap();
        }
        let builder = StructArray::from_data(
            DataType::Struct(fields),
            vec![foo_builder.as_arc(), str_builder.as_arc()],
            None,
        );
        builder
    }

    fn get_indexed_field_mixed_test(
        list_of_tuples: Vec<(Option<i64>, Vec<Option<&str>>)>,
        expected_strings: Vec<Vec<Option<&str>>>,
        expected_ints: Vec<Option<i64>>,
    ) -> Result<()> {
        let struct_col = "s";
        let fields = vec![
            Field::new("foo", DataType::Int64, true),
            Field::new(
                "bar",
                DataType::List(Box::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
        ];
        let schema = Schema::new(vec![Field::new(
            struct_col,
            DataType::Struct(fields.clone()),
            true,
        )]);
        let struct_col = build_struct(fields, list_of_tuples.clone());

        let struct_col_expr = col("s", &schema).unwrap();
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(struct_col)])?;

        let int_field_key = ScalarValue::Utf8(Some("foo".to_string()));
        let get_field_expr = Arc::new(GetIndexedFieldExpr::new(
            struct_col_expr.clone(),
            int_field_key,
        ));
        let result = get_field_expr
            .evaluate(&batch)?
            .into_array(batch.num_rows());
        let result = result
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("failed to downcast to Int64Array");
        let expected = &Int64Array::from(expected_ints);
        assert_eq!(expected, result);

        let list_field_key = ScalarValue::Utf8(Some("bar".to_string()));
        let get_list_expr =
            Arc::new(GetIndexedFieldExpr::new(struct_col_expr, list_field_key));
        let result = get_list_expr.evaluate(&batch)?.into_array(batch.num_rows());
        let mut array =
            MutableListArray::<i32, MutableListArray<i32, MutableUtf8Array<i32>>>::new();
        array
            .try_push(Some(list_of_tuples.into_iter().map(|t| Some(t.1))))
            .unwrap();
        let expected = array.as_arc();
        assert_eq!(expected.as_ref(), result.as_ref());

        for (i, expected) in expected_strings.into_iter().enumerate() {
            let get_nested_str_expr = Arc::new(GetIndexedFieldExpr::new(
                get_list_expr.clone(),
                ScalarValue::Int64(Some(i as i64)),
            ));
            let result = get_nested_str_expr
                .evaluate(&batch)?
                .into_array(batch.num_rows());
            let result = result
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .unwrap_or_else(|| {
                    panic!("failed to downcast to StringArray : {:?}", result)
                });
            let expected = &Utf8Array::from(&expected.as_slice());
            assert_eq!(expected, result);
        }
        Ok(())
    }

    #[test]
    fn get_indexed_field_struct() -> Result<()> {
        let list_of_structs = vec![
            (Some(10), vec![Some("a"), Some("b"), None]),
            (Some(15), vec![None, Some("c"), Some("d")]),
            (None, vec![Some("e"), None, Some("f")]),
        ];

        let expected_list = vec![
            vec![Some("a"), None, Some("e")],
            vec![Some("b"), Some("c"), None],
            vec![None, Some("d"), Some("f")],
        ];

        let expected_ints = vec![Some(10), Some(15), None];

        get_indexed_field_mixed_test(
            list_of_structs.clone(),
            expected_list,
            expected_ints,
        )?;
        Ok(())
    }
}
