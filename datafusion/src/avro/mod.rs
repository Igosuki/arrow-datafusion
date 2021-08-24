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

//! This module contains utilities to manipulate avro metadata.

use crate::arrow::datatypes::{DataType, IntervalUnit, Schema, TimeUnit};
use crate::error::Result;
use arrow::datatypes::Field;
use avro_rs::schema::Name;
use avro_rs::types::Value;
use avro_rs::Schema as AvroSchema;
use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::ptr::null;

/// Converts an avro schema to an arrow schema
pub fn to_arrow_schema(avro_schema: &avro_rs::Schema) -> Result<Schema> {
    let mut schema_fields = vec![];
    match avro_schema {
        AvroSchema::Record { fields, .. } => {
            for field in fields {
                schema_fields.push(schema_to_field_with_props(
                    &field.schema,
                    Some(&field.name),
                    false,
                    Some(&external_props(&field.schema)),
                )?)
            }
        }
        schema => schema_fields.push(schema_to_field(schema, Some(""), false)?),
    }

    let schema = Schema::new(schema_fields);
    Ok(schema)
}

fn schema_to_field(
    schema: &avro_rs::Schema,
    name: Option<&str>,
    nullable: bool,
) -> Result<Field> {
    schema_to_field_with_props(schema, name, nullable, Some(&Default::default()))
}

fn schema_to_field_with_props(
    schema: &AvroSchema,
    name: Option<&str>,
    nullable: bool,
    props: Option<&BTreeMap<String, String>>,
) -> Result<Field> {
    let mut nullable = nullable;
    let field_type: DataType = match schema {
        AvroSchema::Null => DataType::Null,
        AvroSchema::Boolean => DataType::Boolean,
        AvroSchema::Int => DataType::Int32,
        AvroSchema::Long => DataType::Int64,
        AvroSchema::Float => DataType::Float32,
        AvroSchema::Double => DataType::Float64,
        AvroSchema::Bytes => DataType::Binary,
        AvroSchema::String => DataType::Utf8,
        AvroSchema::Array(item_schema) => DataType::List(Box::new(
            schema_to_field_with_props(item_schema, None, false, None)?,
        )),
        AvroSchema::Map(value_schema) => {
            let value_field =
                schema_to_field_with_props(value_schema, Some("value"), false, None)?;
            DataType::Dictionary(
                Box::new(DataType::Utf8),
                Box::new(value_field.data_type().clone()),
            )
        }
        AvroSchema::Union(us) => {
            nullable = us.find_schema(&Value::Null).is_some();
            let fields: Result<Vec<Field>> = us
                .variants()
                .into_iter()
                .map(|s| schema_to_field_with_props(&s, None, nullable, None))
                .collect();
            DataType::Union(fields?)
        }
        AvroSchema::Record { name, fields, .. } => {
            let fields: Result<Vec<Field>> = fields
                .into_iter()
                .map(|field| {
                    let mut props = BTreeMap::new();
                    if let Some(doc) = &field.doc {
                        props.insert("doc".to_string(), doc.clone());
                    }
                    /*if let Some(aliases) = fields.aliases {
                        props.insert("aliases", aliases);
                    }*/
                    schema_to_field_with_props(
                        &field.schema,
                        Some(&format!("{}.{}", name.fullname(None), field.name)),
                        false,
                        Some(&props),
                    )
                })
                .collect();
            DataType::Struct(fields?)
        }
        AvroSchema::Enum { symbols, name, .. } => {
            return Ok(Field::new_dict(
                &name.fullname(None),
                index_type(symbols.len()),
                false,
                0,
                false,
            ))
        }
        AvroSchema::Fixed { size, .. } => DataType::FixedSizeBinary(*size as i32),
        AvroSchema::Decimal {
            precision, scale, ..
        } => DataType::Decimal(*precision, *scale),
        AvroSchema::Uuid => DataType::Utf8,
        AvroSchema::Date => DataType::Date32,
        AvroSchema::TimeMillis => DataType::Time32(TimeUnit::Millisecond),
        AvroSchema::TimeMicros => DataType::Time64(TimeUnit::Microsecond),
        AvroSchema::TimestampMillis => DataType::Timestamp(TimeUnit::Millisecond, None),
        AvroSchema::TimestampMicros => DataType::Timestamp(TimeUnit::Microsecond, None),
        AvroSchema::Duration => DataType::Duration(TimeUnit::Millisecond),
    };

    let data_type = field_type.clone();
    let name = name.unwrap_or_else(|| default_field_name(&data_type));

    let mut field = Field::new(name, field_type, nullable);
    field.set_metadata(props.cloned());
    Ok(field)
}

fn default_field_name(dt: &DataType) -> &str {
    match dt {
        DataType::Null => "null",
        DataType::Boolean => "bit",
        DataType::Int8 => "tinyint",
        DataType::Int16 => "smallint",
        DataType::Int32 => "int",
        DataType::Int64 => "bigint",
        DataType::UInt8 => "uint1",
        DataType::UInt16 => "uint2",
        DataType::UInt32 => "uint4",
        DataType::UInt64 => "uint8",
        DataType::Float16 => "float2",
        DataType::Float32 => "float4",
        DataType::Float64 => "float8",
        DataType::Date32 => "dateday",
        DataType::Date64 => "datemilli",
        DataType::Time32(tu) | DataType::Time64(tu) => match tu {
            TimeUnit::Second => "timesec",
            TimeUnit::Millisecond => "timemilli",
            TimeUnit::Microsecond => "timemicro",
            TimeUnit::Nanosecond => "timenano",
        },
        DataType::Timestamp(tu, tz) => {
            if tz.is_some() {
                match tu {
                    TimeUnit::Second => "timestampsectz",
                    TimeUnit::Millisecond => "timestampmillitz",
                    TimeUnit::Microsecond => "timestampmicrotz",
                    TimeUnit::Nanosecond => "timestampnanotz",
                }
            } else {
                match tu {
                    TimeUnit::Second => "timestampsec",
                    TimeUnit::Millisecond => "timestampmilli",
                    TimeUnit::Microsecond => "timestampmicro",
                    TimeUnit::Nanosecond => "timestampnano",
                }
            }
        }
        DataType::Duration(_) => "duration",
        DataType::Interval(unit) => match unit {
            IntervalUnit::YearMonth => "intervalyear",
            IntervalUnit::DayTime => "intervalmonth",
        },
        DataType::Binary => "varbinary",
        DataType::FixedSizeBinary(_) => "fixedsizebinary",
        DataType::LargeBinary => "largevarbinary",
        DataType::Utf8 => "varchar",
        DataType::LargeUtf8 => "largevarchar",
        DataType::List(_) => "list",
        DataType::FixedSizeList(_, _) => "fixed_size_list",
        DataType::LargeList(_) => "largelist",
        DataType::Struct(_) => "struct",
        DataType::Union(_) => "union",
        DataType::Dictionary(_, _) => "map",
        DataType::Decimal(_, _) => "decimal",
    }
}

fn index_type(len: usize) -> DataType {
    if len <= usize::from(u8::MAX) {
        DataType::Int8
    } else if len <= usize::from(u16::MAX) {
        DataType::Int16
    } else if usize::try_from(u32::MAX).map(|i| len < i).unwrap_or(false) {
        DataType::Int32
    } else {
        DataType::Int64
    }
}

fn external_props(schema: &AvroSchema) -> BTreeMap<String, String> {
    let mut props = BTreeMap::new();
    match &schema {
        AvroSchema::Record {
            doc: Some(ref doc), ..
        }
        | AvroSchema::Enum {
            doc: Some(ref doc), ..
        } => {
            props.insert("doc".to_string(), doc.clone());
        }
        _ => {}
    }
    match &schema {
        AvroSchema::Record {
            name:
                Name {
                    aliases: Some(aliases),
                    namespace,
                    ..
                },
            ..
        }
        | AvroSchema::Enum {
            name:
                Name {
                    aliases: Some(aliases),
                    namespace,
                    ..
                },
            ..
        }
        | AvroSchema::Fixed {
            name:
                Name {
                    aliases: Some(aliases),
                    namespace,
                    ..
                },
            ..
        } => {
            let aliases: Vec<String> = aliases
                .into_iter()
                .map(|alias| fullname(alias, namespace.as_ref(), None))
                .collect();
            props.insert("aliases".to_string(), format!("[{}]", aliases.join(",")));
        }
        _ => {}
    }
    props
}

fn get_metadata(
    _schema: AvroSchema,
    props: BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut metadata: BTreeMap<String, String> = Default::default();
    metadata.extend(props);
    return metadata;
}

/// Returns the fully qualified name for a field
pub fn fullname(
    name: &str,
    namespace: Option<&String>,
    default_namespace: Option<&str>,
) -> String {
    if name.contains('.') {
        name.to_string()
    } else {
        let namespace = namespace.as_ref().map(|s| s.as_ref()).or(default_namespace);

        match namespace {
            Some(ref namespace) => format!("{}.{}", namespace, name),
            None => name.to_string(),
        }
    }
}
