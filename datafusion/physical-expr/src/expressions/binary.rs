// This file contains both Apache Software Foundation (ASF) licensed code as
// well as Synnada, Inc. extensions. Changes that constitute Synnada, Inc.
// extensions are available in the SYNNADA-CONTRIBUTIONS.txt file. Synnada, Inc.
// claims copyright only for Synnada, Inc. extensions. The license notice
// applicable to non-Synnada sections of the file is given below.
// --
//
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

mod kernels;

use std::hash::{Hash, Hasher};
use std::{any::Any, sync::Arc};

use crate::array_expressions::{
    array_append, array_concat, array_has_all, array_prepend,
};
use crate::expressions::datum::{apply, apply_cmp};
use crate::intervals::cp_solver::{propagate_arithmetic, propagate_comparison};
use crate::physical_expr::down_cast_any_ref;
use crate::sort_properties::SortProperties;
use crate::PhysicalExpr;

use arrow::array::*;
use arrow::compute::cast;
use arrow::compute::kernels::boolean::{and_kleene, not, or_kleene};
use arrow::compute::kernels::cmp::*;
use arrow::compute::kernels::comparison::regexp_is_match_utf8;
use arrow::compute::kernels::comparison::regexp_is_match_utf8_scalar;
use arrow::compute::kernels::concat_elements::concat_elements_utf8;
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;

use arrow_array::LargeStringArray;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::scalar::{dt_max_ms, dt_min_ms, mdn_max_ns, mdn_min_ns};
use datafusion_common::{internal_err, DataFusionError, Result, ScalarValue};
use datafusion_expr::interval_arithmetic::{apply_operator, Interval};
use datafusion_expr::type_coercion::binary::get_result_type;
use datafusion_expr::{ColumnarValue, Operator};

use kernels::{
    bitwise_and_dyn, bitwise_and_dyn_scalar, bitwise_or_dyn, bitwise_or_dyn_scalar,
    bitwise_shift_left_dyn, bitwise_shift_left_dyn_scalar, bitwise_shift_right_dyn,
    bitwise_shift_right_dyn_scalar, bitwise_xor_dyn, bitwise_xor_dyn_scalar,
};

/// Binary expression
#[derive(Debug, Hash, Clone)]
pub struct BinaryExpr {
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
}

impl BinaryExpr {
    /// Create new binary expression
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        op: Operator,
        right: Arc<dyn PhysicalExpr>,
    ) -> Self {
        Self { left, op, right }
    }

    /// Get the left side of the binary expression
    pub fn left(&self) -> &Arc<dyn PhysicalExpr> {
        &self.left
    }

    /// Get the right side of the binary expression
    pub fn right(&self) -> &Arc<dyn PhysicalExpr> {
        &self.right
    }

    /// Get the operator for this binary expression
    pub fn op(&self) -> &Operator {
        &self.op
    }
}

impl std::fmt::Display for BinaryExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Put parentheses around child binary expressions so that we can see the difference
        // between `(a OR b) AND c` and `a OR (b AND c)`. We only insert parentheses when needed,
        // based on operator precedence. For example, `(a AND b) OR c` and `a AND b OR c` are
        // equivalent and the parentheses are not necessary.

        fn write_child(
            f: &mut std::fmt::Formatter,
            expr: &dyn PhysicalExpr,
            precedence: u8,
        ) -> std::fmt::Result {
            if let Some(child) = expr.as_any().downcast_ref::<BinaryExpr>() {
                let p = child.op.precedence();
                if p == 0 || p < precedence {
                    write!(f, "({child})")?;
                } else {
                    write!(f, "{child}")?;
                }
            } else {
                write!(f, "{expr}")?;
            }

            Ok(())
        }

        let precedence = self.op.precedence();
        write_child(f, self.left.as_ref(), precedence)?;
        write!(f, " {} ", self.op)?;
        write_child(f, self.right.as_ref(), precedence)
    }
}

/// Invoke a compute kernel on a pair of binary data arrays
macro_rules! compute_utf8_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $DT:ident) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast left side array");
        let rr = $RIGHT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast right side array");
        Ok(Arc::new(paste::expr! {[<$OP _utf8>]}(&ll, &rr)?))
    }};
}

macro_rules! binary_string_array_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        match $LEFT.data_type() {
            DataType::Utf8 => compute_utf8_op!($LEFT, $RIGHT, $OP, StringArray),
            DataType::LargeUtf8 => compute_utf8_op!($LEFT, $RIGHT, $OP, LargeStringArray),
            other => internal_err!(
                "Data type {:?} not supported for binary operation '{}' on string arrays",
                other, stringify!($OP)
            ),
        }
    }};
}

/// Invoke a boolean kernel on a pair of arrays
macro_rules! boolean_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        let ll = as_boolean_array($LEFT).expect("boolean_op failed to downcast array");
        let rr = as_boolean_array($RIGHT).expect("boolean_op failed to downcast array");
        Ok(Arc::new($OP(&ll, &rr)?))
    }};
}

macro_rules! binary_string_array_flag_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $NOT:expr, $FLAG:expr) => {{
        match $LEFT.data_type() {
            DataType::Utf8 => {
                compute_utf8_flag_op!($LEFT, $RIGHT, $OP, StringArray, $NOT, $FLAG)
            }
            DataType::LargeUtf8 => {
                compute_utf8_flag_op!($LEFT, $RIGHT, $OP, LargeStringArray, $NOT, $FLAG)
            }
            other => internal_err!(
                "Data type {:?} not supported for binary_string_array_flag_op operation '{}' on string array",
                other, stringify!($OP)
            ),
        }
    }};
}

/// Invoke a compute kernel on a pair of binary data arrays with flags
macro_rules! compute_utf8_flag_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $ARRAYTYPE:ident, $NOT:expr, $FLAG:expr) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$ARRAYTYPE>()
            .expect("compute_utf8_flag_op failed to downcast array");
        let rr = $RIGHT
            .as_any()
            .downcast_ref::<$ARRAYTYPE>()
            .expect("compute_utf8_flag_op failed to downcast array");

        let flag = if $FLAG {
            Some($ARRAYTYPE::from(vec!["i"; ll.len()]))
        } else {
            None
        };
        let mut array = paste::expr! {[<$OP _utf8>]}(&ll, &rr, flag.as_ref())?;
        if $NOT {
            array = not(&array).unwrap();
        }
        Ok(Arc::new(array))
    }};
}

macro_rules! binary_string_array_flag_op_scalar {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $NOT:expr, $FLAG:expr) => {{
        let result: Result<Arc<dyn Array>> = match $LEFT.data_type() {
            DataType::Utf8 => {
                compute_utf8_flag_op_scalar!($LEFT, $RIGHT, $OP, StringArray, $NOT, $FLAG)
            }
            DataType::LargeUtf8 => {
                compute_utf8_flag_op_scalar!($LEFT, $RIGHT, $OP, LargeStringArray, $NOT, $FLAG)
            }
            other => internal_err!(
                "Data type {:?} not supported for binary_string_array_flag_op_scalar operation '{}' on string array",
                other, stringify!($OP)
            ),
        };
        Some(result)
    }};
}

/// Invoke a compute kernel on a data array and a scalar value with flag
macro_rules! compute_utf8_flag_op_scalar {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $ARRAYTYPE:ident, $NOT:expr, $FLAG:expr) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$ARRAYTYPE>()
            .expect("compute_utf8_flag_op_scalar failed to downcast array");

        if let ScalarValue::Utf8(Some(string_value))|ScalarValue::LargeUtf8(Some(string_value)) = $RIGHT {
            let flag = if $FLAG { Some("i") } else { None };
            let mut array =
                paste::expr! {[<$OP _utf8_scalar>]}(&ll, &string_value, flag)?;
            if $NOT {
                array = not(&array).unwrap();
            }
            Ok(Arc::new(array))
        } else {
            internal_err!(
                "compute_utf8_flag_op_scalar failed to cast literal value {} for operation '{}'",
                $RIGHT, stringify!($OP)
            )
        }
    }};
}

impl PhysicalExpr for BinaryExpr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        get_result_type(
            &self.left.data_type(input_schema)?,
            &self.op,
            &self.right.data_type(input_schema)?,
        )
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        Ok(self.left.nullable(input_schema)? || self.right.nullable(input_schema)?)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        use arrow::compute::kernels::numeric::*;

        let lhs = self.left.evaluate(batch)?;
        let rhs = self.right.evaluate(batch)?;
        let left_data_type = lhs.data_type();
        let right_data_type = rhs.data_type();

        let schema = batch.schema();
        let input_schema = schema.as_ref();

        match self.op {
            Operator::Plus => return apply(&lhs, &rhs, add_wrapping),
            Operator::Minus => return apply(&lhs, &rhs, sub_wrapping),
            Operator::Multiply => return apply(&lhs, &rhs, mul_wrapping),
            Operator::Divide => return apply(&lhs, &rhs, div),
            Operator::Modulo => return apply(&lhs, &rhs, rem),
            Operator::Eq => return apply_cmp(&lhs, &rhs, eq),
            Operator::NotEq => return apply_cmp(&lhs, &rhs, neq),
            Operator::Lt | Operator::Gt | Operator::LtEq | Operator::GtEq
            // Assumes the same types are compared, which is being consistent with `temporal_coercion` function.
                if matches!(
                    lhs.data_type(),
                    DataType::Interval(IntervalUnit::DayTime)
                        | DataType::Interval(IntervalUnit::MonthDayNano)
                ) =>
            {
                return apply_interval_cmp(&lhs, &rhs, self.op)
            }
            Operator::Lt => return apply_cmp(&lhs, &rhs, lt),
            Operator::Gt => return apply_cmp(&lhs, &rhs, gt),
            Operator::LtEq => return apply_cmp(&lhs, &rhs, lt_eq),
            Operator::GtEq => return apply_cmp(&lhs, &rhs, gt_eq),
            Operator::IsDistinctFrom => return apply_cmp(&lhs, &rhs, distinct),
            Operator::IsNotDistinctFrom => return apply_cmp(&lhs, &rhs, not_distinct),
            _ => {}
        }

        let result_type = self.data_type(input_schema)?;

        // Attempt to use special kernels if one input is scalar and the other is an array
        let scalar_result = match (&lhs, &rhs) {
            (ColumnarValue::Array(array), ColumnarValue::Scalar(scalar)) => {
                // if left is array and right is literal - use scalar operations
                self.evaluate_array_scalar(array, scalar.clone())?.map(|r| {
                    r.and_then(|a| to_result_type_array(&self.op, a, &result_type))
                })
            }
            (_, _) => None, // default to array implementation
        };

        if let Some(result) = scalar_result {
            return result.map(ColumnarValue::Array);
        }

        // if both arrays or both literals - extract arrays and continue execution
        let (left, right) = (
            lhs.into_array(batch.num_rows())?,
            rhs.into_array(batch.num_rows())?,
        );
        self.evaluate_with_resolved_args(left, &left_data_type, right, &right_data_type)
            .map(ColumnarValue::Array)
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.left.clone(), self.right.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(BinaryExpr::new(
            children[0].clone(),
            self.op,
            children[1].clone(),
        )))
    }

    fn evaluate_bounds(&self, children: &[&Interval]) -> Result<Interval> {
        // Get children intervals:
        let left_interval = children[0];
        let right_interval = children[1];
        // Calculate current node's interval:
        apply_operator(&self.op, left_interval, right_interval)
    }

    fn propagate_constraints(
        &self,
        interval: &Interval,
        children: &[&Interval],
    ) -> Result<Option<Vec<Interval>>> {
        // Get children intervals.
        let left_interval = children[0];
        let right_interval = children[1];

        if self.op.eq(&Operator::And) {
            if interval.eq(&Interval::CERTAINLY_TRUE) {
                // A certainly true logical conjunction can only derive from possibly
                // true operands. Otherwise, we prove infeasability.
                Ok((!left_interval.eq(&Interval::CERTAINLY_FALSE)
                    && !right_interval.eq(&Interval::CERTAINLY_FALSE))
                .then(|| vec![Interval::CERTAINLY_TRUE, Interval::CERTAINLY_TRUE]))
            } else if interval.eq(&Interval::CERTAINLY_FALSE) {
                // If the logical conjunction is certainly false, one of the
                // operands must be false. However, it's not always possible to
                // determine which operand is false, leading to different scenarios.

                // If one operand is certainly true and the other one is uncertain,
                // then the latter must be certainly false.
                if left_interval.eq(&Interval::CERTAINLY_TRUE)
                    && right_interval.eq(&Interval::UNCERTAIN)
                {
                    Ok(Some(vec![
                        Interval::CERTAINLY_TRUE,
                        Interval::CERTAINLY_FALSE,
                    ]))
                } else if right_interval.eq(&Interval::CERTAINLY_TRUE)
                    && left_interval.eq(&Interval::UNCERTAIN)
                {
                    Ok(Some(vec![
                        Interval::CERTAINLY_FALSE,
                        Interval::CERTAINLY_TRUE,
                    ]))
                }
                // If both children are uncertain, or if one is certainly false,
                // we cannot conclusively refine their intervals. In this case,
                // propagation does not result in any interval changes.
                else {
                    Ok(Some(vec![]))
                }
            } else {
                // An uncertain logical conjunction result can not shrink the
                // end-points of its children.
                Ok(Some(vec![]))
            }
        } else if self.op.eq(&Operator::Or) {
            if interval.eq(&Interval::CERTAINLY_FALSE) {
                // A certainly false logical conjunction can only derive from certainly
                // false operands. Otherwise, we prove infeasability.
                Ok((!left_interval.eq(&Interval::CERTAINLY_TRUE)
                    && !right_interval.eq(&Interval::CERTAINLY_TRUE))
                .then(|| vec![Interval::CERTAINLY_FALSE, Interval::CERTAINLY_FALSE]))
            } else if interval.eq(&Interval::CERTAINLY_TRUE) {
                // If the logical disjunction is certainly true, one of the
                // operands must be true. However, it's not always possible to
                // determine which operand is true, leading to different scenarios.

                // If one operand is certainly false and the other one is uncertain,
                // then the latter must be certainly true.
                if left_interval.eq(&Interval::CERTAINLY_FALSE)
                    && right_interval.eq(&Interval::UNCERTAIN)
                {
                    Ok(Some(vec![
                        Interval::CERTAINLY_FALSE,
                        Interval::CERTAINLY_TRUE,
                    ]))
                } else if right_interval.eq(&Interval::CERTAINLY_FALSE)
                    && left_interval.eq(&Interval::UNCERTAIN)
                {
                    Ok(Some(vec![
                        Interval::CERTAINLY_TRUE,
                        Interval::CERTAINLY_FALSE,
                    ]))
                }
                // If both children are uncertain, or if one is certainly true,
                // we cannot conclusively refine their intervals. In this case,
                // propagation does not result in any interval changes.
                else {
                    Ok(Some(vec![]))
                }
            } else {
                // An uncertain logical disjunction result can not shrink the
                // end-points of its children.
                Ok(Some(vec![]))
            }
        } else if self.op.is_comparison_operator() {
            Ok(
                propagate_comparison(&self.op, interval, left_interval, right_interval)?
                    .map(|(left, right)| vec![left, right]),
            )
        } else {
            Ok(
                propagate_arithmetic(&self.op, interval, left_interval, right_interval)?
                    .map(|(left, right)| vec![left, right]),
            )
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        let mut s = state;
        self.hash(&mut s);
    }

    /// For each operator, [`BinaryExpr`] has distinct ordering rules.
    /// TODO: There may be rules specific to some data types (such as division and multiplication on unsigned integers)
    fn get_ordering(&self, children: &[SortProperties]) -> SortProperties {
        let (left_child, right_child) = (&children[0], &children[1]);
        match self.op() {
            Operator::Plus => left_child.add(right_child),
            Operator::Minus => left_child.sub(right_child),
            Operator::Gt | Operator::GtEq => left_child.gt_or_gteq(right_child),
            Operator::Lt | Operator::LtEq => right_child.gt_or_gteq(left_child),
            Operator::And | Operator::Or => left_child.and_or(right_child),
            _ => SortProperties::Unordered,
        }
    }
}

impl PartialEq<dyn Any> for BinaryExpr {
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| self.left.eq(&x.left) && self.op == x.op && self.right.eq(&x.right))
            .unwrap_or(false)
    }
}

/// Casts dictionary array to result type for binary numerical operators. Such operators
/// between array and scalar produce a dictionary array other than primitive array of the
/// same operators between array and array. This leads to inconsistent result types causing
/// errors in the following query execution. For such operators between array and scalar,
/// we cast the dictionary array to primitive array.
fn to_result_type_array(
    op: &Operator,
    array: ArrayRef,
    result_type: &DataType,
) -> Result<ArrayRef> {
    if array.data_type() == result_type {
        Ok(array)
    } else if op.is_numerical_operators() {
        match array.data_type() {
            DataType::Dictionary(_, value_type) => {
                if value_type.as_ref() == result_type {
                    Ok(cast(&array, result_type)?)
                } else {
                    internal_err!(
                            "Incompatible Dictionary value type {value_type:?} with result type {result_type:?} of Binary operator {op:?}"
                        )
                }
            }
            _ => Ok(array),
        }
    } else {
        Ok(array)
    }
}

/// Compares two [`ScalarValue::IntervalDayTime`] or [`ScalarValue::IntervalMonthDayNano`]
/// arrays or scalars. The comparison semantic is that the interval values on both sides
/// of the comparison work as if they reference a common timestamp. If comparing these
/// intervals with respect to this reference gives a definite answer, like 1 month and
/// 1 month + 1 day, answer is given as the result. However, if there is an indefinite
/// case, like 1 month and 30 days, the result will be false for both greater than and
/// less than comparisons.
fn apply_interval_cmp(
    lhs: &ColumnarValue,
    rhs: &ColumnarValue,
    op: Operator,
) -> Result<ColumnarValue> {
    // No need to check data length which is already done.
    let (lhs_min_values, rhs_min_values, lhs_max_values, rhs_max_values) = if lhs
        .data_type()
        .eq(&DataType::Interval(IntervalUnit::DayTime))
    {
        (
            dt_in_millis(lhs, dt_min_ms),
            dt_in_millis(rhs, dt_min_ms),
            dt_in_millis(lhs, dt_max_ms),
            dt_in_millis(rhs, dt_max_ms),
        )
    } else {
        (
            mdn_in_nanos(lhs, mdn_min_ns),
            mdn_in_nanos(rhs, mdn_min_ns),
            mdn_in_nanos(lhs, mdn_max_ns),
            mdn_in_nanos(rhs, mdn_max_ns),
        )
    };
    let eval_op = match op {
        Operator::Lt => lt,
        Operator::LtEq => lt_eq,
        Operator::Gt => gt,
        Operator::GtEq => gt_eq,
        _ => unreachable!(),
    };
    let (min_eval, max_eval) = (
        apply(&lhs_min_values, &rhs_min_values, |l, r| {
            Ok(Arc::new(eval_op(l, r)?))
        })?,
        apply(&lhs_max_values, &rhs_max_values, |l, r| {
            Ok(Arc::new(eval_op(l, r)?))
        })?,
    );
    Ok(definite_interval_cmp(min_eval, max_eval))
}

fn dt_in_millis(dt: &ColumnarValue, f: fn(i64) -> i64) -> ColumnarValue {
    use ColumnarValue::*;
    match dt {
        Array(dt) => Array(Arc::new(PrimitiveArray::<IntervalDayTimeType>::from_iter(
            dt.as_primitive::<IntervalDayTimeType>()
                .iter()
                .map(|dt| dt.map(f)),
        ))),
        Scalar(ScalarValue::IntervalDayTime(dt)) => {
            Scalar(ScalarValue::IntervalDayTime(dt.map(f)))
        }
        _ => unreachable!(),
    }
}

fn mdn_in_nanos(mdn: &ColumnarValue, f: fn(i128) -> i128) -> ColumnarValue {
    use ColumnarValue::*;
    match mdn {
        Array(mdn) => Array(Arc::new(
            PrimitiveArray::<IntervalMonthDayNanoType>::from_iter(
                mdn.as_primitive::<IntervalMonthDayNanoType>()
                    .iter()
                    .map(|mdn| mdn.map(f)),
            ),
        )),
        Scalar(ScalarValue::IntervalMonthDayNano(mdn)) => {
            Scalar(ScalarValue::IntervalMonthDayNano(mdn.map(f)))
        }
        _ => unreachable!(),
    }
}

fn definite_interval_cmp(v1: ColumnarValue, v2: ColumnarValue) -> ColumnarValue {
    use ColumnarValue::*;
    match (v1, v2) {
        (Array(v1), Array(v2)) => Array(Arc::new(BooleanArray::from_iter(
            v1.as_boolean()
                .iter()
                .zip(v2.as_boolean())
                .map(|(v1, v2)| v1.and_then(|v1| v2.map(|v2| v1 & v2))),
        ))),
        (Scalar(ScalarValue::Boolean(v1)), Scalar(ScalarValue::Boolean(v2))) => {
            Scalar(ScalarValue::Boolean(v1.and_then(|v1| v2.map(|v2| v1 & v2))))
        }
        _ => unreachable!(),
    }
}

impl BinaryExpr {
    /// Evaluate the expression of the left input is an array and
    /// right is literal - use scalar operations
    fn evaluate_array_scalar(
        &self,
        array: &dyn Array,
        scalar: ScalarValue,
    ) -> Result<Option<Result<ArrayRef>>> {
        use Operator::*;
        let scalar_result = match &self.op {
            RegexMatch => binary_string_array_flag_op_scalar!(
                array,
                scalar,
                regexp_is_match,
                false,
                false
            ),
            RegexIMatch => binary_string_array_flag_op_scalar!(
                array,
                scalar,
                regexp_is_match,
                false,
                true
            ),
            RegexNotMatch => binary_string_array_flag_op_scalar!(
                array,
                scalar,
                regexp_is_match,
                true,
                false
            ),
            RegexNotIMatch => binary_string_array_flag_op_scalar!(
                array,
                scalar,
                regexp_is_match,
                true,
                true
            ),
            BitwiseAnd => bitwise_and_dyn_scalar(array, scalar),
            BitwiseOr => bitwise_or_dyn_scalar(array, scalar),
            BitwiseXor => bitwise_xor_dyn_scalar(array, scalar),
            BitwiseShiftRight => bitwise_shift_right_dyn_scalar(array, scalar),
            BitwiseShiftLeft => bitwise_shift_left_dyn_scalar(array, scalar),
            // if scalar operation is not supported - fallback to array implementation
            _ => None,
        };

        Ok(scalar_result)
    }

    fn evaluate_with_resolved_args(
        &self,
        left: Arc<dyn Array>,
        left_data_type: &DataType,
        right: Arc<dyn Array>,
        right_data_type: &DataType,
    ) -> Result<ArrayRef> {
        use Operator::*;
        match &self.op {
            IsDistinctFrom | IsNotDistinctFrom | Lt | LtEq | Gt | GtEq | Eq | NotEq
            | Plus | Minus | Multiply | Divide | Modulo => unreachable!(),
            And => {
                if left_data_type == &DataType::Boolean {
                    boolean_op!(&left, &right, and_kleene)
                } else {
                    internal_err!(
                        "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                        self.op,
                        left.data_type(),
                        right.data_type()
                    )
                }
            }
            Or => {
                if left_data_type == &DataType::Boolean {
                    boolean_op!(&left, &right, or_kleene)
                } else {
                    internal_err!(
                        "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                        self.op,
                        left_data_type,
                        right_data_type
                    )
                }
            }
            RegexMatch => {
                binary_string_array_flag_op!(left, right, regexp_is_match, false, false)
            }
            RegexIMatch => {
                binary_string_array_flag_op!(left, right, regexp_is_match, false, true)
            }
            RegexNotMatch => {
                binary_string_array_flag_op!(left, right, regexp_is_match, true, false)
            }
            RegexNotIMatch => {
                binary_string_array_flag_op!(left, right, regexp_is_match, true, true)
            }
            BitwiseAnd => bitwise_and_dyn(left, right),
            BitwiseOr => bitwise_or_dyn(left, right),
            BitwiseXor => bitwise_xor_dyn(left, right),
            BitwiseShiftRight => bitwise_shift_right_dyn(left, right),
            BitwiseShiftLeft => bitwise_shift_left_dyn(left, right),
            StringConcat => match (left_data_type, right_data_type) {
                (DataType::List(_), DataType::List(_)) => array_concat(&[left, right]),
                (DataType::List(_), _) => array_append(&[left, right]),
                (_, DataType::List(_)) => array_prepend(&[left, right]),
                _ => binary_string_array_op!(left, right, concat_elements),
            },
            AtArrow => array_has_all(&[left, right]),
            ArrowAt => array_has_all(&[right, left]),
        }
    }
}

/// Create a binary expression whose arguments are correctly coerced.
/// This function errors if it is not possible to coerce the arguments
/// to computational types supported by the operator.
pub fn binary(
    lhs: Arc<dyn PhysicalExpr>,
    op: Operator,
    rhs: Arc<dyn PhysicalExpr>,
    _input_schema: &Schema,
) -> Result<Arc<dyn PhysicalExpr>> {
    Ok(Arc::new(BinaryExpr::new(lhs, op, rhs)))
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::{HashMap, HashSet};
    use std::ops::Neg;
    use std::sync::Arc;

    use super::*;
    use crate::expressions::{col, lit, Column};
    use crate::expressions::{try_cast, Literal};
    use arrow::datatypes::{
        ArrowNumericType, Decimal128Type, Field, Int32Type, SchemaRef,
    };
    use arrow_schema::ArrowError;
    use datafusion_common::Result;
    use datafusion_expr::type_coercion::binary::get_input_types;

    /// Since the implementation is not actively used yet, it has been moved into the test module.
    /// It will be moved back to the source module of the BinaryExpr as soon as it is put into use.
    struct BinaryExprEquivalenceChecker<'a> {
        column_types: &'a HashMap<Column, DataType>,
        is_map_valid: bool,
        mutable_order_of_floats: &'a bool,
    }

    #[derive(Eq, PartialEq, Hash, Clone, Debug, Copy)]
    enum OperatorInfo {
        Plus,
        Minus,
        Multiply,
        Divide,
    }

    impl Neg for OperatorInfo {
        type Output = Self;
        fn neg(self) -> Self::Output {
            match self {
                OperatorInfo::Minus => OperatorInfo::Plus,
                OperatorInfo::Plus => OperatorInfo::Minus,
                OperatorInfo::Multiply => OperatorInfo::Divide,
                OperatorInfo::Divide => OperatorInfo::Multiply,
            }
        }
    }

    impl<'a> BinaryExprEquivalenceChecker<'a> {
        pub fn is_equal(
            this: &BinaryExpr,
            other: &BinaryExpr,
            column_types: &'a Option<HashMap<Column, DataType>>,
            mutable_order_of_floats: &'a bool,
        ) -> Result<bool> {
            let empty_map = HashMap::new();
            let checker = if let Some(map) = column_types {
                BinaryExprEquivalenceChecker {
                    column_types: map,
                    is_map_valid: true,
                    mutable_order_of_floats,
                }
            } else {
                BinaryExprEquivalenceChecker {
                    column_types: &empty_map,
                    is_map_valid: false,
                    mutable_order_of_floats,
                }
            };

            checker.compare(this, other)
        }

        /// This function takes two `BinaryExpr`s to determine whether they are equivalent.
        /// The `column_types` map specifies data types of the columns. If this map is `None`,
        /// we treat the expressions as mathematical formulas and check for equality of form.
        /// If it is `Some`, we consider the impact of data types, meaning integers and floats
        /// have their own rules.
        fn compare(&self, this: &BinaryExpr, other: &BinaryExpr) -> Result<bool> {
            if self.is_map_valid {
                let this_float = self.contains_float_value(this);
                let other_float = self.contains_float_value(other);
                let possible_integers =
                    !(this_float.unwrap_or(false) && other_float.unwrap_or(false));
                let consider_float = !self.mutable_order_of_floats
                    && (this_float.unwrap_or(true) || other_float.unwrap_or(true));
                // When we are given column data types, we need to be careful when
                // deciding whether we can consider equivalent term orderings. Due
                // to how data types behave, we can not allow a change in the order
                // of operations when:
                // - There is a division between two integers, or
                // - There are floating-point values in the expression unless we
                //   are explicitly allowed to ignore floating-point intricacies.
                if (((this.op == Operator::Divide) || (other.op == Operator::Divide))
                    && possible_integers)
                    || ((matches!(
                        this.op,
                        Operator::Plus
                            | Operator::Minus
                            | Operator::Multiply
                            | Operator::Divide
                    ) || matches!(
                        other.op,
                        Operator::Plus
                            | Operator::Minus
                            | Operator::Multiply
                            | Operator::Divide
                    )) && consider_float)
                {
                    return self.check_children_separately(this, other);
                }
            }
            // At this point, we know we can consider various term ordering equivalences.
            // First, check if the two expressions have the same operator at the root:
            if this.op == other.op {
                // These operators can swap their children, even in deeper levels.
                // For example, (1 * (2 * 3)) = (2 * (3 * 1)).
                if matches!(
                    this.op,
                    Operator::And | Operator::Or | Operator::Eq | Operator::Multiply
                ) {
                    // NOTE: The Plus operator is also in this category but it is handled
                    //       differently, because there may be some Minus operators in between
                    //       Plus operators.
                    // NOTE: The reason Multiply appears here is that its behaviour changes for
                    //       integers and floats -- consider the cases where we have a sequence
                    //       of Multiply operators vs. the case where there Multiply and Divide
                    //       operators appear in mixed order.
                    Ok(self.resolve_interchangable_ops(&this.op, this, other))
                } else if matches!(this.op, Operator::Plus | Operator::Minus) {
                    self.resolve_reciprocal(this, other, Operator::Plus)
                }
                // Order of terms involving integers can be changed freely under the
                // Multiply operator. However, the cases involving Divide should be
                // handled like the Plus and Minus cases.
                else if matches!(this.op, Operator::Divide)
                    || matches!(other.op, Operator::Divide)
                {
                    self.resolve_reciprocal(this, other, Operator::Multiply)
                }
                // Base case, check whether corresponding sides are equal:
                else {
                    return self.check_children_separately(this, other);
                }
            }
            // We have distinct operators at expression roots.
            else {
                // Comparison operators can be changed symmetrically.
                if this.op.is_comparison_operator() && this.op.swap() == Some(other.op) {
                    // Check whether swapped expressions are equal:
                    let first_match = self.is_exprs_equal(&this.left, &other.right)?;
                    let second_match = self.is_exprs_equal(&this.right, &other.left)?;
                    Ok(first_match && second_match)
                }
                // One side is Plus, the other side is Minus:
                else if matches!(this.op, Operator::Plus | Operator::Minus)
                    && matches!(other.op, Operator::Plus | Operator::Minus)
                {
                    self.resolve_reciprocal(this, other, Operator::Plus)
                }
                // One side is Multiply, the other side is Divide:
                else if matches!(this.op, Operator::Multiply | Operator::Divide)
                    && matches!(other.op, Operator::Divide | Operator::Multiply)
                {
                    self.resolve_reciprocal(this, other, Operator::Multiply)
                } else {
                    Ok(false)
                }
            }
        }

        /// This function *separately* checks left and right children of the given
        /// expressions for equality without considering any cross interactions.
        fn check_children_separately(
            &self,
            this_expr: &BinaryExpr,
            other_expr: &BinaryExpr,
        ) -> Result<bool> {
            let left_check = self.is_exprs_equal(this_expr.left(), other_expr.left())?;
            let right_check =
                self.is_exprs_equal(this_expr.right(), other_expr.right())?;

            Ok(left_check && right_check)
        }

        /// This function determines whether the expression promotes a floating point
        /// value eventually. If it does, it returns `Some(True)`. If there are
        /// columns whose datatypes are not available in the `column_types` map,
        /// and there is an uncertain possibility, it returns `None`. In such cases,
        /// the decision is made at the outer scope according to the operation.
        fn contains_float_value(&self, expr: &BinaryExpr) -> Option<bool> {
            match (
                self.expr_contains_float_value(&expr.left),
                self.expr_contains_float_value(&expr.right),
            ) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            }
        }

        /// Checks whether the given expression are equal. This function considers
        /// various equivalent term orderings when making this decision. For example,
        /// 1 * (2 * 3) is equal to 2 * (1 * 3) according to this function.
        fn is_exprs_equal(
            &self,
            lhs: &Arc<dyn PhysicalExpr>,
            rhs: &Arc<dyn PhysicalExpr>,
        ) -> Result<bool> {
            match (
                lhs.as_any().downcast_ref::<BinaryExpr>(),
                rhs.as_any().downcast_ref::<BinaryExpr>(),
            ) {
                (Some(lhs), Some(rhs)) => self.compare(lhs, rhs),
                _ => Ok(lhs.eq(rhs)),
            }
        }

        /// Finds whether one of the leaf expressions have a floating-point type.
        /// A return value of `None` means unknown type.
        fn expr_contains_float_value(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
        ) -> Option<bool> {
            match (
                expr.as_any().downcast_ref::<Literal>(),
                expr.as_any().downcast_ref::<BinaryExpr>(),
                expr.as_any().downcast_ref::<Column>(),
            ) {
                (Some(literal), _, _) => Some(literal.value().data_type().is_floating()),
                (_, Some(binary), _) => self.contains_float_value(binary),
                (_, _, Some(column)) => self
                    .column_types
                    .get(column)
                    .map(|datatype| datatype.is_floating()),
                _ => None,
            }
        }

        /// This function searches for the expression `expr` in the given list of
        /// expressions (`expr_list`) and returns its index. If the expression is
        /// not present in the list, it returns `None`.
        fn find_match_idx(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            expr_list: &[Arc<dyn PhysicalExpr>],
        ) -> Option<usize> {
            if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
                // Search the binary expression `binary` inside the `exprs` vector,
                // find its index if it exists:
                expr_list.iter().position(|expr| {
                    expr.as_any()
                        .downcast_ref::<BinaryExpr>()
                        .map(|binary_other| self.compare(binary, binary_other))
                        .unwrap_or(Ok(false))
                        .unwrap_or(false)
                })
            } else {
                // Search the PhysicalExpr `i_expr` inside `rhs_and_children`. Find its index if exists
                expr_list.iter().position(|item| item.eq(expr))
            }
        }

        /// This function checks the "equality" of PhysicalExpr arrays, ignoring the
        /// order of elements. All elements must match one to one and none of the
        /// elements can be left out.
        fn check_match(
            &self,
            expr_vec1: &[Arc<dyn PhysicalExpr>],
            expr_vec2: &[Arc<dyn PhysicalExpr>],
        ) -> bool {
            if expr_vec1.len() != expr_vec2.len() {
                return false;
            }
            let mut matching_indexes = HashSet::new();
            for i_expr in expr_vec1.iter() {
                let match_idx = self.find_match_idx(i_expr, expr_vec2);
                match_idx.map(|idx| matching_indexes.insert(idx));
            }
            expr_vec2.len() == matching_indexes.len()
        }

        /// This function collects all children of the expressions having
        /// a sequence of the same operators, like And, Or, Equal, and Multiply.
        /// If the content of these left and right collections are equal,
        /// it returns true; otherwise, returns false.
        fn resolve_interchangable_ops(
            &self,
            op: &Operator,
            lhs: &BinaryExpr,
            rhs: &BinaryExpr,
        ) -> bool {
            let mut lhs_children = Vec::<Arc<dyn PhysicalExpr>>::new();
            let mut rhs_children = Vec::<Arc<dyn PhysicalExpr>>::new();

            // Children of unified operations are collected.
            collect_op_child(op, &mut lhs_children, lhs.left.clone());
            collect_op_child(op, &mut lhs_children, lhs.right.clone());

            collect_op_child(op, &mut rhs_children, rhs.left.clone());
            collect_op_child(op, &mut rhs_children, rhs.right.clone());

            // Their equalities are checked, ignoring the orders.
            self.check_match(&lhs_children, &rhs_children)
        }

        /// This function determines all reciprocal posibilities of + and -, or * and /
        /// operator orderings. + and - operators also consider the sign of the literal
        /// in the expression.
        pub fn resolve_reciprocal(
            &self,
            lhs_expr: &BinaryExpr,
            rhs_expr: &BinaryExpr,
            op: Operator,
        ) -> Result<bool> {
            // The "positive" vector holds the expressions with a plus sign in
            // addition/subtraction mode, or the expressions in the numerator in
            // multiplication/division mode.
            let (mut lhs_positive_vec, mut lhs_negative_vec) = (Vec::new(), Vec::new());
            // The "negative" vector holds the expressions with a minus sign in
            // addition/subtraction mode, or the expressions in the denominator in
            // multiplication/division mode.
            let (mut rhs_positive_vec, mut rhs_negative_vec) = (Vec::new(), Vec::new());

            let op_info = if op == Operator::Plus {
                OperatorInfo::Plus
            } else if op == Operator::Multiply {
                OperatorInfo::Multiply
            } else {
                return Err(DataFusionError::Internal(
                    "Undefined operator for binary equivalence".to_string(),
                ));
            };
            get_reciprocal_vecs(
                Arc::from(lhs_expr.clone()),
                &mut lhs_positive_vec,
                &mut lhs_negative_vec,
                op_info,
            )?;
            get_reciprocal_vecs(
                Arc::from(rhs_expr.clone()),
                &mut rhs_positive_vec,
                &mut rhs_negative_vec,
                op_info,
            )?;

            Ok(self.check_match(&lhs_negative_vec, &rhs_negative_vec)
                && self.check_match(&lhs_positive_vec, &rhs_positive_vec))
        }
    }

    /// Collects all the leaf expressions separated by `op` operator. For instance,
    /// `(a and b) and c` will collect `[a, b, c]`.
    fn collect_op_child(
        op: &Operator,
        container: &mut Vec<Arc<dyn PhysicalExpr>>,
        child: Arc<dyn PhysicalExpr>,
    ) {
        if let Some(binary) = child.as_any().downcast_ref::<BinaryExpr>() {
            if binary.op != *op {
                container.push(child);
            } else {
                collect_op_child(op, container, binary.left.clone());
                collect_op_child(op, container, binary.right.clone());
            }
        } else {
            container.push(child);
        }
    }

    /// This is a helper function for both Plus/Minus and Multiply/Divide modes.
    /// It either collects Plus/Minus operands, or collects Multiply/Divide operands
    /// (positives are Plus and Multiply, negatives are Minus and Divide).
    fn get_reciprocal_vecs(
        expr: Arc<dyn PhysicalExpr>,
        positive_vec: &mut Vec<Arc<dyn PhysicalExpr>>,
        negative_vec: &mut Vec<Arc<dyn PhysicalExpr>>,
        op_info: OperatorInfo,
    ) -> Result<()> {
        // The paramter `op_info` determines the mode. + and - operators must be
        // differentiated from * and / opreators across the whole expression.
        if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
            if (matches!(binary_expr.op(), Operator::Plus | Operator::Minus)
                && matches!(op_info, OperatorInfo::Plus | OperatorInfo::Minus))
                || (matches!(binary_expr.op(), Operator::Multiply | Operator::Divide)
                    && matches!(op_info, OperatorInfo::Multiply | OperatorInfo::Divide))
            {
                get_reciprocal_vecs(
                    binary_expr.left().clone(),
                    positive_vec,
                    negative_vec,
                    op_info,
                )?;
                let neg_op =
                    if matches!(binary_expr.op(), Operator::Plus | Operator::Multiply) {
                        op_info
                    } else {
                        -op_info
                    };
                get_reciprocal_vecs(
                    binary_expr.right().clone(),
                    positive_vec,
                    negative_vec,
                    neg_op,
                )?;
            } else {
                match op_info {
                    OperatorInfo::Plus | OperatorInfo::Multiply => {
                        positive_vec.push(Arc::new(binary_expr.clone()))
                    }
                    OperatorInfo::Minus | OperatorInfo::Divide => {
                        negative_vec.push(Arc::new(binary_expr.clone()))
                    }
                }
            }
        } else if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
            // Literal signs are only considered in + or - operations.
            if matches!(op_info, OperatorInfo::Plus | OperatorInfo::Minus) {
                let zero = &ScalarValue::new_zero(&literal.value().data_type())?;
                match (op_info, literal.value().partial_cmp(zero)) {
                    (
                        OperatorInfo::Plus,
                        Some(Ordering::Greater) | Some(Ordering::Equal),
                    ) => positive_vec.push(Arc::new(literal.clone())),
                    (OperatorInfo::Plus, Some(Ordering::Less)) => {
                        let zero = ScalarValue::new_zero(&literal.value().data_type())?;
                        let result = zero.sub(literal.value())?;
                        negative_vec.push(Arc::new(Literal::new(result)));
                    }
                    (
                        OperatorInfo::Minus,
                        Some(Ordering::Less) | Some(Ordering::Equal),
                    ) => {
                        let zero = ScalarValue::new_zero(&literal.value().data_type())?;
                        let result = zero.sub(literal.value())?;
                        positive_vec.push(Arc::new(Literal::new(result)));
                    }
                    (OperatorInfo::Minus, Some(Ordering::Greater)) => {
                        negative_vec.push(Arc::new(literal.clone()));
                    }
                    (_, _) => {
                        return Err(DataFusionError::Execution(format!(
                            "Ordering couldn't be calculated between {:?} and {:?}",
                            literal.value(),
                            zero
                        )))
                    }
                }
            }
        } else if op_info == OperatorInfo::Plus || op_info == OperatorInfo::Multiply {
            positive_vec.push(expr);
        } else if op_info == OperatorInfo::Minus || op_info == OperatorInfo::Divide {
            negative_vec.push(expr);
        }
        Ok(())
    }

    /// Performs a binary operation, applying any type coercion necessary
    fn binary_op(
        left: Arc<dyn PhysicalExpr>,
        op: Operator,
        right: Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let left_type = left.data_type(schema)?;
        let right_type = right.data_type(schema)?;
        let (lhs, rhs) = get_input_types(&left_type, &op, &right_type)?;

        let left_expr = try_cast(left, schema, lhs)?;
        let right_expr = try_cast(right, schema, rhs)?;
        binary(left_expr, op, right_expr, schema)
    }

    #[test]
    fn binary_comparison() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![1, 2, 4, 8, 16]);

        // expression: "a < b"
        let lt = binary(
            col("a", &schema)?,
            Operator::Lt,
            col("b", &schema)?,
            &schema,
        )?;
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)])?;

        let result = lt
            .evaluate(&batch)?
            .into_array(batch.num_rows())
            .expect("Failed to convert to array");
        assert_eq!(result.len(), 5);

        let expected = [false, false, true, true, true];
        let result =
            as_boolean_array(&result).expect("failed to downcast to BooleanArray");
        for (i, &expected_item) in expected.iter().enumerate().take(5) {
            assert_eq!(result.value(i), expected_item);
        }

        Ok(())
    }

    #[test]
    fn binary_nested() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let a = Int32Array::from(vec![2, 4, 6, 8, 10]);
        let b = Int32Array::from(vec![2, 5, 4, 8, 8]);

        // expression: "a < b OR a == b"
        let expr = binary(
            binary(
                col("a", &schema)?,
                Operator::Lt,
                col("b", &schema)?,
                &schema,
            )?,
            Operator::Or,
            binary(
                col("a", &schema)?,
                Operator::Eq,
                col("b", &schema)?,
                &schema,
            )?,
            &schema,
        )?;
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)])?;

        assert_eq!("a@0 < b@1 OR a@0 = b@1", format!("{expr}"));

        let result = expr
            .evaluate(&batch)?
            .into_array(batch.num_rows())
            .expect("Failed to convert to array");
        assert_eq!(result.len(), 5);

        let expected = [true, true, false, true, false];
        let result =
            as_boolean_array(&result).expect("failed to downcast to BooleanArray");
        for (i, &expected_item) in expected.iter().enumerate().take(5) {
            assert_eq!(result.value(i), expected_item);
        }

        Ok(())
    }

    // runs an end-to-end test of physical type coercion:
    // 1. construct a record batch with two columns of type A and B
    //  (*_ARRAY is the Rust Arrow array type, and *_TYPE is the DataType of the elements)
    // 2. construct a physical expression of A OP B
    // 3. evaluate the expression
    // 4. verify that the resulting expression is of type C
    // 5. verify that the results of evaluation are $VEC
    macro_rules! test_coercion {
        ($A_ARRAY:ident, $A_TYPE:expr, $A_VEC:expr, $B_ARRAY:ident, $B_TYPE:expr, $B_VEC:expr, $OP:expr, $C_ARRAY:ident, $C_TYPE:expr, $VEC:expr,) => {{
            let schema = Schema::new(vec![
                Field::new("a", $A_TYPE, false),
                Field::new("b", $B_TYPE, false),
            ]);
            let a = $A_ARRAY::from($A_VEC);
            let b = $B_ARRAY::from($B_VEC);
            let (lhs, rhs) = get_input_types(&$A_TYPE, &$OP, &$B_TYPE)?;

            let left = try_cast(col("a", &schema)?, &schema, lhs)?;
            let right = try_cast(col("b", &schema)?, &schema, rhs)?;

            // verify that we can construct the expression
            let expression = binary(left, $OP, right, &schema)?;
            let batch = RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![Arc::new(a), Arc::new(b)],
            )?;

            // verify that the expression's type is correct
            assert_eq!(expression.data_type(&schema)?, $C_TYPE);

            // compute
            let result = expression.evaluate(&batch)?.into_array(batch.num_rows()).expect("Failed to convert to array");

            // verify that the array's data_type is correct
            assert_eq!(*result.data_type(), $C_TYPE);

            // verify that the data itself is downcastable
            let result = result
                .as_any()
                .downcast_ref::<$C_ARRAY>()
                .expect("failed to downcast");
            // verify that the result itself is correct
            for (i, x) in $VEC.iter().enumerate() {
                let v = result.value(i);
                assert_eq!(
                    v,
                    *x,
                    "Unexpected output at position {i}:\n\nActual:\n{v}\n\nExpected:\n{x}"
                );
            }
        }};
    }

    #[test]
    fn test_type_coercion() -> Result<()> {
        test_coercion!(
            Int32Array,
            DataType::Int32,
            vec![1i32, 2i32],
            UInt32Array,
            DataType::UInt32,
            vec![1u32, 2u32],
            Operator::Plus,
            Int32Array,
            DataType::Int32,
            [2i32, 4i32],
        );
        test_coercion!(
            Int32Array,
            DataType::Int32,
            vec![1i32],
            UInt16Array,
            DataType::UInt16,
            vec![1u16],
            Operator::Plus,
            Int32Array,
            DataType::Int32,
            [2i32],
        );
        test_coercion!(
            Float32Array,
            DataType::Float32,
            vec![1f32],
            UInt16Array,
            DataType::UInt16,
            vec![1u16],
            Operator::Plus,
            Float32Array,
            DataType::Float32,
            [2f32],
        );
        test_coercion!(
            Float32Array,
            DataType::Float32,
            vec![2f32],
            UInt16Array,
            DataType::UInt16,
            vec![1u16],
            Operator::Multiply,
            Float32Array,
            DataType::Float32,
            [2f32],
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["1994-12-13", "1995-01-26"],
            Date32Array,
            DataType::Date32,
            vec![9112, 9156],
            Operator::Eq,
            BooleanArray,
            DataType::Boolean,
            [true, true],
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["1994-12-13", "1995-01-26"],
            Date32Array,
            DataType::Date32,
            vec![9113, 9154],
            Operator::Lt,
            BooleanArray,
            DataType::Boolean,
            [true, false],
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["1994-12-13T12:34:56", "1995-01-26T01:23:45"],
            Date64Array,
            DataType::Date64,
            vec![787322096000, 791083425000],
            Operator::Eq,
            BooleanArray,
            DataType::Boolean,
            [true, true],
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["1994-12-13T12:34:56", "1995-01-26T01:23:45"],
            Date64Array,
            DataType::Date64,
            vec![787322096001, 791083424999],
            Operator::Lt,
            BooleanArray,
            DataType::Boolean,
            [true, false],
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["abc"; 5],
            StringArray,
            DataType::Utf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexMatch,
            BooleanArray,
            DataType::Boolean,
            [true, false, true, false, false],
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["abc"; 5],
            StringArray,
            DataType::Utf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexIMatch,
            BooleanArray,
            DataType::Boolean,
            [true, true, true, true, false],
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["abc"; 5],
            StringArray,
            DataType::Utf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexNotMatch,
            BooleanArray,
            DataType::Boolean,
            [false, true, false, true, true],
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["abc"; 5],
            StringArray,
            DataType::Utf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexNotIMatch,
            BooleanArray,
            DataType::Boolean,
            [false, false, false, false, true],
        );
        test_coercion!(
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["abc"; 5],
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexMatch,
            BooleanArray,
            DataType::Boolean,
            [true, false, true, false, false],
        );
        test_coercion!(
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["abc"; 5],
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexIMatch,
            BooleanArray,
            DataType::Boolean,
            [true, true, true, true, false],
        );
        test_coercion!(
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["abc"; 5],
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexNotMatch,
            BooleanArray,
            DataType::Boolean,
            [false, true, false, true, true],
        );
        test_coercion!(
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["abc"; 5],
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexNotIMatch,
            BooleanArray,
            DataType::Boolean,
            [false, false, false, false, true],
        );
        test_coercion!(
            Int16Array,
            DataType::Int16,
            vec![1i16, 2i16, 3i16],
            Int64Array,
            DataType::Int64,
            vec![10i64, 4i64, 5i64],
            Operator::BitwiseAnd,
            Int64Array,
            DataType::Int64,
            [0i64, 0i64, 1i64],
        );
        test_coercion!(
            UInt16Array,
            DataType::UInt16,
            vec![1u16, 2u16, 3u16],
            UInt64Array,
            DataType::UInt64,
            vec![10u64, 4u64, 5u64],
            Operator::BitwiseAnd,
            UInt64Array,
            DataType::UInt64,
            [0u64, 0u64, 1u64],
        );
        test_coercion!(
            Int16Array,
            DataType::Int16,
            vec![3i16, 2i16, 3i16],
            Int64Array,
            DataType::Int64,
            vec![10i64, 6i64, 5i64],
            Operator::BitwiseOr,
            Int64Array,
            DataType::Int64,
            [11i64, 6i64, 7i64],
        );
        test_coercion!(
            UInt16Array,
            DataType::UInt16,
            vec![1u16, 2u16, 3u16],
            UInt64Array,
            DataType::UInt64,
            vec![10u64, 4u64, 5u64],
            Operator::BitwiseOr,
            UInt64Array,
            DataType::UInt64,
            [11u64, 6u64, 7u64],
        );
        test_coercion!(
            Int16Array,
            DataType::Int16,
            vec![3i16, 2i16, 3i16],
            Int64Array,
            DataType::Int64,
            vec![10i64, 6i64, 5i64],
            Operator::BitwiseXor,
            Int64Array,
            DataType::Int64,
            [9i64, 4i64, 6i64],
        );
        test_coercion!(
            UInt16Array,
            DataType::UInt16,
            vec![3u16, 2u16, 3u16],
            UInt64Array,
            DataType::UInt64,
            vec![10u64, 6u64, 5u64],
            Operator::BitwiseXor,
            UInt64Array,
            DataType::UInt64,
            [9u64, 4u64, 6u64],
        );
        test_coercion!(
            Int16Array,
            DataType::Int16,
            vec![4i16, 27i16, 35i16],
            Int64Array,
            DataType::Int64,
            vec![2i64, 3i64, 4i64],
            Operator::BitwiseShiftRight,
            Int64Array,
            DataType::Int64,
            [1i64, 3i64, 2i64],
        );
        test_coercion!(
            UInt16Array,
            DataType::UInt16,
            vec![4u16, 27u16, 35u16],
            UInt64Array,
            DataType::UInt64,
            vec![2u64, 3u64, 4u64],
            Operator::BitwiseShiftRight,
            UInt64Array,
            DataType::UInt64,
            [1u64, 3u64, 2u64],
        );
        test_coercion!(
            Int16Array,
            DataType::Int16,
            vec![2i16, 3i16, 4i16],
            Int64Array,
            DataType::Int64,
            vec![4i64, 12i64, 7i64],
            Operator::BitwiseShiftLeft,
            Int64Array,
            DataType::Int64,
            [32i64, 12288i64, 512i64],
        );
        test_coercion!(
            UInt16Array,
            DataType::UInt16,
            vec![2u16, 3u16, 4u16],
            UInt64Array,
            DataType::UInt64,
            vec![4u64, 12u64, 7u64],
            Operator::BitwiseShiftLeft,
            UInt64Array,
            DataType::UInt64,
            [32u64, 12288u64, 512u64],
        );
        Ok(())
    }

    // Note it would be nice to use the same test_coercion macro as
    // above, but sadly the type of the values of the dictionary are
    // not encoded in the rust type of the DictionaryArray. Thus there
    // is no way at the time of this writing to create a dictionary
    // array using the `From` trait
    #[test]
    fn test_dictionary_type_to_array_coercion() -> Result<()> {
        // Test string  a string dictionary
        let dict_type =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let string_type = DataType::Utf8;

        // build dictionary
        let mut dict_builder = StringDictionaryBuilder::<Int32Type>::new();

        dict_builder.append("one")?;
        dict_builder.append_null();
        dict_builder.append("three")?;
        dict_builder.append("four")?;
        let dict_array = Arc::new(dict_builder.finish()) as ArrayRef;

        let str_array = Arc::new(StringArray::from(vec![
            Some("not one"),
            Some("two"),
            None,
            Some("four"),
        ])) as ArrayRef;

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", dict_type.clone(), true),
            Field::new("b", string_type.clone(), true),
        ]));

        // Test 1: a = b
        let result = BooleanArray::from(vec![Some(false), None, None, Some(true)]);
        apply_logic_op(&schema, &dict_array, &str_array, Operator::Eq, result)?;

        // Test 2: now test the other direction
        // b = a
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", string_type, true),
            Field::new("b", dict_type, true),
        ]));
        let result = BooleanArray::from(vec![Some(false), None, None, Some(true)]);
        apply_logic_op(&schema, &str_array, &dict_array, Operator::Eq, result)?;

        Ok(())
    }

    #[test]
    fn plus_op() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![1, 2, 4, 8, 16]);

        apply_arithmetic::<Int32Type>(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Plus,
            Int32Array::from(vec![2, 4, 7, 12, 21]),
        )?;

        Ok(())
    }

    #[test]
    fn plus_op_dict() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
                true,
            ),
            Field::new(
                "b",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
                true,
            ),
        ]);

        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let keys = Int8Array::from(vec![Some(0), None, Some(1), Some(3), None]);
        let a = DictionaryArray::try_new(keys, Arc::new(a))?;

        let b = Int32Array::from(vec![1, 2, 4, 8, 16]);
        let keys = Int8Array::from(vec![0, 1, 1, 2, 1]);
        let b = DictionaryArray::try_new(keys, Arc::new(b))?;

        apply_arithmetic::<Int32Type>(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Plus,
            Int32Array::from(vec![Some(2), None, Some(4), Some(8), None]),
        )?;

        Ok(())
    }

    #[test]
    fn plus_op_dict_decimal() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Decimal128(10, 0)),
                ),
                true,
            ),
            Field::new(
                "b",
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Decimal128(10, 0)),
                ),
                true,
            ),
        ]);

        let value = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value),
                Some(value + 2),
                Some(value - 1),
                Some(value + 1),
            ],
            10,
            0,
        ));

        let keys = Int8Array::from(vec![Some(0), Some(2), None, Some(3), Some(0)]);
        let a = DictionaryArray::try_new(keys, decimal_array)?;

        let keys = Int8Array::from(vec![Some(0), None, Some(3), Some(2), Some(2)]);
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value + 1),
                Some(value + 3),
                Some(value),
                Some(value + 2),
            ],
            10,
            0,
        ));
        let b = DictionaryArray::try_new(keys, decimal_array)?;

        apply_arithmetic(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Plus,
            create_decimal_array(&[Some(247), None, None, Some(247), Some(246)], 11, 0),
        )?;

        Ok(())
    }

    #[test]
    fn plus_op_scalar() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Plus,
            ScalarValue::Int32(Some(1)),
            Arc::new(Int32Array::from(vec![2, 3, 4, 5, 6])),
        )?;

        Ok(())
    }

    #[test]
    fn plus_op_dict_scalar() -> Result<()> {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
            true,
        )]);

        let mut dict_builder = PrimitiveDictionaryBuilder::<Int8Type, Int32Type>::new();

        dict_builder.append(1)?;
        dict_builder.append_null();
        dict_builder.append(2)?;
        dict_builder.append(5)?;

        let a = dict_builder.finish();

        let expected: PrimitiveArray<Int32Type> =
            PrimitiveArray::from(vec![Some(2), None, Some(3), Some(6)]);

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Plus,
            ScalarValue::Dictionary(
                Box::new(DataType::Int8),
                Box::new(ScalarValue::Int32(Some(1))),
            ),
            Arc::new(expected),
        )?;

        Ok(())
    }

    #[test]
    fn plus_op_dict_scalar_decimal() -> Result<()> {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(DataType::Decimal128(10, 0)),
            ),
            true,
        )]);

        let value = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[Some(value), None, Some(value - 1), Some(value + 1)],
            10,
            0,
        ));

        let keys = Int8Array::from(vec![0, 2, 1, 3, 0]);
        let a = DictionaryArray::try_new(keys, decimal_array)?;

        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value + 1),
                Some(value),
                None,
                Some(value + 2),
                Some(value + 1),
            ],
            11,
            0,
        ));

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Plus,
            ScalarValue::Dictionary(
                Box::new(DataType::Int8),
                Box::new(ScalarValue::Decimal128(Some(1), 10, 0)),
            ),
            decimal_array,
        )?;

        Ok(())
    }

    #[test]
    fn minus_op() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![1, 2, 4, 8, 16]));
        let b = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));

        apply_arithmetic::<Int32Type>(
            schema.clone(),
            vec![a.clone(), b.clone()],
            Operator::Minus,
            Int32Array::from(vec![0, 0, 1, 4, 11]),
        )?;

        // should handle have negative values in result (for signed)
        apply_arithmetic::<Int32Type>(
            schema,
            vec![b, a],
            Operator::Minus,
            Int32Array::from(vec![0, 0, -1, -4, -11]),
        )?;

        Ok(())
    }

    #[test]
    fn minus_op_dict() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
                true,
            ),
            Field::new(
                "b",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
                true,
            ),
        ]);

        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let keys = Int8Array::from(vec![Some(0), None, Some(1), Some(3), None]);
        let a = DictionaryArray::try_new(keys, Arc::new(a))?;

        let b = Int32Array::from(vec![1, 2, 4, 8, 16]);
        let keys = Int8Array::from(vec![0, 1, 1, 2, 1]);
        let b = DictionaryArray::try_new(keys, Arc::new(b))?;

        apply_arithmetic::<Int32Type>(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Minus,
            Int32Array::from(vec![Some(0), None, Some(0), Some(0), None]),
        )?;

        Ok(())
    }

    #[test]
    fn minus_op_dict_decimal() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Decimal128(10, 0)),
                ),
                true,
            ),
            Field::new(
                "b",
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Decimal128(10, 0)),
                ),
                true,
            ),
        ]);

        let value = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value),
                Some(value + 2),
                Some(value - 1),
                Some(value + 1),
            ],
            10,
            0,
        ));

        let keys = Int8Array::from(vec![Some(0), Some(2), None, Some(3), Some(0)]);
        let a = DictionaryArray::try_new(keys, decimal_array)?;

        let keys = Int8Array::from(vec![Some(0), None, Some(3), Some(2), Some(2)]);
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value + 1),
                Some(value + 3),
                Some(value),
                Some(value + 2),
            ],
            10,
            0,
        ));
        let b = DictionaryArray::try_new(keys, decimal_array)?;

        apply_arithmetic(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Minus,
            create_decimal_array(&[Some(-1), None, None, Some(1), Some(0)], 11, 0),
        )?;

        Ok(())
    }

    #[test]
    fn minus_op_scalar() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Minus,
            ScalarValue::Int32(Some(1)),
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4])),
        )?;

        Ok(())
    }

    #[test]
    fn minus_op_dict_scalar() -> Result<()> {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
            true,
        )]);

        let mut dict_builder = PrimitiveDictionaryBuilder::<Int8Type, Int32Type>::new();

        dict_builder.append(1)?;
        dict_builder.append_null();
        dict_builder.append(2)?;
        dict_builder.append(5)?;

        let a = dict_builder.finish();

        let expected: PrimitiveArray<Int32Type> =
            PrimitiveArray::from(vec![Some(0), None, Some(1), Some(4)]);

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Minus,
            ScalarValue::Dictionary(
                Box::new(DataType::Int8),
                Box::new(ScalarValue::Int32(Some(1))),
            ),
            Arc::new(expected),
        )?;

        Ok(())
    }

    #[test]
    fn minus_op_dict_scalar_decimal() -> Result<()> {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(DataType::Decimal128(10, 0)),
            ),
            true,
        )]);

        let value = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[Some(value), None, Some(value - 1), Some(value + 1)],
            10,
            0,
        ));

        let keys = Int8Array::from(vec![0, 2, 1, 3, 0]);
        let a = DictionaryArray::try_new(keys, decimal_array)?;

        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value - 1),
                Some(value - 2),
                None,
                Some(value),
                Some(value - 1),
            ],
            11,
            0,
        ));

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Minus,
            ScalarValue::Dictionary(
                Box::new(DataType::Int8),
                Box::new(ScalarValue::Decimal128(Some(1), 10, 0)),
            ),
            decimal_array,
        )?;

        Ok(())
    }

    #[test]
    fn multiply_op() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![4, 8, 16, 32, 64]));
        let b = Arc::new(Int32Array::from(vec![2, 4, 8, 16, 32]));

        apply_arithmetic::<Int32Type>(
            schema,
            vec![a, b],
            Operator::Multiply,
            Int32Array::from(vec![8, 32, 128, 512, 2048]),
        )?;

        Ok(())
    }

    #[test]
    fn multiply_op_dict() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
                true,
            ),
            Field::new(
                "b",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
                true,
            ),
        ]);

        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let keys = Int8Array::from(vec![Some(0), None, Some(1), Some(3), None]);
        let a = DictionaryArray::try_new(keys, Arc::new(a))?;

        let b = Int32Array::from(vec![1, 2, 4, 8, 16]);
        let keys = Int8Array::from(vec![0, 1, 1, 2, 1]);
        let b = DictionaryArray::try_new(keys, Arc::new(b))?;

        apply_arithmetic::<Int32Type>(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Multiply,
            Int32Array::from(vec![Some(1), None, Some(4), Some(16), None]),
        )?;

        Ok(())
    }

    #[test]
    fn multiply_op_dict_decimal() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Decimal128(10, 0)),
                ),
                true,
            ),
            Field::new(
                "b",
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Decimal128(10, 0)),
                ),
                true,
            ),
        ]);

        let value = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value),
                Some(value + 2),
                Some(value - 1),
                Some(value + 1),
            ],
            10,
            0,
        )) as ArrayRef;

        let keys = Int8Array::from(vec![Some(0), Some(2), None, Some(3), Some(0)]);
        let a = DictionaryArray::try_new(keys, decimal_array)?;

        let keys = Int8Array::from(vec![Some(0), None, Some(3), Some(2), Some(2)]);
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value + 1),
                Some(value + 3),
                Some(value),
                Some(value + 2),
            ],
            10,
            0,
        ));
        let b = DictionaryArray::try_new(keys, decimal_array)?;

        apply_arithmetic(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Multiply,
            create_decimal_array(
                &[Some(15252), None, None, Some(15252), Some(15129)],
                21,
                0,
            ),
        )?;

        Ok(())
    }

    #[test]
    fn multiply_op_scalar() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Multiply,
            ScalarValue::Int32(Some(2)),
            Arc::new(Int32Array::from(vec![2, 4, 6, 8, 10])),
        )?;

        Ok(())
    }

    #[test]
    fn multiply_op_dict_scalar() -> Result<()> {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
            true,
        )]);

        let mut dict_builder = PrimitiveDictionaryBuilder::<Int8Type, Int32Type>::new();

        dict_builder.append(1)?;
        dict_builder.append_null();
        dict_builder.append(2)?;
        dict_builder.append(5)?;

        let a = dict_builder.finish();

        let expected: PrimitiveArray<Int32Type> =
            PrimitiveArray::from(vec![Some(2), None, Some(4), Some(10)]);

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Multiply,
            ScalarValue::Dictionary(
                Box::new(DataType::Int8),
                Box::new(ScalarValue::Int32(Some(2))),
            ),
            Arc::new(expected),
        )?;

        Ok(())
    }

    #[test]
    fn multiply_op_dict_scalar_decimal() -> Result<()> {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(DataType::Decimal128(10, 0)),
            ),
            true,
        )]);

        let value = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[Some(value), None, Some(value - 1), Some(value + 1)],
            10,
            0,
        ));

        let keys = Int8Array::from(vec![0, 2, 1, 3, 0]);
        let a = DictionaryArray::try_new(keys, decimal_array)?;

        let decimal_array = Arc::new(create_decimal_array(
            &[Some(246), Some(244), None, Some(248), Some(246)],
            21,
            0,
        ));

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Multiply,
            ScalarValue::Dictionary(
                Box::new(DataType::Int8),
                Box::new(ScalarValue::Decimal128(Some(2), 10, 0)),
            ),
            decimal_array,
        )?;

        Ok(())
    }

    #[test]
    fn divide_op() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![8, 32, 128, 512, 2048]));
        let b = Arc::new(Int32Array::from(vec![2, 4, 8, 16, 32]));

        apply_arithmetic::<Int32Type>(
            schema,
            vec![a, b],
            Operator::Divide,
            Int32Array::from(vec![4, 8, 16, 32, 64]),
        )?;

        Ok(())
    }

    #[test]
    fn divide_op_dict() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
                true,
            ),
            Field::new(
                "b",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
                true,
            ),
        ]);

        let mut dict_builder = PrimitiveDictionaryBuilder::<Int8Type, Int32Type>::new();

        dict_builder.append(1)?;
        dict_builder.append_null();
        dict_builder.append(2)?;
        dict_builder.append(5)?;
        dict_builder.append(0)?;

        let a = dict_builder.finish();

        let b = Int32Array::from(vec![1, 2, 4, 8, 16]);
        let keys = Int8Array::from(vec![0, 1, 1, 2, 1]);
        let b = DictionaryArray::try_new(keys, Arc::new(b))?;

        apply_arithmetic::<Int32Type>(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Divide,
            Int32Array::from(vec![Some(1), None, Some(1), Some(1), Some(0)]),
        )?;

        Ok(())
    }

    #[test]
    fn divide_op_dict_decimal() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Decimal128(10, 0)),
                ),
                true,
            ),
            Field::new(
                "b",
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Decimal128(10, 0)),
                ),
                true,
            ),
        ]);

        let value = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value),
                Some(value + 2),
                Some(value - 1),
                Some(value + 1),
            ],
            10,
            0,
        ));

        let keys = Int8Array::from(vec![Some(0), Some(2), None, Some(3), Some(0)]);
        let a = DictionaryArray::try_new(keys, decimal_array)?;

        let keys = Int8Array::from(vec![Some(0), None, Some(3), Some(2), Some(2)]);
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value + 1),
                Some(value + 3),
                Some(value),
                Some(value + 2),
            ],
            10,
            0,
        ));
        let b = DictionaryArray::try_new(keys, decimal_array)?;

        apply_arithmetic(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Divide,
            create_decimal_array(
                &[
                    Some(9919), // 0.9919
                    None,
                    None,
                    Some(10081), // 1.0081
                    Some(10000), // 1.0
                ],
                14,
                4,
            ),
        )?;

        Ok(())
    }

    #[test]
    fn divide_op_scalar() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Divide,
            ScalarValue::Int32(Some(2)),
            Arc::new(Int32Array::from(vec![0, 1, 1, 2, 2])),
        )?;

        Ok(())
    }

    #[test]
    fn divide_op_dict_scalar() -> Result<()> {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
            true,
        )]);

        let mut dict_builder = PrimitiveDictionaryBuilder::<Int8Type, Int32Type>::new();

        dict_builder.append(1)?;
        dict_builder.append_null();
        dict_builder.append(2)?;
        dict_builder.append(5)?;

        let a = dict_builder.finish();

        let expected: PrimitiveArray<Int32Type> =
            PrimitiveArray::from(vec![Some(0), None, Some(1), Some(2)]);

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Divide,
            ScalarValue::Dictionary(
                Box::new(DataType::Int8),
                Box::new(ScalarValue::Int32(Some(2))),
            ),
            Arc::new(expected),
        )?;

        Ok(())
    }

    #[test]
    fn divide_op_dict_scalar_decimal() -> Result<()> {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(DataType::Decimal128(10, 0)),
            ),
            true,
        )]);

        let value = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[Some(value), None, Some(value - 1), Some(value + 1)],
            10,
            0,
        ));

        let keys = Int8Array::from(vec![0, 2, 1, 3, 0]);
        let a = DictionaryArray::try_new(keys, decimal_array)?;

        let decimal_array = Arc::new(create_decimal_array(
            &[Some(615000), Some(610000), None, Some(620000), Some(615000)],
            14,
            4,
        ));

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Divide,
            ScalarValue::Dictionary(
                Box::new(DataType::Int8),
                Box::new(ScalarValue::Decimal128(Some(2), 10, 0)),
            ),
            decimal_array,
        )?;

        Ok(())
    }

    #[test]
    fn modulus_op() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![8, 32, 128, 512, 2048]));
        let b = Arc::new(Int32Array::from(vec![2, 4, 7, 14, 32]));

        apply_arithmetic::<Int32Type>(
            schema,
            vec![a, b],
            Operator::Modulo,
            Int32Array::from(vec![0, 0, 2, 8, 0]),
        )?;

        Ok(())
    }

    #[test]
    fn modulus_op_dict() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
                true,
            ),
            Field::new(
                "b",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
                true,
            ),
        ]);

        let mut dict_builder = PrimitiveDictionaryBuilder::<Int8Type, Int32Type>::new();

        dict_builder.append(1)?;
        dict_builder.append_null();
        dict_builder.append(2)?;
        dict_builder.append(5)?;
        dict_builder.append(0)?;

        let a = dict_builder.finish();

        let b = Int32Array::from(vec![1, 2, 4, 8, 16]);
        let keys = Int8Array::from(vec![0, 1, 1, 2, 1]);
        let b = DictionaryArray::try_new(keys, Arc::new(b))?;

        apply_arithmetic::<Int32Type>(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Modulo,
            Int32Array::from(vec![Some(0), None, Some(0), Some(1), Some(0)]),
        )?;

        Ok(())
    }

    #[test]
    fn modulus_op_dict_decimal() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Decimal128(10, 0)),
                ),
                true,
            ),
            Field::new(
                "b",
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Decimal128(10, 0)),
                ),
                true,
            ),
        ]);

        let value = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value),
                Some(value + 2),
                Some(value - 1),
                Some(value + 1),
            ],
            10,
            0,
        ));

        let keys = Int8Array::from(vec![Some(0), Some(2), None, Some(3), Some(0)]);
        let a = DictionaryArray::try_new(keys, decimal_array)?;

        let keys = Int8Array::from(vec![Some(0), None, Some(3), Some(2), Some(2)]);
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value + 1),
                Some(value + 3),
                Some(value),
                Some(value + 2),
            ],
            10,
            0,
        ));
        let b = DictionaryArray::try_new(keys, decimal_array)?;

        apply_arithmetic(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Modulo,
            create_decimal_array(&[Some(123), None, None, Some(1), Some(0)], 10, 0),
        )?;

        Ok(())
    }

    #[test]
    fn modulus_op_scalar() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Modulo,
            ScalarValue::Int32(Some(2)),
            Arc::new(Int32Array::from(vec![1, 0, 1, 0, 1])),
        )?;

        Ok(())
    }

    #[test]
    fn modules_op_dict_scalar() -> Result<()> {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
            true,
        )]);

        let mut dict_builder = PrimitiveDictionaryBuilder::<Int8Type, Int32Type>::new();

        dict_builder.append(1)?;
        dict_builder.append_null();
        dict_builder.append(2)?;
        dict_builder.append(5)?;

        let a = dict_builder.finish();

        let expected: PrimitiveArray<Int32Type> =
            PrimitiveArray::from(vec![Some(1), None, Some(0), Some(1)]);

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Modulo,
            ScalarValue::Dictionary(
                Box::new(DataType::Int8),
                Box::new(ScalarValue::Int32(Some(2))),
            ),
            Arc::new(expected),
        )?;

        Ok(())
    }

    #[test]
    fn modulus_op_dict_scalar_decimal() -> Result<()> {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(DataType::Decimal128(10, 0)),
            ),
            true,
        )]);

        let value = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[Some(value), None, Some(value - 1), Some(value + 1)],
            10,
            0,
        ));

        let keys = Int8Array::from(vec![0, 2, 1, 3, 0]);
        let a = DictionaryArray::try_new(keys, decimal_array)?;

        let decimal_array = Arc::new(create_decimal_array(
            &[Some(1), Some(0), None, Some(0), Some(1)],
            10,
            0,
        ));

        apply_arithmetic_scalar(
            Arc::new(schema),
            vec![Arc::new(a)],
            Operator::Modulo,
            ScalarValue::Dictionary(
                Box::new(DataType::Int8),
                Box::new(ScalarValue::Decimal128(Some(2), 10, 0)),
            ),
            decimal_array,
        )?;

        Ok(())
    }

    fn apply_arithmetic<T: ArrowNumericType>(
        schema: SchemaRef,
        data: Vec<ArrayRef>,
        op: Operator,
        expected: PrimitiveArray<T>,
    ) -> Result<()> {
        let arithmetic_op =
            binary_op(col("a", &schema)?, op, col("b", &schema)?, &schema)?;
        let batch = RecordBatch::try_new(schema, data)?;
        let result = arithmetic_op
            .evaluate(&batch)?
            .into_array(batch.num_rows())
            .expect("Failed to convert to array");

        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    fn apply_arithmetic_scalar(
        schema: SchemaRef,
        data: Vec<ArrayRef>,
        op: Operator,
        literal: ScalarValue,
        expected: ArrayRef,
    ) -> Result<()> {
        let lit = Arc::new(Literal::new(literal));
        let arithmetic_op = binary_op(col("a", &schema)?, op, lit, &schema)?;
        let batch = RecordBatch::try_new(schema, data)?;
        let result = arithmetic_op
            .evaluate(&batch)?
            .into_array(batch.num_rows())
            .expect("Failed to convert to array");

        assert_eq!(&result, &expected);
        Ok(())
    }

    fn apply_logic_op(
        schema: &SchemaRef,
        left: &ArrayRef,
        right: &ArrayRef,
        op: Operator,
        expected: BooleanArray,
    ) -> Result<()> {
        let op = binary_op(col("a", schema)?, op, col("b", schema)?, schema)?;
        let data: Vec<ArrayRef> = vec![left.clone(), right.clone()];
        let batch = RecordBatch::try_new(schema.clone(), data)?;
        let result = op
            .evaluate(&batch)?
            .into_array(batch.num_rows())
            .expect("Failed to convert to array");

        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    // Test `scalar <op> arr` produces expected
    fn apply_logic_op_scalar_arr(
        schema: &SchemaRef,
        scalar: &ScalarValue,
        arr: &ArrayRef,
        op: Operator,
        expected: &BooleanArray,
    ) -> Result<()> {
        let scalar = lit(scalar.clone());
        let op = binary_op(scalar, op, col("a", schema)?, schema)?;
        let batch = RecordBatch::try_new(Arc::clone(schema), vec![Arc::clone(arr)])?;
        let result = op
            .evaluate(&batch)?
            .into_array(batch.num_rows())
            .expect("Failed to convert to array");
        assert_eq!(result.as_ref(), expected);

        Ok(())
    }

    // Test `arr <op> scalar` produces expected
    fn apply_logic_op_arr_scalar(
        schema: &SchemaRef,
        arr: &ArrayRef,
        scalar: &ScalarValue,
        op: Operator,
        expected: &BooleanArray,
    ) -> Result<()> {
        let scalar = lit(scalar.clone());
        let op = binary_op(col("a", schema)?, op, scalar, schema)?;
        let batch = RecordBatch::try_new(Arc::clone(schema), vec![Arc::clone(arr)])?;
        let result = op
            .evaluate(&batch)?
            .into_array(batch.num_rows())
            .expect("Failed to convert to array");
        assert_eq!(result.as_ref(), expected);

        Ok(())
    }

    #[test]
    fn and_with_nulls_op() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Boolean, true),
            Field::new("b", DataType::Boolean, true),
        ]);
        let a = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
        ])) as ArrayRef;
        let b = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            None,
            None,
            None,
        ])) as ArrayRef;

        let expected = BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(false),
            None,
            Some(false),
            None,
        ]);
        apply_logic_op(&Arc::new(schema), &a, &b, Operator::And, expected)?;

        Ok(())
    }

    #[test]
    fn or_with_nulls_op() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Boolean, true),
            Field::new("b", DataType::Boolean, true),
        ]);
        let a = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
        ])) as ArrayRef;
        let b = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            None,
            None,
            None,
        ])) as ArrayRef;

        let expected = BooleanArray::from(vec![
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            None,
            Some(true),
            None,
            None,
        ]);
        apply_logic_op(&Arc::new(schema), &a, &b, Operator::Or, expected)?;

        Ok(())
    }

    /// Returns (schema, a: BooleanArray, b: BooleanArray) with all possible inputs
    ///
    /// a: [true, true, true,  NULL, NULL, NULL,  false, false, false]
    /// b: [true, NULL, false, true, NULL, false, true,  NULL,  false]
    fn bool_test_arrays() -> (SchemaRef, ArrayRef, ArrayRef) {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Boolean, true),
            Field::new("b", DataType::Boolean, true),
        ]);
        let a: BooleanArray = [
            Some(true),
            Some(true),
            Some(true),
            None,
            None,
            None,
            Some(false),
            Some(false),
            Some(false),
        ]
        .iter()
        .collect();
        let b: BooleanArray = [
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            Some(false),
        ]
        .iter()
        .collect();
        (Arc::new(schema), Arc::new(a), Arc::new(b))
    }

    /// Returns (schema, BooleanArray) with [true, NULL, false]
    fn scalar_bool_test_array() -> (SchemaRef, ArrayRef) {
        let schema = Schema::new(vec![Field::new("a", DataType::Boolean, true)]);
        let a: BooleanArray = [Some(true), None, Some(false)].iter().collect();
        (Arc::new(schema), Arc::new(a))
    }

    #[test]
    fn eq_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(true),
            None,
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            Some(true),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::Eq, expected).unwrap();
    }

    #[test]
    fn eq_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::Eq,
            &expected,
        )
        .unwrap();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::Eq,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::Eq,
            &expected,
        )
        .unwrap();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::Eq,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn neq_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(false),
            None,
            Some(true),
            None,
            None,
            None,
            Some(true),
            None,
            Some(false),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::NotEq, expected).unwrap();
    }

    #[test]
    fn neq_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::NotEq,
            &expected,
        )
        .unwrap();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::NotEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::NotEq,
            &expected,
        )
        .unwrap();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::NotEq,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn lt_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(false),
            None,
            Some(false),
            None,
            None,
            None,
            Some(true),
            None,
            Some(false),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::Lt, expected).unwrap();
    }

    #[test]
    fn lt_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(false), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::Lt,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::Lt,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::Lt,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(false)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::Lt,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn lt_eq_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(true),
            None,
            Some(false),
            None,
            None,
            None,
            Some(true),
            None,
            Some(true),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::LtEq, expected).unwrap();
    }

    #[test]
    fn lt_eq_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::LtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(true)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::LtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::LtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::LtEq,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn gt_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(false),
            None,
            Some(true),
            None,
            None,
            None,
            Some(false),
            None,
            Some(false),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::Gt, expected).unwrap();
    }

    #[test]
    fn gt_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::Gt,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(false)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::Gt,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::Gt,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::Gt,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn gt_eq_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(true),
            None,
            Some(true),
            None,
            None,
            None,
            Some(false),
            None,
            Some(true),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::GtEq, expected).unwrap();
    }

    #[test]
    fn gt_eq_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(true), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::GtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::GtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::GtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(true)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::GtEq,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn is_distinct_from_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(false),
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(true),
            Some(true),
            Some(true),
            Some(false),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::IsDistinctFrom, expected).unwrap();
    }

    #[test]
    fn is_not_distinct_from_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            Some(true),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::IsNotDistinctFrom, expected).unwrap();
    }

    #[test]
    fn relatively_deeply_nested() {
        // Reproducer for https://github.com/apache/arrow-datafusion/issues/419

        // where even relatively shallow binary expressions overflowed
        // the stack in debug builds

        let input: Vec<_> = vec![1, 2, 3, 4, 5].into_iter().map(Some).collect();
        let a: Int32Array = input.iter().collect();

        let batch = RecordBatch::try_from_iter(vec![("a", Arc::new(a) as _)]).unwrap();
        let schema = batch.schema();

        // build a left deep tree ((((a + a) + a) + a ....
        let tree_depth: i32 = 100;
        let expr = (0..tree_depth)
            .map(|_| col("a", schema.as_ref()).unwrap())
            .reduce(|l, r| binary(l, Operator::Plus, r, &schema).unwrap())
            .unwrap();

        let result = expr
            .evaluate(&batch)
            .expect("evaluation")
            .into_array(batch.num_rows())
            .expect("Failed to convert to array");

        let expected: Int32Array = input
            .into_iter()
            .map(|i| i.map(|i| i * tree_depth))
            .collect();
        assert_eq!(result.as_ref(), &expected);
    }

    fn create_decimal_array(
        array: &[Option<i128>],
        precision: u8,
        scale: i8,
    ) -> Decimal128Array {
        let mut decimal_builder = Decimal128Builder::with_capacity(array.len());
        for value in array.iter().copied() {
            decimal_builder.append_option(value)
        }
        decimal_builder
            .finish()
            .with_precision_and_scale(precision, scale)
            .unwrap()
    }

    #[test]
    fn comparison_dict_decimal_scalar_expr_test() -> Result<()> {
        // scalar of decimal compare with dictionary decimal array
        let value_i128 = 123;
        let decimal_scalar = ScalarValue::Dictionary(
            Box::new(DataType::Int8),
            Box::new(ScalarValue::Decimal128(Some(value_i128), 25, 3)),
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(DataType::Decimal128(25, 3)),
            ),
            true,
        )]));
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value_i128),
                None,
                Some(value_i128 - 1),
                Some(value_i128 + 1),
            ],
            25,
            3,
        ));

        let keys = Int8Array::from(vec![Some(0), None, Some(2), Some(3)]);
        let dictionary =
            Arc::new(DictionaryArray::try_new(keys, decimal_array)?) as ArrayRef;

        // array = scalar
        apply_logic_op_arr_scalar(
            &schema,
            &dictionary,
            &decimal_scalar,
            Operator::Eq,
            &BooleanArray::from(vec![Some(true), None, Some(false), Some(false)]),
        )
        .unwrap();
        // array != scalar
        apply_logic_op_arr_scalar(
            &schema,
            &dictionary,
            &decimal_scalar,
            Operator::NotEq,
            &BooleanArray::from(vec![Some(false), None, Some(true), Some(true)]),
        )
        .unwrap();
        //  array < scalar
        apply_logic_op_arr_scalar(
            &schema,
            &dictionary,
            &decimal_scalar,
            Operator::Lt,
            &BooleanArray::from(vec![Some(false), None, Some(true), Some(false)]),
        )
        .unwrap();

        //  array <= scalar
        apply_logic_op_arr_scalar(
            &schema,
            &dictionary,
            &decimal_scalar,
            Operator::LtEq,
            &BooleanArray::from(vec![Some(true), None, Some(true), Some(false)]),
        )
        .unwrap();
        // array > scalar
        apply_logic_op_arr_scalar(
            &schema,
            &dictionary,
            &decimal_scalar,
            Operator::Gt,
            &BooleanArray::from(vec![Some(false), None, Some(false), Some(true)]),
        )
        .unwrap();

        // array >= scalar
        apply_logic_op_arr_scalar(
            &schema,
            &dictionary,
            &decimal_scalar,
            Operator::GtEq,
            &BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
        )
        .unwrap();

        Ok(())
    }

    #[test]
    fn comparison_decimal_expr_test() -> Result<()> {
        // scalar of decimal compare with decimal array
        let value_i128 = 123;
        let decimal_scalar = ScalarValue::Decimal128(Some(value_i128), 25, 3);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Decimal128(25, 3),
            true,
        )]));
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value_i128),
                None,
                Some(value_i128 - 1),
                Some(value_i128 + 1),
            ],
            25,
            3,
        )) as ArrayRef;
        // array = scalar
        apply_logic_op_arr_scalar(
            &schema,
            &decimal_array,
            &decimal_scalar,
            Operator::Eq,
            &BooleanArray::from(vec![Some(true), None, Some(false), Some(false)]),
        )
        .unwrap();
        // array != scalar
        apply_logic_op_arr_scalar(
            &schema,
            &decimal_array,
            &decimal_scalar,
            Operator::NotEq,
            &BooleanArray::from(vec![Some(false), None, Some(true), Some(true)]),
        )
        .unwrap();
        //  array < scalar
        apply_logic_op_arr_scalar(
            &schema,
            &decimal_array,
            &decimal_scalar,
            Operator::Lt,
            &BooleanArray::from(vec![Some(false), None, Some(true), Some(false)]),
        )
        .unwrap();

        //  array <= scalar
        apply_logic_op_arr_scalar(
            &schema,
            &decimal_array,
            &decimal_scalar,
            Operator::LtEq,
            &BooleanArray::from(vec![Some(true), None, Some(true), Some(false)]),
        )
        .unwrap();
        // array > scalar
        apply_logic_op_arr_scalar(
            &schema,
            &decimal_array,
            &decimal_scalar,
            Operator::Gt,
            &BooleanArray::from(vec![Some(false), None, Some(false), Some(true)]),
        )
        .unwrap();

        // array >= scalar
        apply_logic_op_arr_scalar(
            &schema,
            &decimal_array,
            &decimal_scalar,
            Operator::GtEq,
            &BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
        )
        .unwrap();

        // scalar of different data type with decimal array
        let decimal_scalar = ScalarValue::Decimal128(Some(123_456), 10, 3);
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        // scalar == array
        apply_logic_op_scalar_arr(
            &schema,
            &decimal_scalar,
            &(Arc::new(Int64Array::from(vec![Some(124), None])) as ArrayRef),
            Operator::Eq,
            &BooleanArray::from(vec![Some(false), None]),
        )
        .unwrap();

        // array != scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Int64Array::from(vec![Some(123), None, Some(1)])) as ArrayRef),
            &decimal_scalar,
            Operator::NotEq,
            &BooleanArray::from(vec![Some(true), None, Some(true)]),
        )
        .unwrap();

        // array < scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Int64Array::from(vec![Some(123), None, Some(124)])) as ArrayRef),
            &decimal_scalar,
            Operator::Lt,
            &BooleanArray::from(vec![Some(true), None, Some(false)]),
        )
        .unwrap();

        // array > scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Int64Array::from(vec![Some(123), None, Some(124)])) as ArrayRef),
            &decimal_scalar,
            Operator::Gt,
            &BooleanArray::from(vec![Some(false), None, Some(true)]),
        )
        .unwrap();

        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float64, true)]));
        // array == scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Float64Array::from(vec![Some(123.456), None, Some(123.457)]))
                as ArrayRef),
            &decimal_scalar,
            Operator::Eq,
            &BooleanArray::from(vec![Some(true), None, Some(false)]),
        )
        .unwrap();

        // array <= scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Float64Array::from(vec![
                Some(123.456),
                None,
                Some(123.457),
                Some(123.45),
            ])) as ArrayRef),
            &decimal_scalar,
            Operator::LtEq,
            &BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
        )
        .unwrap();
        // array >= scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Float64Array::from(vec![
                Some(123.456),
                None,
                Some(123.457),
                Some(123.45),
            ])) as ArrayRef),
            &decimal_scalar,
            Operator::GtEq,
            &BooleanArray::from(vec![Some(true), None, Some(true), Some(false)]),
        )
        .unwrap();

        let value: i128 = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[Some(value), None, Some(value - 1), Some(value + 1)],
            10,
            0,
        )) as ArrayRef;

        // comparison array op for decimal array
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Decimal128(10, 0), true),
            Field::new("b", DataType::Decimal128(10, 0), true),
        ]));
        let right_decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value - 1),
                Some(value),
                Some(value + 1),
                Some(value + 1),
            ],
            10,
            0,
        )) as ArrayRef;

        apply_logic_op(
            &schema,
            &decimal_array,
            &right_decimal_array,
            Operator::Eq,
            BooleanArray::from(vec![Some(false), None, Some(false), Some(true)]),
        )
        .unwrap();

        apply_logic_op(
            &schema,
            &decimal_array,
            &right_decimal_array,
            Operator::NotEq,
            BooleanArray::from(vec![Some(true), None, Some(true), Some(false)]),
        )
        .unwrap();

        apply_logic_op(
            &schema,
            &decimal_array,
            &right_decimal_array,
            Operator::Lt,
            BooleanArray::from(vec![Some(false), None, Some(true), Some(false)]),
        )
        .unwrap();

        apply_logic_op(
            &schema,
            &decimal_array,
            &right_decimal_array,
            Operator::LtEq,
            BooleanArray::from(vec![Some(false), None, Some(true), Some(true)]),
        )
        .unwrap();

        apply_logic_op(
            &schema,
            &decimal_array,
            &right_decimal_array,
            Operator::Gt,
            BooleanArray::from(vec![Some(true), None, Some(false), Some(false)]),
        )
        .unwrap();

        apply_logic_op(
            &schema,
            &decimal_array,
            &right_decimal_array,
            Operator::GtEq,
            BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
        )
        .unwrap();

        // compare decimal array with other array type
        let value: i64 = 123;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Decimal128(10, 0), true),
        ]));

        let int64_array = Arc::new(Int64Array::from(vec![
            Some(value),
            Some(value - 1),
            Some(value),
            Some(value + 1),
        ])) as ArrayRef;

        // eq: int64array == decimal array
        apply_logic_op(
            &schema,
            &int64_array,
            &decimal_array,
            Operator::Eq,
            BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
        )
        .unwrap();
        // neq: int64array != decimal array
        apply_logic_op(
            &schema,
            &int64_array,
            &decimal_array,
            Operator::NotEq,
            BooleanArray::from(vec![Some(false), None, Some(true), Some(false)]),
        )
        .unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));

        let value: i128 = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value), // 1.23
                None,
                Some(value - 1), // 1.22
                Some(value + 1), // 1.24
            ],
            10,
            2,
        )) as ArrayRef;
        let float64_array = Arc::new(Float64Array::from(vec![
            Some(1.23),
            Some(1.22),
            Some(1.23),
            Some(1.24),
        ])) as ArrayRef;
        // lt: float64array < decimal array
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::Lt,
            BooleanArray::from(vec![Some(false), None, Some(false), Some(false)]),
        )
        .unwrap();
        // lt_eq: float64array <= decimal array
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::LtEq,
            BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
        )
        .unwrap();
        // gt: float64array > decimal array
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::Gt,
            BooleanArray::from(vec![Some(false), None, Some(true), Some(false)]),
        )
        .unwrap();
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::GtEq,
            BooleanArray::from(vec![Some(true), None, Some(true), Some(true)]),
        )
        .unwrap();
        // is distinct: float64array is distinct decimal array
        // TODO: now we do not refactor the `is distinct or is not distinct` rule of coercion.
        // traced by https://github.com/apache/arrow-datafusion/issues/1590
        // the decimal array will be casted to float64array
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::IsDistinctFrom,
            BooleanArray::from(vec![Some(false), Some(true), Some(true), Some(false)]),
        )
        .unwrap();
        // is not distinct
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::IsNotDistinctFrom,
            BooleanArray::from(vec![Some(true), Some(false), Some(false), Some(true)]),
        )
        .unwrap();

        Ok(())
    }

    fn apply_decimal_arithmetic_op(
        schema: &SchemaRef,
        left: &ArrayRef,
        right: &ArrayRef,
        op: Operator,
        expected: ArrayRef,
    ) -> Result<()> {
        let arithmetic_op = binary_op(col("a", schema)?, op, col("b", schema)?, schema)?;
        let data: Vec<ArrayRef> = vec![left.clone(), right.clone()];
        let batch = RecordBatch::try_new(schema.clone(), data)?;
        let result = arithmetic_op
            .evaluate(&batch)?
            .into_array(batch.num_rows())
            .expect("Failed to convert to array");

        assert_eq!(result.as_ref(), expected.as_ref());
        Ok(())
    }

    #[test]
    fn arithmetic_decimal_expr_test() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));
        let value: i128 = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value), // 1.23
                None,
                Some(value - 1), // 1.22
                Some(value + 1), // 1.24
            ],
            10,
            2,
        )) as ArrayRef;
        let int32_array = Arc::new(Int32Array::from(vec![
            Some(123),
            Some(122),
            Some(123),
            Some(124),
        ])) as ArrayRef;

        // add: Int32array add decimal array
        let expect = Arc::new(create_decimal_array(
            &[Some(12423), None, Some(12422), Some(12524)],
            13,
            2,
        )) as ArrayRef;
        apply_decimal_arithmetic_op(
            &schema,
            &int32_array,
            &decimal_array,
            Operator::Plus,
            expect,
        )
        .unwrap();

        // subtract: decimal array subtract int32 array
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Decimal128(10, 2), true),
            Field::new("b", DataType::Int32, true),
        ]));
        let expect = Arc::new(create_decimal_array(
            &[Some(-12177), None, Some(-12178), Some(-12276)],
            13,
            2,
        )) as ArrayRef;
        apply_decimal_arithmetic_op(
            &schema,
            &decimal_array,
            &int32_array,
            Operator::Minus,
            expect,
        )
        .unwrap();

        // multiply: decimal array multiply int32 array
        let expect = Arc::new(create_decimal_array(
            &[Some(15129), None, Some(15006), Some(15376)],
            21,
            2,
        )) as ArrayRef;
        apply_decimal_arithmetic_op(
            &schema,
            &decimal_array,
            &int32_array,
            Operator::Multiply,
            expect,
        )
        .unwrap();

        // divide: int32 array divide decimal array
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));
        let expect = Arc::new(create_decimal_array(
            &[Some(1000000), None, Some(1008196), Some(1000000)],
            16,
            4,
        )) as ArrayRef;
        apply_decimal_arithmetic_op(
            &schema,
            &int32_array,
            &decimal_array,
            Operator::Divide,
            expect,
        )
        .unwrap();

        // modulus: int32 array modulus decimal array
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));
        let expect = Arc::new(create_decimal_array(
            &[Some(000), None, Some(100), Some(000)],
            10,
            2,
        )) as ArrayRef;
        apply_decimal_arithmetic_op(
            &schema,
            &int32_array,
            &decimal_array,
            Operator::Modulo,
            expect,
        )
        .unwrap();

        Ok(())
    }

    #[test]
    fn arithmetic_decimal_float_expr_test() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));
        let value: i128 = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value), // 1.23
                None,
                Some(value - 1), // 1.22
                Some(value + 1), // 1.24
            ],
            10,
            2,
        )) as ArrayRef;
        let float64_array = Arc::new(Float64Array::from(vec![
            Some(123.0),
            Some(122.0),
            Some(123.0),
            Some(124.0),
        ])) as ArrayRef;

        // add: float64 array add decimal array
        let expect = Arc::new(Float64Array::from(vec![
            Some(124.23),
            None,
            Some(124.22),
            Some(125.24),
        ])) as ArrayRef;
        apply_decimal_arithmetic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::Plus,
            expect,
        )
        .unwrap();

        // subtract: decimal array subtract float64 array
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));
        let expect = Arc::new(Float64Array::from(vec![
            Some(121.77),
            None,
            Some(121.78),
            Some(122.76),
        ])) as ArrayRef;
        apply_decimal_arithmetic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::Minus,
            expect,
        )
        .unwrap();

        // multiply: decimal array multiply float64 array
        let expect = Arc::new(Float64Array::from(vec![
            Some(151.29),
            None,
            Some(150.06),
            Some(153.76),
        ])) as ArrayRef;
        apply_decimal_arithmetic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::Multiply,
            expect,
        )
        .unwrap();

        // divide: float64 array divide decimal array
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));
        let expect = Arc::new(Float64Array::from(vec![
            Some(100.0),
            None,
            Some(100.81967213114754),
            Some(100.0),
        ])) as ArrayRef;
        apply_decimal_arithmetic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::Divide,
            expect,
        )
        .unwrap();

        // modulus: float64 array modulus decimal array
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));
        let expect = Arc::new(Float64Array::from(vec![
            Some(1.7763568394002505e-15),
            None,
            Some(1.0000000000000027),
            Some(8.881784197001252e-16),
        ])) as ArrayRef;
        apply_decimal_arithmetic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::Modulo,
            expect,
        )
        .unwrap();

        Ok(())
    }

    #[test]
    fn arithmetic_divide_zero() -> Result<()> {
        // other data type
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]));
        let a = Arc::new(Int32Array::from(vec![100]));
        let b = Arc::new(Int32Array::from(vec![0]));

        let err = apply_arithmetic::<Int32Type>(
            schema,
            vec![a, b],
            Operator::Divide,
            Int32Array::from(vec![Some(4), Some(8), Some(16), Some(32), Some(64)]),
        )
        .unwrap_err();

        assert!(
            matches!(err, DataFusionError::ArrowError(ArrowError::DivideByZero)),
            "{err}"
        );

        // decimal
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Decimal128(25, 3), true),
            Field::new("b", DataType::Decimal128(25, 3), true),
        ]));
        let left_decimal_array = Arc::new(create_decimal_array(&[Some(1234567)], 25, 3));
        let right_decimal_array = Arc::new(create_decimal_array(&[Some(0)], 25, 3));

        let err = apply_arithmetic::<Decimal128Type>(
            schema,
            vec![left_decimal_array, right_decimal_array],
            Operator::Divide,
            create_decimal_array(
                &[Some(12345670000000000000000000000000000), None],
                38,
                29,
            ),
        )
        .unwrap_err();

        assert!(
            matches!(err, DataFusionError::ArrowError(ArrowError::DivideByZero)),
            "{err}"
        );

        Ok(())
    }

    #[test]
    fn bitwise_array_test() -> Result<()> {
        let left = Arc::new(Int32Array::from(vec![Some(12), None, Some(11)])) as ArrayRef;
        let right =
            Arc::new(Int32Array::from(vec![Some(1), Some(3), Some(7)])) as ArrayRef;
        let mut result = bitwise_and_dyn(left.clone(), right.clone())?;
        let expected = Int32Array::from(vec![Some(0), None, Some(3)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_or_dyn(left.clone(), right.clone())?;
        let expected = Int32Array::from(vec![Some(13), None, Some(15)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_xor_dyn(left.clone(), right.clone())?;
        let expected = Int32Array::from(vec![Some(13), None, Some(12)]);
        assert_eq!(result.as_ref(), &expected);

        let left =
            Arc::new(UInt32Array::from(vec![Some(12), None, Some(11)])) as ArrayRef;
        let right =
            Arc::new(UInt32Array::from(vec![Some(1), Some(3), Some(7)])) as ArrayRef;
        let mut result = bitwise_and_dyn(left.clone(), right.clone())?;
        let expected = UInt32Array::from(vec![Some(0), None, Some(3)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_or_dyn(left.clone(), right.clone())?;
        let expected = UInt32Array::from(vec![Some(13), None, Some(15)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_xor_dyn(left.clone(), right.clone())?;
        let expected = UInt32Array::from(vec![Some(13), None, Some(12)]);
        assert_eq!(result.as_ref(), &expected);

        Ok(())
    }

    #[test]
    fn bitwise_shift_array_test() -> Result<()> {
        let input = Arc::new(Int32Array::from(vec![Some(2), None, Some(10)])) as ArrayRef;
        let modules =
            Arc::new(Int32Array::from(vec![Some(2), Some(4), Some(8)])) as ArrayRef;
        let mut result = bitwise_shift_left_dyn(input.clone(), modules.clone())?;

        let expected = Int32Array::from(vec![Some(8), None, Some(2560)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_shift_right_dyn(result.clone(), modules.clone())?;
        assert_eq!(result.as_ref(), &input);

        let input =
            Arc::new(UInt32Array::from(vec![Some(2), None, Some(10)])) as ArrayRef;
        let modules =
            Arc::new(UInt32Array::from(vec![Some(2), Some(4), Some(8)])) as ArrayRef;
        let mut result = bitwise_shift_left_dyn(input.clone(), modules.clone())?;

        let expected = UInt32Array::from(vec![Some(8), None, Some(2560)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_shift_right_dyn(result.clone(), modules.clone())?;
        assert_eq!(result.as_ref(), &input);
        Ok(())
    }

    #[test]
    fn bitwise_shift_array_overflow_test() -> Result<()> {
        let input = Arc::new(Int32Array::from(vec![Some(2)])) as ArrayRef;
        let modules = Arc::new(Int32Array::from(vec![Some(100)])) as ArrayRef;
        let result = bitwise_shift_left_dyn(input.clone(), modules.clone())?;

        let expected = Int32Array::from(vec![Some(32)]);
        assert_eq!(result.as_ref(), &expected);

        let input = Arc::new(UInt32Array::from(vec![Some(2)])) as ArrayRef;
        let modules = Arc::new(UInt32Array::from(vec![Some(100)])) as ArrayRef;
        let result = bitwise_shift_left_dyn(input.clone(), modules.clone())?;

        let expected = UInt32Array::from(vec![Some(32)]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn bitwise_scalar_test() -> Result<()> {
        let left = Arc::new(Int32Array::from(vec![Some(12), None, Some(11)])) as ArrayRef;
        let right = ScalarValue::from(3i32);
        let mut result = bitwise_and_dyn_scalar(&left, right.clone()).unwrap()?;
        let expected = Int32Array::from(vec![Some(0), None, Some(3)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_or_dyn_scalar(&left, right.clone()).unwrap()?;
        let expected = Int32Array::from(vec![Some(15), None, Some(11)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_xor_dyn_scalar(&left, right).unwrap()?;
        let expected = Int32Array::from(vec![Some(15), None, Some(8)]);
        assert_eq!(result.as_ref(), &expected);

        let left =
            Arc::new(UInt32Array::from(vec![Some(12), None, Some(11)])) as ArrayRef;
        let right = ScalarValue::from(3u32);
        let mut result = bitwise_and_dyn_scalar(&left, right.clone()).unwrap()?;
        let expected = UInt32Array::from(vec![Some(0), None, Some(3)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_or_dyn_scalar(&left, right.clone()).unwrap()?;
        let expected = UInt32Array::from(vec![Some(15), None, Some(11)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_xor_dyn_scalar(&left, right).unwrap()?;
        let expected = UInt32Array::from(vec![Some(15), None, Some(8)]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn bitwise_shift_scalar_test() -> Result<()> {
        let input = Arc::new(Int32Array::from(vec![Some(2), None, Some(4)])) as ArrayRef;
        let module = ScalarValue::from(10i32);
        let mut result =
            bitwise_shift_left_dyn_scalar(&input, module.clone()).unwrap()?;

        let expected = Int32Array::from(vec![Some(2048), None, Some(4096)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_shift_right_dyn_scalar(&result, module).unwrap()?;
        assert_eq!(result.as_ref(), &input);

        let input = Arc::new(UInt32Array::from(vec![Some(2), None, Some(4)])) as ArrayRef;
        let module = ScalarValue::from(10u32);
        let mut result =
            bitwise_shift_left_dyn_scalar(&input, module.clone()).unwrap()?;

        let expected = UInt32Array::from(vec![Some(2048), None, Some(4096)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_shift_right_dyn_scalar(&result, module).unwrap()?;
        assert_eq!(result.as_ref(), &input);
        Ok(())
    }

    #[test]
    fn test_display_and_or_combo() {
        let expr = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                lit(ScalarValue::from(1)),
                Operator::And,
                lit(ScalarValue::from(2)),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                lit(ScalarValue::from(3)),
                Operator::And,
                lit(ScalarValue::from(4)),
            )),
        );
        assert_eq!(expr.to_string(), "1 AND 2 AND 3 AND 4");

        let expr = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                lit(ScalarValue::from(1)),
                Operator::Or,
                lit(ScalarValue::from(2)),
            )),
            Operator::Or,
            Arc::new(BinaryExpr::new(
                lit(ScalarValue::from(3)),
                Operator::Or,
                lit(ScalarValue::from(4)),
            )),
        );
        assert_eq!(expr.to_string(), "1 OR 2 OR 3 OR 4");

        let expr = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                lit(ScalarValue::from(1)),
                Operator::And,
                lit(ScalarValue::from(2)),
            )),
            Operator::Or,
            Arc::new(BinaryExpr::new(
                lit(ScalarValue::from(3)),
                Operator::And,
                lit(ScalarValue::from(4)),
            )),
        );
        assert_eq!(expr.to_string(), "1 AND 2 OR 3 AND 4");

        let expr = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                lit(ScalarValue::from(1)),
                Operator::Or,
                lit(ScalarValue::from(2)),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                lit(ScalarValue::from(3)),
                Operator::Or,
                lit(ScalarValue::from(4)),
            )),
        );
        assert_eq!(expr.to_string(), "(1 OR 2) AND (3 OR 4)");
    }

    #[test]
    fn test_to_result_type_array() {
        let values = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let keys = Int8Array::from(vec![Some(0), None, Some(2), Some(3)]);
        let dictionary =
            Arc::new(DictionaryArray::try_new(keys, values).unwrap()) as ArrayRef;

        // Casting Dictionary to Int32
        let casted =
            to_result_type_array(&Operator::Plus, dictionary.clone(), &DataType::Int32)
                .unwrap();
        assert_eq!(
            &casted,
            &(Arc::new(Int32Array::from(vec![Some(1), None, Some(3), Some(4)]))
                as ArrayRef)
        );

        // Array has same datatype as result type, no casting
        let casted = to_result_type_array(
            &Operator::Plus,
            dictionary.clone(),
            dictionary.data_type(),
        )
        .unwrap();
        assert_eq!(&casted, &dictionary);

        // Not numerical operator, no casting
        let casted =
            to_result_type_array(&Operator::Eq, dictionary.clone(), &DataType::Int32)
                .unwrap();
        assert_eq!(&casted, &dictionary);
    }

    #[test]
    fn test_bin_eq() -> Result<()> {
        // 3 - ( (5+(-2)) - (4+1) )
        let lhs = BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::from(3))),
            Operator::Minus,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(5))),
                    Operator::Plus,
                    Arc::new(Literal::new(ScalarValue::from(-2))),
                )),
                Operator::Minus,
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(4))),
                    Operator::Plus,
                    Arc::new(Literal::new(ScalarValue::from(1))),
                )),
            )),
        );
        // 4 + ( (3-(-1)) + (2+(-5)) )
        let rhs = BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::from(4))),
            Operator::Plus,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(3))),
                    Operator::Minus,
                    Arc::new(Literal::new(ScalarValue::from(-1))),
                )),
                Operator::Plus,
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(2))),
                    Operator::Plus,
                    Arc::new(Literal::new(ScalarValue::from(-5))),
                )),
            )),
        );
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs, &rhs, &None, &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs, &lhs, &None, &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);

        // (2*b) >= (3+a) AND 3.0 < (c*100)
        let lhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(2))),
                    Operator::Multiply,
                    Arc::new(Column::new("b", 2)),
                )),
                Operator::GtEq,
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(3))),
                    Operator::Plus,
                    Arc::new(Column::new("a", 1)),
                )),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                Arc::new(Literal::new(ScalarValue::from(3.0))),
                Operator::Lt,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("c", 3)),
                    Operator::Multiply,
                    Arc::new(Literal::new(ScalarValue::from(100))),
                )),
            )),
        );
        // (100*c) > 3.0 AND (b*2) >= (a+3)
        let rhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(100))),
                    Operator::Multiply,
                    Arc::new(Column::new("c", 3)),
                )),
                Operator::Gt,
                Arc::new(Literal::new(ScalarValue::from(3.0))),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("b", 2)),
                    Operator::Multiply,
                    Arc::new(Literal::new(ScalarValue::from(2))),
                )),
                Operator::GtEq,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 1)),
                    Operator::Plus,
                    Arc::new(Literal::new(ScalarValue::from(3))),
                )),
            )),
        );
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs, &rhs, &None, &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(HashMap::new()),
            &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs, &lhs, &None, &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(HashMap::new()),
            &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);

        // (a + ((b+2) / (c*5))) * d
        let lhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Plus,
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("b", 2)),
                        Operator::Plus,
                        Arc::new(Literal::new(ScalarValue::from(2))),
                    )),
                    Operator::Divide,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("c", 3)),
                        Operator::Multiply,
                        Arc::new(Literal::new(ScalarValue::from(5))),
                    )),
                )),
            )),
            Operator::Multiply,
            Arc::new(Column::new("d", 4)),
        );
        //  d * (((2+b) / (5*c)) + a)
        let rhs = BinaryExpr::new(
            Arc::new(Column::new("d", 4)),
            Operator::Multiply,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Literal::new(ScalarValue::from(2))),
                        Operator::Plus,
                        Arc::new(Column::new("b", 2)),
                    )),
                    Operator::Divide,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Literal::new(ScalarValue::from(5))),
                        Operator::Multiply,
                        Arc::new(Column::new("c", 3)),
                    )),
                )),
                Operator::Plus,
                Arc::new(Column::new("a", 1)),
            )),
        );
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs, &rhs, &None, &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(HashMap::new()),
            &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs, &lhs, &None, &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(HashMap::new()),
            &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);

        // (( (int)a * (int)b ) * (int)c ) / (float)d )
        let lhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 1)),
                    Operator::Multiply,
                    Arc::new(Column::new("b", 2)),
                )),
                Operator::Multiply,
                Arc::new(Column::new("c", 3)),
            )),
            Operator::Divide,
            Arc::new(Column::new("d", 4)),
        );
        // (( (int)c * (int)a ) * (int)b ) / (float)d )
        let rhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("c", 3)),
                    Operator::Multiply,
                    Arc::new(Column::new("a", 1)),
                )),
                Operator::Multiply,
                Arc::new(Column::new("b", 2)),
            )),
            Operator::Divide,
            Arc::new(Column::new("d", 4)),
        );
        let column_map: HashMap<Column, DataType> = vec![
            (Column::new("a", 1), DataType::Int32),
            (Column::new("b", 2), DataType::Int32),
            (Column::new("c", 3), DataType::Int32),
            (Column::new("d", 4), DataType::Float64),
        ]
        .into_iter()
        .collect();
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(column_map.clone()),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(column_map),
            &false
        )?);
        let column_map: HashMap<Column, DataType> =
            vec![(Column::new("d", 4), DataType::Float64)]
                .into_iter()
                .collect();
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(column_map.clone()),
            &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs, &rhs, &None, &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(column_map),
            &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs, &lhs, &None, &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);

        // ( (unknown)a * (int)b ) / ( (int)c * (int)d )
        let lhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Multiply,
                Arc::new(Column::new("b", 2)),
            )),
            Operator::Divide,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("c", 3)),
                Operator::Multiply,
                Arc::new(Column::new("d", 4)),
            )),
        );
        // ( (unknown)a * (int)b ) / ( (int)d * (int)c )
        let rhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Multiply,
                Arc::new(Column::new("b", 2)),
            )),
            Operator::Divide,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("d", 4)),
                Operator::Multiply,
                Arc::new(Column::new("c", 3)),
            )),
        );
        let column_map: HashMap<Column, DataType> = vec![
            (Column::new("b", 2), DataType::Int32),
            (Column::new("c", 3), DataType::Int32),
            (Column::new("d", 4), DataType::Int32),
        ]
        .into_iter()
        .collect();
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(column_map.clone()),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(HashMap::new()),
            &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(column_map),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(HashMap::new()),
            &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);

        // (float)a / ((float)b * ((float)c / (float)d))
        let lhs = BinaryExpr::new(
            Arc::new(Column::new("a", 1)),
            Operator::Divide,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("b", 2)),
                Operator::Multiply,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("c", 3)),
                    Operator::Divide,
                    Arc::new(Column::new("d", 4)),
                )),
            )),
        );
        // ((float)a / (float)b) * ((float)d / (float)c)
        let rhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Divide,
                Arc::new(Column::new("b", 2)),
            )),
            Operator::Multiply,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("d", 4)),
                Operator::Divide,
                Arc::new(Column::new("c", 3)),
            )),
        );
        let column_map: HashMap<Column, DataType> = vec![
            (Column::new("a", 1), DataType::Float32),
            (Column::new("b", 2), DataType::Float64),
            (Column::new("c", 3), DataType::Float64),
            (Column::new("d", 4), DataType::Float32),
        ]
        .into_iter()
        .collect();
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(column_map.clone()),
            &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs, &rhs, &None, &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs, &rhs, &None, &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(column_map),
            &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs, &lhs, &None, &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs, &lhs, &None, &true
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);

        // (a*b) / c
        let lhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Multiply,
                Arc::new(Column::new("b", 2)),
            )),
            Operator::Divide,
            Arc::new(Column::new("c", 3)),
        );
        // (a/c) * b
        let rhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Divide,
                Arc::new(Column::new("c", 3)),
            )),
            Operator::Multiply,
            Arc::new(Column::new("b", 2)),
        );
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);
        for (lhs, rhs) in [(lhs.clone(), rhs.clone()), (rhs, lhs)] {
            assert!(BinaryExprEquivalenceChecker::is_equal(
                &lhs, &rhs, &None, &false
            )?);
            assert!(BinaryExprEquivalenceChecker::is_equal(
                &lhs,
                &rhs,
                &Some(
                    vec![
                        (Column::new("b", 2), DataType::Int32),
                        (Column::new("a", 1), DataType::Float32)
                    ]
                    .into_iter()
                    .collect()
                ),
                &true
            )?);
            assert!(BinaryExprEquivalenceChecker::is_equal(
                &lhs,
                &rhs,
                &Some(
                    vec![
                        (Column::new("c", 3), DataType::Int32),
                        (Column::new("a", 1), DataType::Float32)
                    ]
                    .into_iter()
                    .collect()
                ),
                &true
            )?);
        }

        Ok(())
    }

    #[test]
    fn test_bin_not_eq() -> Result<()> {
        // (a*b) / c
        let lhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Multiply,
                Arc::new(Column::new("b", 2)),
            )),
            Operator::Divide,
            Arc::new(Column::new("c", 3)),
        );
        // (a/c) * b
        let rhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Divide,
                Arc::new(Column::new("c", 3)),
            )),
            Operator::Multiply,
            Arc::new(Column::new("b", 2)),
        );
        for (lhs, rhs) in [(lhs.clone(), rhs.clone()), (rhs, lhs)] {
            assert!(!BinaryExprEquivalenceChecker::is_equal(
                &lhs,
                &rhs,
                &Some(HashMap::new()),
                &false
            )?);
            assert!(!BinaryExprEquivalenceChecker::is_equal(
                &lhs,
                &rhs,
                &Some(
                    vec![
                        (Column::new("b", 2), DataType::Int32),
                        (Column::new("a", 1), DataType::Float32)
                    ]
                    .into_iter()
                    .collect()
                ),
                &false
            )?);
            assert!(!BinaryExprEquivalenceChecker::is_equal(
                &lhs,
                &rhs,
                &Some(
                    vec![
                        (Column::new("c", 3), DataType::Int32),
                        (Column::new("a", 1), DataType::Float32)
                    ]
                    .into_iter()
                    .collect()
                ),
                &false
            )?);
        }
        // a and b are floating point types
        // a * b
        // b * a
        let lhs = BinaryExpr::new(
            Arc::new(Column::new("a", 1)),
            Operator::Multiply,
            Arc::new(Column::new("b", 2)),
        );
        let rhs = BinaryExpr::new(
            Arc::new(Column::new("b", 2)),
            Operator::Multiply,
            Arc::new(Column::new("a", 1)),
        );
        for (lhs, rhs) in [(lhs.clone(), rhs.clone()), (rhs, lhs)] {
            assert!(!BinaryExprEquivalenceChecker::is_equal(
                &lhs,
                &rhs,
                &Some(HashMap::new()),
                &false
            )?);
            assert!(!BinaryExprEquivalenceChecker::is_equal(
                &lhs,
                &rhs,
                &Some(
                    vec![(Column::new("a", 1), DataType::Float32)]
                        .into_iter()
                        .collect()
                ),
                &false
            )?);
            assert!(!BinaryExprEquivalenceChecker::is_equal(
                &lhs,
                &rhs,
                &Some(
                    vec![(Column::new("b", 2), DataType::Float32)]
                        .into_iter()
                        .collect()
                ),
                &false
            )?);
        }

        // 2 > b
        let lhs = BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::from(2))),
            Operator::Gt,
            Arc::new(Column::new("b", 2)),
        );
        // 2 >= b
        let rhs = BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::from(2))),
            Operator::GtEq,
            Arc::new(Column::new("b", 2)),
        );
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &lhs, &rhs, &None, &true
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &rhs, &lhs, &None, &true
        )?);

        // (3.3 + a)
        let lhs = BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::from(3.3))),
            Operator::Plus,
            Arc::new(Column::new("a", 1)),
        );
        // (a + 3.3)
        let rhs = BinaryExpr::new(
            Arc::new(Column::new("a", 1)),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::from(3.3))),
        );
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);

        // 3 - ( (5+(-2)) + (1-4) )
        let lhs = BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::from(3))),
            Operator::Minus,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(5))),
                    Operator::Plus,
                    Arc::new(Literal::new(ScalarValue::from(-2))),
                )),
                Operator::Plus,
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(1))),
                    Operator::Minus,
                    Arc::new(Literal::new(ScalarValue::from(4))),
                )),
            )),
        );
        // 3 - ( (4+(-2)) + (2-4) )
        let rhs = BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::from(3))),
            Operator::Minus,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(4))),
                    Operator::Plus,
                    Arc::new(Literal::new(ScalarValue::from(-2))),
                )),
                Operator::Plus,
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(2))),
                    Operator::Minus,
                    Arc::new(Literal::new(ScalarValue::from(4))),
                )),
            )),
        );
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &lhs, &rhs, &None, &true
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &rhs, &lhs, &None, &true
        )?);

        // (c*d) + ( (a-2) + (3+b) )
        let lhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("c", 3)),
                Operator::Multiply,
                Arc::new(Column::new("d", 4)),
            )),
            Operator::Plus,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 1)),
                    Operator::Minus,
                    Arc::new(Literal::new(ScalarValue::from(2))),
                )),
                Operator::Plus,
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(3))),
                    Operator::Plus,
                    Arc::new(Column::new("b", 2)),
                )),
            )),
        );
        // ( (b+(-2)) - (3-a) ) + (c*d)
        let rhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("b", 2)),
                    Operator::Plus,
                    Arc::new(Literal::new(ScalarValue::from(-2))),
                )),
                Operator::Minus,
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(3))),
                    Operator::Minus,
                    Arc::new(Column::new("a", 1)),
                )),
            )),
            Operator::Plus,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("c", 3)),
                Operator::Multiply,
                Arc::new(Column::new("d", 4)),
            )),
        );
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &lhs, &rhs, &None, &true
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &rhs, &lhs, &None, &true
        )?);

        // (2*b) >= (3+a) AND (3.0*2.0) < (c*100)
        let lhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(2))),
                    Operator::Multiply,
                    Arc::new(Column::new("b", 2)),
                )),
                Operator::GtEq,
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(3))),
                    Operator::Plus,
                    Arc::new(Column::new("a", 1)),
                )),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(3.0))),
                    Operator::Multiply,
                    Arc::new(Literal::new(ScalarValue::from(2.0))),
                )),
                Operator::Lt,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("c", 3)),
                    Operator::Multiply,
                    Arc::new(Literal::new(ScalarValue::from(100))),
                )),
            )),
        );
        // (100*c) > (2.0*3.0) AND (2*b) >= (a+3)
        let rhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(100))),
                    Operator::Multiply,
                    Arc::new(Column::new("c", 3)),
                )),
                Operator::Gt,
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(2.0))),
                    Operator::Multiply,
                    Arc::new(Literal::new(ScalarValue::from(3.0))),
                )),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::from(2))),
                    Operator::Multiply,
                    Arc::new(Column::new("b", 2)),
                )),
                Operator::GtEq,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 1)),
                    Operator::Plus,
                    Arc::new(Literal::new(ScalarValue::from(3))),
                )),
            )),
        );
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);

        // (( (int)a * (float)b ) * (int)c ) / (float)d )
        let lhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 1)),
                    Operator::Multiply,
                    Arc::new(Column::new("b", 2)),
                )),
                Operator::Multiply,
                Arc::new(Column::new("c", 3)),
            )),
            Operator::Divide,
            Arc::new(Column::new("d", 4)),
        );
        // (( (int)c * (int)a ) * (float)b ) / (float)d )
        let rhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("c", 3)),
                    Operator::Multiply,
                    Arc::new(Column::new("a", 1)),
                )),
                Operator::Multiply,
                Arc::new(Column::new("b", 2)),
            )),
            Operator::Divide,
            Arc::new(Column::new("d", 4)),
        );
        let column_map: HashMap<Column, DataType> = vec![
            (Column::new("a", 1), DataType::Int32),
            (Column::new("b", 2), DataType::Float32),
            (Column::new("c", 3), DataType::Int32),
            (Column::new("d", 4), DataType::Float64),
        ]
        .into_iter()
        .collect();
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(column_map.clone()),
            &false
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(column_map),
            &false
        )?);

        // ( (int)a * (int)b ) / ( (unknown)c * (int)d )
        let lhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Multiply,
                Arc::new(Column::new("b", 2)),
            )),
            Operator::Divide,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("c", 3)),
                Operator::Multiply,
                Arc::new(Column::new("d", 4)),
            )),
        );
        // ( (int)a * (int)b ) / ( (int)d * (unknown)c )
        let rhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Multiply,
                Arc::new(Column::new("b", 2)),
            )),
            Operator::Divide,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("d", 4)),
                Operator::Multiply,
                Arc::new(Column::new("c", 3)),
            )),
        );
        let column_map: HashMap<Column, DataType> = vec![
            (Column::new("a", 1), DataType::Int32),
            (Column::new("b", 2), DataType::Int32),
            (Column::new("d", 4), DataType::Int32),
        ]
        .into_iter()
        .collect();
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(column_map.clone()),
            &false
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(column_map),
            &false
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);

        // a / (b * (c / d))
        let lhs = BinaryExpr::new(
            Arc::new(Column::new("a", 1)),
            Operator::Divide,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("b", 2)),
                Operator::Multiply,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("c", 3)),
                    Operator::Divide,
                    Arc::new(Column::new("d", 4)),
                )),
            )),
        );
        // (a / b) * (d / c)
        let rhs = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Divide,
                Arc::new(Column::new("b", 2)),
            )),
            Operator::Multiply,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("d", 4)),
                Operator::Divide,
                Arc::new(Column::new("c", 3)),
            )),
        );

        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &lhs,
            &rhs,
            &Some(HashMap::new()),
            &true
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(HashMap::new()),
            &false
        )?);
        assert!(!BinaryExprEquivalenceChecker::is_equal(
            &rhs,
            &lhs,
            &Some(HashMap::new()),
            &true
        )?);

        Ok(())
    }

    #[test]
    fn interval_dt_mdn_cmp() {
        // DayTime definitely true cases:
        let a = ColumnarValue::Array(Arc::new(IntervalDayTimeArray::from(vec![
            Some(IntervalDayTimeType::make_value(0, -5)),
            Some(IntervalDayTimeType::make_value(3, -1_000_000)),
            Some(IntervalDayTimeType::make_value(4, -1000)),
            Some(IntervalDayTimeType::make_value(10, 20)),
            Some(IntervalDayTimeType::make_value(1, 2)),
        ])));
        let b = ColumnarValue::Array(Arc::new(IntervalDayTimeArray::from(vec![
            Some(IntervalDayTimeType::make_value(0, -10)),
            Some(IntervalDayTimeType::make_value(3, -2_000_000)),
            Some(IntervalDayTimeType::make_value(2, 1000)),
            Some(IntervalDayTimeType::make_value(5, 6)),
            Some(IntervalDayTimeType::make_value(1, 1)),
        ])));
        let res = apply_interval_cmp(&a, &b, Operator::Gt).unwrap();
        let res_eq = apply_interval_cmp(&a, &b, Operator::GtEq).unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(i.unwrap());
        }
        let ColumnarValue::Array(arr) = res_eq else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(i.unwrap());
        }
        let res = apply_interval_cmp(&b, &a, Operator::Lt).unwrap();
        let res_eq = apply_interval_cmp(&b, &a, Operator::LtEq).unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(i.unwrap());
        }
        let ColumnarValue::Array(arr) = res_eq else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(i.unwrap());
        }
        // DayTime indefinite cases:
        let a = ColumnarValue::Array(Arc::new(IntervalDayTimeArray::from(vec![
            Some(IntervalDayTimeType::make_value(1, 0)),
            Some(IntervalDayTimeType::make_value(3, -86_400_001)),
        ])));
        let b = ColumnarValue::Array(Arc::new(IntervalDayTimeArray::from(vec![
            Some(IntervalDayTimeType::make_value(0, 86_400_999)),
            Some(IntervalDayTimeType::make_value(2, 0)),
        ])));
        let res = apply_interval_cmp(&a, &b, Operator::Gt).unwrap();
        let res_eq = apply_interval_cmp(&a, &b, Operator::GtEq).unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(!i.unwrap());
        }
        let ColumnarValue::Array(arr) = res_eq else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(!i.unwrap());
        }
        let res = apply_interval_cmp(&b, &a, Operator::Lt).unwrap();
        let res_eq = apply_interval_cmp(&b, &a, Operator::LtEq).unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(!i.unwrap());
        }
        let ColumnarValue::Array(arr) = res_eq else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(!i.unwrap());
        }
        // MonthDayNano definitely true cases:
        let a = ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![
            Some(IntervalMonthDayNanoType::make_value(0, 0, 1)),
            Some(IntervalMonthDayNanoType::make_value(0, 1, -1_000_000_000)),
            Some(IntervalMonthDayNanoType::make_value(3, 2, -100_000_000_000)),
            Some(IntervalMonthDayNanoType::make_value(
                0,
                1,
                86_401_000_000_001,
            )),
            Some(IntervalMonthDayNanoType::make_value(1, 32, 0)),
        ])));
        let b = ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![
            Some(IntervalMonthDayNanoType::make_value(0, 0, 0)),
            Some(IntervalMonthDayNanoType::make_value(0, 1, -8_000_000_000)),
            Some(IntervalMonthDayNanoType::make_value(1, 25, 100_000_000_000)),
            Some(IntervalMonthDayNanoType::make_value(0, 2, 0)),
            Some(IntervalMonthDayNanoType::make_value(2, 0, 0)),
        ])));
        let res = apply_interval_cmp(&a, &b, Operator::Gt).unwrap();
        let res_eq = apply_interval_cmp(&a, &b, Operator::GtEq).unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(i.unwrap());
        }
        let ColumnarValue::Array(arr) = res_eq else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(i.unwrap());
        }
        let res = apply_interval_cmp(&b, &a, Operator::Lt).unwrap();
        let res_eq = apply_interval_cmp(&b, &a, Operator::LtEq).unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(i.unwrap());
        }
        let ColumnarValue::Array(arr) = res_eq else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(i.unwrap());
        }
        // MonthDayNano indefinite cases:
        let a = ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![
            Some(IntervalMonthDayNanoType::make_value(0, 30, 0)),
            Some(IntervalMonthDayNanoType::make_value(1, 0, 1_000_000_000)),
            Some(IntervalMonthDayNanoType::make_value(
                0,
                0,
                86_400_000_000_000,
            )),
        ])));
        let b = ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![
            Some(IntervalMonthDayNanoType::make_value(1, 0, 0)),
            Some(IntervalMonthDayNanoType::make_value(2, -29, 8_000_000_000)),
            Some(IntervalMonthDayNanoType::make_value(0, 1, 0)),
        ])));
        let res = apply_interval_cmp(&a, &b, Operator::Gt).unwrap();
        let res_eq = apply_interval_cmp(&a, &b, Operator::GtEq).unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(!i.unwrap());
        }
        let ColumnarValue::Array(arr) = res_eq else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(!i.unwrap());
        }
        let res = apply_interval_cmp(&b, &a, Operator::Lt).unwrap();
        let res_eq = apply_interval_cmp(&b, &a, Operator::LtEq).unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(!i.unwrap());
        }
        let ColumnarValue::Array(arr) = res_eq else {
            panic!()
        };
        for i in arr.as_any().downcast_ref::<BooleanArray>().unwrap() {
            assert!(!i.unwrap());
        }
    }
}
