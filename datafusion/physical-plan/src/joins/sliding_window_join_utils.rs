// Copyright (C) Synnada, Inc. - All Rights Reserved.
// This file does not contain any Apache Software Foundation copyrighted code.

use std::task::Poll;

use crate::joins::{
    stream_join_utils::{
        calculate_side_prune_length_helper, get_build_side_pruned_exprs,
        get_filter_representation_of_join_side,
        get_filter_representation_schema_of_build_side, get_pruning_anti_indices,
        get_pruning_semi_indices, SortedFilterExpr, StreamJoinStateResult,
    },
    symmetric_hash_join::StreamJoinMetrics,
    utils::{self, append_right_indices, get_anti_indices, get_semi_indices, JoinFilter},
};
use crate::{handle_async_state, handle_state};

use arrow::compute::concat_batches;
use arrow_array::{
    builder::{PrimitiveBuilder, UInt32Builder, UInt64Builder},
    types::{UInt32Type, UInt64Type},
    ArrowPrimitiveType, NativeAdapter, PrimitiveArray, RecordBatch, UInt32Array,
    UInt64Array,
};
use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, JoinSide, JoinType, Result, ScalarValue};
use datafusion_execution::SendableRecordBatchStream;
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_physical_expr::expressions::{Column, PhysicalSortExpr};
use datafusion_physical_expr::intervals::cp_solver::ExprIntervalGraph;
use datafusion_physical_expr::Partitioning;

use async_trait::async_trait;
use futures::{ready, FutureExt, StreamExt};
use hashbrown::HashSet;
use std::{mem, usize};

/// We use this buffer to keep track of the probe side pulling.
pub struct ProbeBuffer {
    /// The batch used for join operations.
    pub(crate) current_batch: RecordBatch,
    /// The batches buffered in `ProbePull` state.
    pub(crate) candidate_buffer: Vec<RecordBatch>,
    /// Join keys/columns.
    pub(crate) on: Vec<Column>,
}

impl ProbeBuffer {
    /// Creates a new `ProbeBuffer` with the given schema and join keys.
    ///
    /// # Arguments
    ///
    /// * `schema` - The schema of the input batches.
    /// * `on` - A vector storing join columns.
    ///
    /// # Returns
    ///
    /// A new `ProbeBuffer`.
    pub fn new(schema: SchemaRef, on: Vec<Column>) -> Self {
        Self {
            current_batch: RecordBatch::new_empty(schema),
            candidate_buffer: vec![],
            on,
        }
    }

    /// Returns the size of this `ProbeBuffer` in bytes.
    pub fn size(&self) -> usize {
        let mut size = 0;
        size += self.current_batch.get_array_memory_size();
        size += std::mem::size_of_val(&self.on);
        size
    }
}

/// Represents a lazy join computation on two data streams.
///
/// The `LazyJoinStream` trait defines the core logic for a lazy join operation
/// on two streams of record batches. In contrast to an eager join, which
/// immediately computes the join for the provided data, a lazy join computes
/// the join as batches are pulled from the input streams. This allows for potential
/// optimizations, such as skipping batches that are not relevant for the join.
///
/// Implementing this trait requires defining various asynchronous methods
/// representing different stages of the join process, such as pulling data from
/// the probe and build sides, performing the join, and handling end-of-stream conditions.
///
/// ```text
///              Build
///            +---------+
///            | a  | b  |        Probe
///            |---------|       +-------+
///            | 1  | a  |       | x | y |
///            |    |    |       |-------|
///            | 2  | b  |       | 3 | a |
///            |    |    |       |   |   |
///            | 3  | c  |       | 4 | v |
///            |    |    |       |   |   |
///            | 5  | c  |       | 5 | a |
///            |    |    |       |   |   |
///            | 6  | a  |       | 6 | x |
///            |    |    |       |   |   |
///            | 8  | a  |       +-------+
///            |    |    |
///            | 8  | d  |
///            |    |    |
///            | 10 | c  |
///            |    |    |
///            | 12 | y  |
///            |    |    |
///            +---------+
///
///  Join conditions: b = y AND a > x + 3 AND a < x + 8
///
///  These conditions imply a sliding window since left and right side tables
///  are ordered according to columns `a` and `x`, respectively.
///
///  We use this information to partially materialize and prune the build side.
///
///              Build
///            +---------+
///            | a  | b  |        Probe
///            |---------|       +-------+
///            | 1  | a  |       | x | y |
///            |    |    |      /|-------|
///            | 2  | b  |    /  | 3 | a |
///            |    |    |   /   |   |   |
///            | 3  | c  | /     | 4 | v |
///            |    |    |/      |   |   |
///            | 5  | c  |       | 5 | a |
///            |    |    |       |   |   |
///            | 6  | a  |       | 6 | x |
///            |    |    |       |   |   |
///            | 8  | a  |      |+-------+
///            |    |    |      /
///            | 8  | d  |     |
///            |    |    |     /
///            | 10 | c  |    |
///            |    |    |    /
///            | 12 | y  |   |
///            |    |    |   /
///            +---------+  |
///                         / Joinable range
///                        |
///                        |
///            -------------
///
///
///  Probe side requires data from the build side as long as `a` satisfies
///  a < 6 + 8 = 14. Thus, we should fetch the build side until we see a value
///  in column `a` greater than 14. This is how we guarantee that we can match
///  and emit probe side data with all possible rows from the build side.
/// ```
///
/// # Methods
///
/// * `fetch_and_process_next_from_probe_stream`: Asynchronously handles pulling
///   data from the probe stream.
/// * `fetch_and_process_build_batches_by_interval`: Asynchronously handles pulling
///   data from the build stream, given a certain interval of interest.
/// * `process_join_operation`: Asynchronously performs the join operation on the
///   current batches.
/// * `handle_left_stream_end`: Asynchronously handles the situation when the build
///   stream is exhausted.
/// * `process_probe_stream_end`: Asynchronously handles the situation when the probe
///   stream is exhausted.
/// * `process_batches_before_finalization`: Asynchronously handles the situation when
///   both streams are exhausted.
/// * `poll_next_impl`: Continuously polls the next record batch based on the current
///   state of the join.
///
/// Additionally, various helper methods provide access to internal attributes,
/// metrics, and state management.
///
/// # Async Traits
///
/// This trait makes use of the `async_trait` attribute to allow asynchronous method definitions.
#[async_trait]
pub trait LazyJoinStream {
    /// Asynchronously pulls the next batch from the right (probe) stream.
    ///
    /// This default implementation repeatedly polls the probe stream until it finds a suitable batch for interval
    /// calculation. If no batches are found, the state is set to `ProbeExhausted`.
    ///
    /// # Returns
    ///
    /// * `Result<StreamJoinStateResult<Option<RecordBatch>>>` - The state result after processing the probe batch.
    async fn fetch_and_process_next_from_probe_stream(
        &mut self,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        let mut continue_polling = true;
        while continue_polling {
            match self.mut_probe_stream().next().await {
                Some(Ok(batch)) => {
                    // Update metrics for polled batch:
                    self.metrics().right.input_batches.add(1);
                    self.metrics().right.input_rows.add(batch.num_rows());
                    // Check if the batch meets interval calculation criteria:
                    continue_polling = !is_batch_suitable_interval_calculation(
                        self.filter(),
                        self.probe_sorted_filter_expr(),
                        &batch,
                        JoinSide::Right,
                    )?;
                    // Add the batch into the candidate buffer:
                    self.probe_buffer().candidate_buffer.push(batch);
                }
                Some(Err(e)) => return Err(e),
                None => break,
            }
        }

        if self.probe_buffer().candidate_buffer.is_empty() {
            // If no batches were collected, change state to "ProbeExhausted":
            self.set_state(LazyJoinStreamState::ProbeExhausted);
            return Ok(StreamJoinStateResult::Continue);
        }

        // Get probe batch by joining all the collected batches:
        self.probe_buffer().current_batch =
            get_probe_batch(std::mem::take(&mut self.probe_buffer().candidate_buffer))?;

        if self.probe_buffer().current_batch.num_rows() == 0 {
            return Ok(StreamJoinStateResult::Continue);
        }

        // Calculate the necessary build side interval with probe RecordBatch
        let interval = self.calculate_the_necessary_build_side_range()?;
        // Update state to "PullBuild" with the calculated interval:
        self.set_state(LazyJoinStreamState::PullBuild { interval });
        Ok(StreamJoinStateResult::Continue)
    }

    /// Asynchronously pulls batches from the build stream based on the given
    /// interval.
    ///
    /// Polls the build stream for batches that fit the specified interval. Once
    /// the suitable buffer is reached, the state transitions to `Join`.
    ///
    /// If there are no additional batches arriving on the left side, it becomes safe to proceed
    /// with the join operation. Furthermore, the probe batch can be safely discarded
    /// after the operation because it will no longer encounter any rows from the build side.
    ///
    /// # Parameters
    ///
    /// * `interval: Vec<(PhysicalSortExpr, Interval)>` - The specified interval for
    ///   the build batches.
    ///
    /// # Returns
    ///
    /// * `Result<StreamJoinStateResult<Option<RecordBatch>>>` - The state result after processing
    ///    the build batch.
    async fn fetch_and_process_build_batches_by_interval(
        &mut self,
        interval: Vec<(PhysicalSortExpr, Interval)>,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        loop {
            // Poll the build stream for a batch.
            match self.mut_build_stream().next().await {
                Some(Ok(batch)) => {
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    self.metrics().left.input_batches.add(1);
                    self.metrics().left.input_rows.add(batch.num_rows());
                    // Update the internal state using this batch.
                    self.update_build_buffer(&batch)?;
                    // Check if the batch meets the interval criteria.
                    if !check_if_sliding_window_condition_is_met(
                        self.filter(),
                        &batch,
                        &interval,
                    )? {
                        continue;
                    }
                }
                Some(Err(e)) => return Err(e),
                None => {}
            }
            // If interval criteria are met or we exhaust the stream, switch
            // state and break the loop:
            self.set_state(LazyJoinStreamState::Join);
            break;
        }
        Ok(StreamJoinStateResult::Continue)
    }

    /// Performs the actual join of probe and build batches and handles state
    /// management.
    ///
    /// # Returns
    ///
    /// * `Result<StreamJoinStateResult<Option<RecordBatch>>>` - The result after performing the join.
    fn handle_join_operation(
        &mut self,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        let result = self.process_join_operation();
        self.set_state(LazyJoinStreamState::PullProbe);
        result
    }

    /// Performs the actual join of current probe batch and suitable build buffer. The exact joining
    /// mechanism should be defined in this method.
    ///
    /// # Returns
    ///
    /// * `Result<StreamJoinStateResult<Option<RecordBatch>>>` - The result after performing the join.
    fn process_join_operation(
        &mut self,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>>;

    /// Asynchronously handles the scenario when the right stream is exhausted.
    ///
    /// In this default implementation, when the right stream is exhausted, it
    /// attempts to pull from the left stream. If a batch is found in the left
    /// stream, it delegates the handling to `process_batch_after_right_end`.
    /// If both streams are exhausted, the state is set to indicate both streams
    /// are exhausted without final results yet.
    ///
    /// # Returns
    ///
    /// * `Result<StreamJoinStateResult<Option<RecordBatch>>>` - The state result after checking
    ///   the exhaustion state.
    async fn handle_probe_stream_end(
        &mut self,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        // Ready the left stream and match its states.
        match self.mut_build_stream().next().await {
            // If the poll returns some batch of data:
            Some(Ok(batch)) => self.process_probe_stream_end(&batch),
            Some(Err(e)) => Err(e),
            // If the poll doesn't return any data, update the state
            // to indicate both streams are exhausted:
            None => {
                self.set_state(LazyJoinStreamState::BothExhausted {
                    final_result: false,
                });
                Ok(StreamJoinStateResult::Continue)
            }
        }
    }

    /// Handles scenarios when the probe stream runs out of batches.
    ///
    /// This method addresses cases when all batches from the probe stream have
    /// been processed. It determines the next steps, either transitioning to a
    /// new state or triggering post-processing actions.
    ///
    /// # Returns
    ///
    /// * `Result<StreamJoinStateResult<Option<RecordBatch>>>` - The state result after
    ///   handling the exhaustion of the probe stream.
    fn process_probe_stream_end(
        &mut self,
        left_batch: &RecordBatch,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>>;

    /// Handles the state when both streams are exhausted and final results are
    /// yet to be produced.
    ///
    /// This default implementation switches the state to indicate both streams
    /// are exhausted with final results and then invokes the handling for this
    /// specific scenario via `process_batches_before_finalization`.
    ///
    /// # Returns
    ///
    /// * `Result<StreamJoinStateResult<Option<RecordBatch>>>` - The state result after both
    ///   streams are exhausted.
    fn prepare_for_final_results_after_exhaustion(
        &mut self,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        self.set_state(LazyJoinStreamState::BothExhausted { final_result: true });
        self.process_batches_before_finalization()
    }

    /// Prepares the stream for delivering final results after both streams are
    /// exhausted. Once both streams have been fully processed, this method
    /// ensures that any remaining results are prepared and delivered in the
    /// correct order, marking the end of the join process.
    ///
    /// # Parameters
    ///
    /// * `final_result: bool` - Indicates whether the stream has fully finished
    ///   processing or if more results may come.
    ///
    /// # Returns
    ///
    /// * `Result<StreamJoinStateResult<Option<RecordBatch>>>` - The state result containing
    ///   the final results or indicating continuation.
    fn process_batches_before_finalization(
        &mut self,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>>;

    fn poll_next_impl(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>>
    where
        Self: Send,
    {
        loop {
            return match self.state() {
                LazyJoinStreamState::PullProbe => {
                    handle_async_state!(
                        self.fetch_and_process_next_from_probe_stream(),
                        cx
                    )
                }
                LazyJoinStreamState::PullBuild { interval } => {
                    handle_async_state!(
                        self.fetch_and_process_build_batches_by_interval(interval),
                        cx
                    )
                }
                LazyJoinStreamState::ProbeExhausted => {
                    handle_async_state!(self.handle_probe_stream_end(), cx)
                }
                LazyJoinStreamState::Join => {
                    handle_state!(self.handle_join_operation())
                }
                LazyJoinStreamState::BothExhausted {
                    final_result: false,
                } => {
                    handle_state!(self.prepare_for_final_results_after_exhaustion())
                }
                LazyJoinStreamState::BothExhausted { final_result: true } => {
                    Poll::Ready(None)
                }
            };
        }
    }

    /// Returns a mutable reference to the probe stream.
    fn mut_probe_stream(&mut self) -> &mut SendableRecordBatchStream;

    /// Returns a mutable reference to the build stream.
    fn mut_build_stream(&mut self) -> &mut SendableRecordBatchStream;

    /// Returns a mutable reference to the stream metrics.
    fn metrics(&mut self) -> &mut StreamJoinMetrics;

    /// Returns the filter used for join operations.
    fn filter(&self) -> &JoinFilter;

    /// Returns a mutable reference to the sorted filter expression for the build side.
    fn mut_build_sorted_filter_expr(&mut self) -> &mut [SortedFilterExpr];

    /// Returns a mutable reference to the sorted filter expression for the probe side.
    fn mut_probe_sorted_filter_expr(&mut self) -> &mut [SortedFilterExpr];

    /// Returns a reference to the sorted filter expression for the build side.
    fn build_sorted_filter_expr(&self) -> &[SortedFilterExpr];

    /// Returns a reference to the sorted filter expression for the probe side.
    fn probe_sorted_filter_expr(&self) -> &[SortedFilterExpr];

    /// Returns a mutable reference to the probe buffer.
    fn probe_buffer(&mut self) -> &mut ProbeBuffer;

    /// Sets the current state of the join stream.
    fn set_state(&mut self, state: LazyJoinStreamState);

    /// Returns the current state of the join stream.
    fn state(&self) -> LazyJoinStreamState;

    /// Updates the build internal state based on a new batch.
    fn update_build_buffer(&mut self, batch: &RecordBatch) -> Result<()>;

    /// Calculates the necessary range for the build side based on the current probe buffer.
    fn calculate_the_necessary_build_side_range(
        &mut self,
    ) -> Result<Vec<(PhysicalSortExpr, Interval)>>;
}

/// `EagerWindowJoinOperations` provides methods for efficiently handling join
/// operations using a sliding window mechanism. This trait encapsulates methods
/// for both the build side (i.e., the side on which we maintain a buffer of data)
/// and the probe side (i.e., the side from which we pull data to match against
/// the build side).
///
/// The trait is designed with an eager approach in mind, meaning that it attempts
/// to join data as soon as possible rather than waiting for certain conditions
/// to hold.
///
/// ```text
/// The buffering works like this:
///
///    Build             Probe
///  +---------+       +--------+
///  | a  | b  |       | x | y  |
///  |---------|       |---|----|
///  | 4  | a  |       | 1 | a  |
///  |    |    |       |   |    |
///  | 4  | b  |       | 1 | b  |
///  |    |    |       |   |    |
///  | 5  | c  |       | 2 | v  |
///  |    |    |       |   |    |
///  | 6  | c  |       | 3 | a  |
///  |    |    |       |   |    |
///  | 6  | a  |       | 5 | g  |
///  |    |    |       |   |    |
///  | 8  | a  |       | 6 | h  |
///  |    |    |       |   |    |
///  | 8  | d  |       | 7 | a  |
///  |    |    |       |   |    |
///  | 10 | c  |       | 7 | g  |
///  |    |    |       |   |    |
///  | 12 | y  |       | 8 | s  |
///  |    |    |       |   |    |
///  +---------+       +--------+
///
///  Join conditions: b = y AND a > x - 8 AND a < x + 8
///
///  These conditions imply a sliding window since left and right side tables
///  are ordered according to columns `a` and `x`, respectively.
///
///  We use this information to select rows and join only once, thus maintaining
///  the probe side order.
///
///    Build             Probe
///   +---------+       +--------+
///   | a  | b  |       | x | y  |
///   |---------|       |---|----|
///   | 4  | a  |       | 1 | a  |
///   |    |    |       |   |    |
///   | 4  | b  |       | 1 | b  |
///   |    |    |       |   |    | Not joinable
///   | 5  | c  |       | 2 | v  | for subsequent
///   |    |    |       |   |    | build data
///   | 6  | c  |       | 3 | a  | ---------------
///   |    |    |      ||   |    |               |
///   | 6  | a  |     / | 5 | g  |               |
///   |    |    |     | |   |    |               |
///   | 8  | a  |    /  | 6 | h  |               |
///   |    |    |    |  |   |    |               |
///   | 8  | d  |   /   | 7 | a  |               |
///   |    |    |   |   |   |    |               |
///   | 10 | c  |  /    | 7 | g  |               |
///   |    |    |  |    |   |    |               |
///   | 12 | y  | /     | 8 | s  |               |
///   |    |    | |     |   |    |               |
///   +---------+/      +--------+               |
///        |                                     |
///        |                       Joinable      |
///        |                       for subsequent|
///        |                       build data    |
///      \ | /                                   |
///       -|-                     ---------------+
///
/// For the first 4 probe rows will not be joinable for upcoming build rows. Thus, we can
/// use them in join for current buffer and remove them. We can ensure that these rows joined
/// only once, thus the order of the join output preserved.
///
///     Build             Probe
///   +---------+       +--------+
///   | a  | b  |       | x | y  |
///   |---------|       |--------|
///   | 4  | a  |       | 1 | a  |
///   |    |    |       |   |    |
///   | 4  | b  |       | 1 | b  |
///   |    |    |       |   |    |
///   | 5  | c  |       | 2 | v  |
///   |    |    |       |   |    |
///   | 6  | c  |       | 3 | a  |
///   |    |    |       |   |    |
///   | 6  | a  |       +--------+
///   |    |    |
///   | 8  | a  |
///   |    |    |       We utilize this segment of
///   | 8  | d  |       the probe side for
///   |    |    |       the join operation
///   | 10 | c  |       and then discard it.
///   |    |    |
///   | 12 | y  |
///   |    |    |
///   +---------+
///
/// ```
///
/// Implementers of this trait are expected to manage the internal buffers of both join sides
/// and facilitate the join process based on the methods provided here.
///
/// Key functionalities include:
/// - Updating the build side's buffer with new batches of data.
/// - Determining which probe side batches can be joined with the build side.
/// - Executing the join operation.
/// - Handling data streams from both the left and right side.
///
/// Implementations should ensure that they manage memory efficiently and handle potential join
/// edge cases (like non-matching rows) appropriately.
pub trait EagerWindowJoinOperations {
    /// Updates the internal build side buffer with the provided batch.
    ///
    /// # Arguments
    ///
    /// * `batch` - The `RecordBatch` that will be added to the build side buffer.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Returns a `Result` indicating the success or failure of the operation.
    fn update_build_buffer_with_batch(&mut self, batch: RecordBatch) -> Result<()>;

    /// Joins the provided `identify_joinable_probe_batch` with the current build side buffer.
    ///
    /// # Arguments
    ///
    /// * `joinable_probe_batch` - A `RecordBatch` from the probe side that is joinable with the build side.
    ///
    /// # Returns
    ///
    /// * `Result<Option<RecordBatch>>` - Returns the joined `RecordBatch` or `None` if no joinable records are found.
    fn join_using_joinable_probe_batch(
        &mut self,
        joinable_probe_batch: RecordBatch,
    ) -> Result<Option<RecordBatch>>;

    /// Identifies a batch from the probe side that can be joined with the build side.
    ///
    /// # Returns
    ///
    /// * `Result<Option<RecordBatch>>` - Returns the joinable `RecordBatch` from the probe side or `None` if no joinable batch is found.
    fn identify_joinable_probe_batch(&mut self) -> Result<Option<RecordBatch>>;

    /// Provides mutable access to the probe buffer.
    ///
    /// # Returns
    ///
    /// * `&mut ProbeBuffer` - Returns a mutable reference to the probe buffer.
    fn get_mutable_probe_buffer(&mut self) -> &mut ProbeBuffer;

    /// Handles a batch from the left (build) stream and attempts to pull and join with data from the probe side.
    ///
    /// This function is critical in managing batches from the left stream, updating the build buffer with new data,
    /// identifying joinable batches from the probe side, and subsequently performing the join operation.
    ///
    /// # Arguments
    ///
    /// * `batch`: The `RecordBatch` from the left (build) stream to be processed.
    ///
    /// # Returns
    ///
    /// * `Result<StreamJoinStateResult<Option<RecordBatch>>>`: A result wrapping the state result. This can be:
    ///     - `StreamJoinStateResult::Ready(Some(batch))`: If there is a resultant batch from the join operation.
    ///     - `StreamJoinStateResult::Continue`: If there is no resultant batch, but the operation was successful.
    ///     - An `Err` variant: If any error occurs during the process.
    ///
    /// # Description
    ///
    /// The function starts by updating the build buffer with the incoming batch from the left stream using
    /// `update_build_buffer_with_batch`. This is essential to maintain an up-to-date state of the build side for future join operations.
    ///
    /// Next, it attempts to identify a joinable batch from the probe side using `identify_joinable_probe_batch`. If a joinable
    /// batch is found, it proceeds to join this batch with the current state of the build buffer using `join_using_joinable_probe_batch`.
    ///
    /// The function returns `StreamJoinStateResult::Continue` if no joinable batch is identified, signaling the caller to continue pulling
    /// more batches.
    ///
    /// If a join is performed, the function checks if the result is a non-empty batch. If so, it returns `StreamJoinStateResult::Ready(Some(batch))`,
    /// wrapping the resultant batch. If the join result is empty, it returns `StreamJoinStateResult::Continue`, signaling that the join was
    /// successful but did not produce any output rows.
    ///
    /// This function ensures that the join operation is consistently applied to batches from the left stream as they arrive,
    /// facilitating real-time processing of streaming data in a join operation.
    fn handle_left_stream_batch_pull(
        &mut self,
        batch: RecordBatch,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        // Incorporating the new batch from the build side into the current state.
        self.update_build_buffer_with_batch(batch)?;

        // Attempting to identify a batch from the probe side that is ready to be joined.
        let result = if let Some(batch) = self.identify_joinable_probe_batch()? {
            self.join_using_joinable_probe_batch(batch)?
        } else {
            return Ok(StreamJoinStateResult::Continue);
        };

        // Producing the final join result based on the outcome of the previous operations.
        if result.is_some() {
            Ok(StreamJoinStateResult::Ready(result))
        } else {
            Ok(StreamJoinStateResult::Continue)
        }
    }

    /// Helper method to handle data pulled from the right side stream.
    ///
    /// # Arguments
    ///
    /// * `batch` - The `RecordBatch` that has been pulled from the right side stream.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Returns a `Result` indicating the success or failure of the operation.
    fn handle_right_stream_batch_pull(
        &mut self,
        batch: RecordBatch,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        let schema = self.get_mutable_probe_buffer().current_batch.schema();
        self.get_mutable_probe_buffer().current_batch = concat_batches(
            &schema,
            vec![&self.get_mutable_probe_buffer().current_batch, &batch],
        )?;
        Ok(StreamJoinStateResult::Continue)
    }

    /// Handles the scenario when the left (build) stream is exhausted, and there is more batchs from the right (probe) stream available.
    ///
    /// This function plays a crucial role in ensuring that all remaining joinable data is processed correctly once we know
    /// there will be no more data coming from the left stream. It takes a batch from the right stream as an input
    /// and performs the necessary join operations to produce the results.
    ///
    /// # Arguments
    ///
    /// * `right_batch`: The `RecordBatch` from the right (probe) stream to be processed.
    ///
    /// # Returns
    ///
    /// * `Result<StreamJoinStateResult<Option<RecordBatch>>>`: A result wrapping the state result. This can be:
    ///     - `StreamJoinStateResult::Ready(Some(batch))`: If there is a resultant batch from the join operation.
    ///     - `StreamJoinStateResult::Continue`: If there is no resultant batch, but the operation was successful, indicating
    ///       continue on the stream.
    ///     - An `Err` variant: If any error occurs during the process.
    ///
    /// # Description
    ///
    /// This function checks if there are any remaining rows in the probe side buffer. If the buffer is empty, it directly
    /// proceeds to join the `right_batch`. If the buffer is not empty, it implies that there are some leftover rows from
    /// previous batches that need to be joined with the build side before proceeding. In this case, it concatenates the
    /// leftover rows in the buffer with the `right_batch` and then performs the join operation.
    ///
    /// This ensures that all rows are accounted for in the join operation, providing accurate and complete results even
    /// when the build side is exhausted. The buffer is then emptied as it is no longer needed, transforming the scenario
    /// into a typical hash join operation for any subsequent batches.
    fn handle_left_stream_end(
        &mut self,
        right_batch: RecordBatch,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        let batch = if self.get_mutable_probe_buffer().current_batch.num_rows() == 0 {
            right_batch
        } else {
            // Since there will be no more new data from the build side, we can safely join each
            // incoming probe batch. It becomes similar to a typical HashJoinStream at this point.
            // Therefore, we empty the buffer as it is no longer needed.
            let schema = self.get_mutable_probe_buffer().current_batch.schema();
            let buffer = mem::replace(
                &mut self.get_mutable_probe_buffer().current_batch,
                RecordBatch::new_empty(schema.clone()),
            );
            concat_batches(&schema, [&buffer, &right_batch])?
        };
        let result = self.join_using_joinable_probe_batch(batch)?;
        if result.is_some() {
            Ok(StreamJoinStateResult::Ready(result))
        } else {
            Ok(StreamJoinStateResult::Continue)
        }
    }

    /// Config method for minimum probe row count to increase the vectorized operations.
    ///
    /// # Returns
    ///
    /// * `usize` - Number of the minimum probe row to join.
    fn minimum_probe_row_count(&self) -> usize;
}

/// This function helps finding non-matching rows of probe buffer with the incoming build stream,
/// so we can join these rows with the current build buffer and then delete it.
/// It takes in a batch of rows from the build side of the join, a buffer for the probe side,
/// as well as various expressions and parameters required for filtering and pruning.
///
/// # Arguments
///
/// * `build_buffer_batch`: A `RecordBatch` representing a batch of rows from the build side of the join.
/// * `probe_buffer`: A mutable reference to a `ProbeBuffer` which holds the state and data of the probe side.
/// * `join_filter`: A reference to a `JoinFilter`, containing the filtering criteria for the join.
/// * `interval_graph`: A mutable reference to an `ExprIntervalGraph`, used for optimizing filter expressions.
/// * `build_filter_exprs`: A mutable slice of `SortedFilterExpr` for the build side, used for filtering and sorting.
/// * `probe_filter_exprs`: A mutable slice of `SortedFilterExpr` for the probe side, used for filtering and sorting.
/// * `minimum_probe_row_count`: The minimum number of rows that should be present in the probe side buffer for the join to be performed.
///
/// # Returns
///
/// This function returns a `Result` wrapping an `Option<RecordBatch>`.
/// * If the number of pruned rows on the probe side is less than `minimum_probe_row_count`, it returns `Ok(None)`, indicating that the join should not proceed yet.
/// * If the pruning is successful and there are enough rows to perform the join, it returns `Ok(Some(RecordBatch))` with the pruned rows.
/// * If there is any error during the process, it returns an `Err`.
///
/// # Description
///
/// The function first calculates the length of the probe side that can be pruned based on the
/// join conditions, filter expressions, and the current state of the build and probe buffers.
/// This is done using the `calculate_side_prune_length_helper` function. If the number of prunable
/// rows is less than `minimum_probe_row_count`, the function returns early with `Ok(None)`,
/// indicating that the join should be delayed.
///
/// If there are enough prunable rows, it slices the current probe batch to separate the joinable
/// rows from the rest. The joinable rows are returned, and the remaining rows are kept in the
/// probe buffer for future processing.
///
/// This helps in reducing the amount of data that needs to be processed in later stages of the
/// join, potentially leading to performance improvements.
pub fn joinable_probe_batch_helper(
    build_buffer_batch: &RecordBatch,
    probe_buffer: &mut ProbeBuffer,
    join_filter: &JoinFilter,
    interval_graph: &mut ExprIntervalGraph,
    build_filter_exprs: &mut [SortedFilterExpr],
    probe_filter_exprs: &mut [SortedFilterExpr],
    minimum_probe_row_count: usize,
) -> Result<Option<RecordBatch>> {
    // Calculate the equality results
    let probe_prune_length = calculate_side_prune_length_helper(
        join_filter,
        interval_graph,
        build_buffer_batch,
        &probe_buffer.current_batch,
        probe_filter_exprs,
        build_filter_exprs,
        JoinSide::Right,
    )?;

    if probe_prune_length < minimum_probe_row_count {
        return Ok(None);
    }
    let joinable_probe_batch = probe_buffer.current_batch.slice(0, probe_prune_length);
    probe_buffer.current_batch = probe_buffer.current_batch.slice(
        probe_prune_length,
        probe_buffer.current_batch.num_rows() - probe_prune_length,
    );
    Ok(Some(joinable_probe_batch))
}

/// Determines if the given batch is suitable for interval calculations based on the join
/// filter and sorted filter expressions.
///
/// The function evaluates the latest row of the batch for each sorted filter expression.
/// It is considered suitable if the evaluated value for all sorted filter expressions are non-null.
/// Empty batches are deemed unsuitable by default.
///
/// # Arguments
/// * `filter`: The `JoinFilter` used to determine the suitability of the batch.
/// * `probe_sorted_filter_exprs`: A slice of sorted filter expressions used to evaluate the suitability of the batch.
/// * `batch`: The `RecordBatch` to evaluate.
/// * `build_side`: The side of the join operation (either `JoinSide::Left` or `JoinSide::Right`).
///
/// # Returns
/// * A `Result` containing a boolean value. Returns `true` if the batch is suitable for interval calculation, `false` otherwise.
///
pub fn is_batch_suitable_interval_calculation(
    filter: &JoinFilter,
    probe_sorted_filter_exprs: &[SortedFilterExpr],
    batch: &RecordBatch,
    build_side: JoinSide,
) -> Result<bool> {
    // Return false if the batch is empty:
    if batch.num_rows() == 0 {
        return Ok(false);
    }

    let intermediate_batch = get_filter_representation_of_join_side(
        filter.schema(),
        &batch.slice(batch.num_rows() - 1, 1),
        filter.column_indices(),
        build_side,
    )?;

    let result = probe_sorted_filter_exprs
        .iter()
        .map(|sorted_filter_expr| {
            let expr = sorted_filter_expr.intermediate_batch_filter_expr();
            let array_ref = expr
                .evaluate(&intermediate_batch)?
                .into_array(intermediate_batch.num_rows())?;
            // Calculate the latest value of the sorted filter expression:
            let latest_value = ScalarValue::try_from_array(&array_ref, 0);
            // Return true if the latest value is not null:
            latest_value.map(|s| !s.is_null())
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(result.into_iter().all(|b| b))
}

/// Calculates the necessary build-side range for join pruning.
///
/// Given a join filter, build inner buffer, and the current state of the expression graph,
/// this function computes the interval range for the build side filter expression and then
/// updates the expression graph with the calculated interval range. This aids in optimizing
/// the join operation by pruning unnecessary rows from the build side and fetching just enough
/// batch.
///
/// # Arguments
/// * `filter`: The join filter which dictates the join condition.
/// * `build_inner_buffer`: The record batch representing the build side of the join.
/// * `graph`: The current state of the expression interval graph to be updated.
/// * `build_sorted_filter_exprs`: Sorted filter expressions related to the build side.
/// * `probe_sorted_filter_exprs`: Sorted filter expressions related to the probe side.
/// * `probe_batch`: The probe record batch.
///
/// # Returns
/// * A vector of tuples containing the physical sort expression and its associated interval
///   for the build side. These tuples represent the range in which join pruning can occur
///   for each expression.
pub fn calculate_the_necessary_build_side_range_helper(
    filter: &JoinFilter,
    graph: &mut ExprIntervalGraph,
    build_sorted_filter_exprs: &mut [SortedFilterExpr],
    probe_sorted_filter_exprs: &mut [SortedFilterExpr],
    probe_batch: &RecordBatch,
) -> Result<Vec<(PhysicalSortExpr, Interval)>> {
    let build_side = JoinSide::Left;
    // Calculate the interval for the build side filter expression (if present):
    update_filter_expr_bounds(
        filter,
        build_sorted_filter_exprs,
        probe_batch,
        probe_sorted_filter_exprs,
        build_side.negate(),
    )?;

    let mut filter_intervals = build_sorted_filter_exprs
        .iter()
        .chain(probe_sorted_filter_exprs.iter())
        .map(|sorted_filter_expr| {
            (
                sorted_filter_expr.node_index(),
                sorted_filter_expr.interval().clone(),
            )
        })
        .collect::<Vec<_>>();

    // Update the physical expression graph using the join filter intervals:
    graph.update_ranges(&mut filter_intervals, Interval::CERTAINLY_TRUE)?;

    let intermediate_schema = get_filter_representation_schema_of_build_side(
        filter.schema(),
        filter.column_indices(),
        build_side,
    )?;

    // Filter expressions that can shrink.
    let shrunk_exprs = graph.get_deepest_pruning_exprs()?;
    // Get only build side filter expressions
    get_build_side_pruned_exprs(shrunk_exprs, intermediate_schema, filter, build_side)
}

/// Checks if the sliding window condition is met for the join operation.
///
/// This function evaluates the incoming build batch against a set of intervals
/// to determine whether the sliding window condition has been satisfied. It assesses
/// that the current window has captured all the relevant rows for the join.
///
/// # Arguments
/// * `filter`: The join filter defining the join condition.
/// * `incoming_build_batch`: The incoming record batch from the build side.
/// * `intervals`: A set of intervals representing the build side's boundaries
///   against which the incoming batch is evaluated.
///
/// # Returns
/// * A boolean value indicating if the sliding window condition is met:
///   * `true` if all rows necessary from the build side for this window have been processed.
///   * `false` otherwise.
pub fn check_if_sliding_window_condition_is_met(
    filter: &JoinFilter,
    incoming_build_batch: &RecordBatch,
    intervals: &[(PhysicalSortExpr, Interval)], // interval in the build side against which we are checking
) -> Result<bool> {
    let latest_build_intermediate_batch = get_filter_representation_of_join_side(
        filter.schema(),
        &incoming_build_batch.slice(incoming_build_batch.num_rows() - 1, 1),
        filter.column_indices(),
        JoinSide::Left,
    )?;

    let results: Vec<bool> = intervals
        .iter()
        .map(|(sorted_shrunk_expr, interval)| {
            let array = sorted_shrunk_expr
                .expr
                .clone()
                .evaluate(&latest_build_intermediate_batch)?
                .into_array(1)?;
            let latest_value = ScalarValue::try_from_array(&array, 0)?;
            if latest_value.is_null() {
                return Ok(false);
            }
            Ok(if sorted_shrunk_expr.options.descending {
                // Data is sorted in descending order, so check if latest value is less
                // than the lower bound of the interval. If it is, we must have processed
                // all rows that are needed from the build side for this window.
                &latest_value < interval.lower()
            } else {
                // Data is sorted in ascending order, so check if latest value is greater
                // than the upper bound of the interval. If it is, we must have processed
                // all rows that are needed from the build side for this window.
                &latest_value > interval.upper()
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(results.iter().all(|e| *e))
}

/// Constructs a single `RecordBatch` from a vector of `RecordBatch`es.
///
/// If there's only one batch in the vector, it's directly returned. Otherwise,
/// all the batches are concatenated to produce a single `RecordBatch`.
///
/// # Arguments
/// * `batches`: A vector of `RecordBatch`es to be combined into a single batch.
///
/// # Returns
/// * A `Result` containing a single `RecordBatch` or an error if the concatenation fails.
///
pub fn get_probe_batch(mut batches: Vec<RecordBatch>) -> Result<RecordBatch> {
    let probe_batch = if batches.len() == 1 {
        batches.remove(0)
    } else {
        let schema = batches[0].schema();
        concat_batches(&schema, &batches)?
    };
    Ok(probe_batch)
}

/// Appends probe indices in order by considering the given build indices.
///
/// This function constructs new build and probe indices by iterating through
/// the provided indices, and appends any missing values between previous and
/// current probe index with a corresponding null build index. It handles various
/// edge cases and returns an error if either index is `None`.
///
/// # Parameters
/// - `build_indices`: `PrimitiveArray` of `UInt64Type` containing build indices.
/// - `probe_indices`: `PrimitiveArray` of `UInt32Type` containing probe indices.
/// - `count_probe_batch`: The number of elements in the probe batch, used for
///   filling in any remaining indices.
///
/// # Returns
/// A `Result` containing a tuple of two arrays:
/// - A `PrimitiveArray` of `UInt64Type` with the newly constructed build indices.
/// - A `PrimitiveArray` of `UInt32Type` with the newly constructed probe indices.
///
/// # Errors
/// Returns an error if there is a failure in calculating probe indices.
fn append_probe_indices_in_order(
    build_indices: PrimitiveArray<UInt64Type>,
    probe_indices: PrimitiveArray<UInt32Type>,
    count_probe_batch: u32,
) -> datafusion_common::Result<(PrimitiveArray<UInt64Type>, PrimitiveArray<UInt32Type>)> {
    // Builders for new indices:
    let mut new_build_indices = UInt64Builder::new();
    let mut new_probe_indices = UInt32Builder::new();

    // Set previous index as zero for the initial loop.
    let mut prev_index = 0;

    // Zip the two iterators.
    for (maybe_build_index, maybe_probe_index) in
        build_indices.iter().zip(probe_indices.iter())
    {
        // Unwrap index options.
        let (build_index, probe_index) = match (maybe_build_index, maybe_probe_index) {
            (Some(bi), Some(pi)) => (bi, pi),
            // If either index is None, return an error.
            _ => {
                return Err(DataFusionError::Internal(
                    "Error on probe indices calculation".to_owned(),
                ))
            }
        };

        // Append values between previous and current left index with null right index.
        for val in prev_index..probe_index {
            new_probe_indices.append_value(val);
            new_build_indices.append_null();
        }

        // Append current indices.
        new_probe_indices.append_value(probe_index);
        new_build_indices.append_value(build_index);

        // Set current left index as previous for the next loop.
        prev_index = probe_index + 1;
    }

    // Append remaining left indices after the last valid left index with null right index.
    for val in prev_index..count_probe_batch {
        new_probe_indices.append_value(val);
        new_build_indices.append_null();
    }

    // Build arrays and return.
    Ok((new_build_indices.finish(), new_probe_indices.finish()))
}

/// Adjusts indices of the probe side according to the specified join type.
///
/// The main purpose of this function is to align the indices for different types
/// of joins, including `Inner`, `Left`, `Right`, `Full`, `RightSemi`, `RightAnti`,
/// `LeftAnti` and `LeftSemi`.
///
/// # Parameters
/// - `build_indices`: The `UInt64Array` containing build indices.
/// - `probe_indices`: The `UInt32Array` containing probe indices.
/// - `count_probe_batch`: The number of elements in the probe batch.
/// - `join_type`: The type of join in question.
///
/// # Returns
/// A `Result` containing a tuple of two arrays:
/// - A `UInt64Array` with the adjusted build indices.
/// - A `UInt32Array` with the adjusted probe indices.
///
/// # Errors
/// Returns an error if there is a failure in processing the indices according
/// to the given join type.
pub(crate) fn adjust_probe_side_indices_by_join_type(
    build_indices: UInt64Array,
    probe_indices: UInt32Array,
    count_probe_batch: usize,
    join_type: JoinType,
) -> Result<(UInt64Array, UInt32Array)> {
    match join_type {
        JoinType::Inner | JoinType::Left => {
            // Unmatched rows for the left join will be produced in the pruning phase.
            Ok((build_indices, probe_indices))
        }
        JoinType::Right => {
            // We use an order preserving index calculation algorithm, since it is possible in theory.
            append_probe_indices_in_order(
                build_indices,
                probe_indices,
                count_probe_batch as u32,
            )
        }
        JoinType::Full => {
            // Unmatched probe rows will be produced in this batch. Since we do
            // not preserve the order, we do not need to iterate through the left
            // indices. This is why we split the full join.

            let right_unmatched_indices =
                get_anti_indices(count_probe_batch, &probe_indices);
            // Combine the matched and unmatched right result together:
            Ok(append_right_indices(
                build_indices,
                probe_indices,
                right_unmatched_indices,
            ))
        }
        JoinType::RightSemi => {
            // We need to remove duplicated records in the probe side:
            let probe_indices = get_semi_indices(count_probe_batch, &probe_indices);
            Ok((build_indices, probe_indices))
        }
        JoinType::RightAnti => {
            // We need to remove duplicated records in the probe side.
            // For this purpose, get anti indices for the probe side:
            let probe_indices = get_anti_indices(count_probe_batch, &probe_indices);
            Ok((build_indices, probe_indices))
        }
        JoinType::LeftAnti | JoinType::LeftSemi => {
            // Matched or unmatched build side rows will be produced in the
            // pruning phase of the build side.
            // When we visit the right batch, we can output the matched left
            // row and don't need to wait for the pruning phase.
            Ok((
                UInt64Array::from_iter_values(vec![]),
                UInt32Array::from_iter_values(vec![]),
            ))
        }
    }
}

/// Calculates the build side outer indices based on the specified join type.
///
/// This function calculates the build side outer indices for specific join types,
/// including `Left`, `LeftAnti`, `LeftSemi` and `Full`. It computes unmatched indices
/// for pruning and constructs corresponding probe indices with null values.
///
/// # Parameters
/// - `prune_length`: Length for pruning calculations.
/// - `visited_rows`: A `HashSet` containing visited row indices.
/// - `deleted_offset`: Offset for deleted indices.
/// - `join_type`: The type of join in question.
///
/// # Returns
/// A `Result` containing a tuple of two arrays:
/// - A `PrimitiveArray` of generic type `L` with build indices.
/// - A `PrimitiveArray` of generic type `R` with probe indices containing null values.
///
/// # Errors
/// No explicit error handling in the function, but it may return errors coming from
/// underlying calls. The case of other join types is not considered, and the function
/// will return an `DatafusionError::Internal` if called with such a join type.
///
/// # Type Parameters
/// - `L`: The Arrow primitive type for build indices.
/// - `R`: The Arrow primitive type for probe indices.
pub fn calculate_build_outer_indices_by_join_type<
    L: ArrowPrimitiveType,
    R: ArrowPrimitiveType,
>(
    prune_length: usize,
    visited_rows: &HashSet<usize>,
    deleted_offset: usize,
    join_type: JoinType,
) -> Result<(PrimitiveArray<L>, PrimitiveArray<R>)>
where
    NativeAdapter<L>: From<<L as ArrowPrimitiveType>::Native>,
{
    let result = match join_type {
        JoinType::Left | JoinType::LeftAnti | JoinType::Full => {
            // Calculate anti indices for pruning:
            let build_unmatched_indices =
                get_pruning_anti_indices(prune_length, deleted_offset, visited_rows);
            // Prepare probe indices with null values corresponding to build side
            // unmatched indices:
            let mut builder =
                PrimitiveBuilder::<R>::with_capacity(build_unmatched_indices.len());
            builder.append_nulls(build_unmatched_indices.len());
            let probe_indices = builder.finish();
            (build_unmatched_indices, probe_indices)
        }
        JoinType::LeftSemi => {
            // Calculate semi indices for pruning:
            let build_unmatched_indices =
                get_pruning_semi_indices(prune_length, deleted_offset, visited_rows);
            // Prepare probe indices with null values corresponding to build side
            // unmatched indices:
            let mut builder =
                PrimitiveBuilder::<R>::with_capacity(build_unmatched_indices.len());
            builder.append_nulls(build_unmatched_indices.len());
            let probe_indices = builder.finish();
            (build_unmatched_indices, probe_indices)
        }
        // Return an internal error if an unsupported join type is given.
        _ => {
            return Err(DataFusionError::Internal(
                "Given join type is not supported".to_owned(),
            ))
        }
    };
    Ok(result)
}

/// Represents the various states of a sliding window join stream.
///
/// This `enum` encapsulates the different states that a join stream might be
/// in throughout its execution. Depending on its current state, the join
/// operation will perform different actions such as pulling data from the build
/// side or the probe side, or performing the join itself.
#[derive(Clone)]
pub enum LazyJoinStreamState {
    /// The action is to pull data from the probe side (right stream).
    /// This state continues to pull data until the probe batches are suitable
    /// for interval calculations, or the probe stream is exhausted.
    PullProbe,
    /// The action is to pull data from the build side (left stream) within a
    /// given interval.
    /// This state continues to pull data until a suitable range of batches is
    /// found, or the build stream is exhausted.
    PullBuild {
        interval: Vec<(PhysicalSortExpr, Interval)>,
    },
    /// The probe side is completely processed. In this state, the build side
    /// will be ready and its results will be processed until the build stream
    /// is also exhausted.
    ProbeExhausted,
    /// Both the build and probe sides have been completely processed.
    /// If `final_result` is `false`, a final result may still be produced from
    /// the build side. Otherwise, the join operation is complete.
    BothExhausted { final_result: bool },
    /// The join operation is actively processing data from both sides to produce
    /// the result. In this state, equal and anti join results are calculated and
    /// combined into a single batch, and the state is updated to `PullProbe` for
    /// the next iteration.
    Join,
}

/// Updates the filter expression bounds for both build and probe sides.
///
/// This function evaluates the build/probe-side sorted filter expressions to
/// determine feasible interval bounds. It then sets these intervals within
/// the expressions. The function sets a null interval for the build side and
/// calculates the actual interval for the probe side based on the sort options.
pub(crate) fn update_filter_expr_bounds(
    filter: &JoinFilter,
    build_sorted_filter_exprs: &mut [SortedFilterExpr],
    probe_batch: &RecordBatch,
    probe_sorted_filter_exprs: &mut [SortedFilterExpr],
    probe_side: JoinSide,
) -> Result<()> {
    let schema = filter.schema();
    // Evaluate the build side order expression to get datatype:
    let build_order_datatype = build_sorted_filter_exprs[0]
        .intermediate_batch_filter_expr()
        .data_type(schema)?;

    // Create a null interval using the null scalar value:
    let unbounded_interval = Interval::make_unbounded(&build_order_datatype)?;

    build_sorted_filter_exprs
        .iter_mut()
        .for_each(|sorted_filter_expr| {
            sorted_filter_expr.set_interval(unbounded_interval.clone());
        });

    let first_probe_intermediate_batch = get_filter_representation_of_join_side(
        schema,
        &probe_batch.slice(0, 1),
        filter.column_indices(),
        probe_side,
    )?;

    let last_probe_intermediate_batch = get_filter_representation_of_join_side(
        schema,
        &probe_batch.slice(probe_batch.num_rows() - 1, 1),
        filter.column_indices(),
        probe_side,
    )?;

    probe_sorted_filter_exprs
        .iter_mut()
        .try_for_each(|sorted_filter_expr| {
            let expr = sorted_filter_expr.intermediate_batch_filter_expr();
            // Evaluate the probe side filter expression with the first batch
            // and convert the result to an array:
            let first_array = expr
                .evaluate(&first_probe_intermediate_batch)?
                .into_array(first_probe_intermediate_batch.num_rows())?;

            // Evaluate the probe side filter expression with the last batch
            // and convert the result to an array:
            let last_array = expr
                .evaluate(&last_probe_intermediate_batch)?
                .into_array(last_probe_intermediate_batch.num_rows())?;
            // Extract the left and right values from the array:
            let left_value = ScalarValue::try_from_array(&first_array, 0)?;
            let right_value = ScalarValue::try_from_array(&last_array, 0)?;
            // Determine the interval bounds based on sort options:
            let interval = if sorted_filter_expr.order().descending {
                Interval::try_new(right_value, left_value)?
            } else {
                Interval::try_new(left_value, right_value)?
            };
            // Set the calculated interval for the sorted filter expression:
            sorted_filter_expr.set_interval(interval);
            Ok(())
        })
}

#[cfg(test)]
mod tests {
    use crate::joins::sliding_window_join_utils::append_probe_indices_in_order;

    use arrow_array::{UInt32Array, UInt64Array};

    #[test]
    fn test_append_left_indices_in_order() {
        let left_indices = UInt32Array::from(vec![Some(1), Some(1), Some(2), Some(4)]);
        let right_indices =
            UInt64Array::from(vec![Some(10), Some(20), Some(30), Some(40)]);
        let left_len = 7;

        let (new_right_indices, new_left_indices) =
            append_probe_indices_in_order(right_indices, left_indices, left_len).unwrap();

        // Expected results
        let expected_left_indices = UInt32Array::from(vec![
            Some(0),
            Some(1),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
        ]);
        let expected_right_indices = UInt64Array::from(vec![
            None,
            Some(10),
            Some(20),
            Some(30),
            None,
            Some(40),
            None,
            None,
        ]);

        assert_eq!(new_left_indices, expected_left_indices);
        assert_eq!(new_right_indices, expected_right_indices);
    }
}

/// Calculate the OutputPartitioning for Partitioned Join
pub fn partitioned_join_output_partitioning(
    join_type: JoinType,
    left_partitioning: Partitioning,
    right_partitioning: Partitioning,
    left_columns_len: usize,
) -> Partitioning {
    match join_type {
        JoinType::Inner | JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti => {
            left_partitioning
        }
        JoinType::RightSemi | JoinType::RightAnti => right_partitioning,
        JoinType::Right => {
            utils::adjust_right_output_partitioning(right_partitioning, left_columns_len)
        }
        JoinType::Full => {
            Partitioning::UnknownPartitioning(right_partitioning.partition_count())
        }
    }
}
