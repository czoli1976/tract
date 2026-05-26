#[macro_use]
mod macros;

pub mod cost_model;
#[macro_use]
pub(crate) mod fuse;
pub(crate) mod input_store;
pub(crate) mod kernel;
#[macro_use]
pub(crate) mod panel_extract;
mod scratch;
mod storage;

#[cfg(test)]
#[macro_use]
pub mod tests;

use crate::multithread::Executor;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::Debug;
use tract_data::internal::*;

pub use cost_model::*;
pub use fuse::*;
pub use input_store::*;
pub use kernel::*;
pub use panel_extract::*;
pub use scratch::*;
pub use storage::*;

pub fn no_prefetch(_ptr: *const u8, _len: usize) {}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum ImplementationQuality {
    /// Individual operations are emulated by individual conversion (f16->f32->f16)
    Dreadful,
    /// Rust scalar operation (with whatever optimisation the compiler manages)
    Generic,
    /// Implicit vectorization (e.g. Rust code, some unrolled loops, explicit template instantiations for small constant)
    RustOptimized,
    /// Explicit vectorization (e.g. intrinsics vector code)
    TargetOptimized,
    /// Hand optimized (assembly)
    ManuallyOptimized,
}

impl ImplementationQuality {
    pub fn best_to_worst() -> &'static [ImplementationQuality] {
        use ImplementationQuality::*;
        &[ManuallyOptimized, TargetOptimized, RustOptimized, Generic, Dreadful]
    }

    pub fn cost(&self) -> usize {
        ImplementationQuality::best_to_worst().iter().position(|x| x == self).unwrap()
    }
}

impl PartialOrd for ImplementationQuality {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(usize::from(*self).cmp(&usize::from(*other)))
    }
}

impl From<ImplementationQuality> for usize {
    fn from(value: ImplementationQuality) -> Self {
        value.cost()
    }
}

pub trait MatMatMul: Debug + dyn_clone::DynClone + Send + Sync + std::any::Any {
    fn name(&self) -> &str;
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;

    fn quality(&self) -> ImplementationQuality;
    fn dynamic_boost(&self) -> isize;

    #[allow(clippy::type_complexity)]
    fn packings(&self) -> &[(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)];

    fn internal_type(&self) -> DatumType;

    unsafe fn c_view(&self, m_axis: Option<usize>, n_axis: Option<usize>) -> OutputStoreSpec;
    unsafe fn c_from_data_and_strides(
        &self,
        item_size: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> OutputStoreSpec;

    fn can_fuse(&self, spec: &FusedSpec) -> bool;

    fn stores(&self) -> Cow<'_, [DatumType]>;

    unsafe fn run(&self, m: usize, n: usize, non_linear: &[FusedSpec]) -> TractResult<()> {
        unsafe {
            let mut scratch = self.allocate_scratch_space();
            self.run_with_scratch_space(m, n, &mut *scratch, non_linear)
        }
    }

    unsafe fn allocate_scratch_space(&self) -> Box<dyn ScratchSpace>;
    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool;
    unsafe fn run_with_scratch_space(
        &self,
        m: usize,
        n: usize,
        scratch: &mut dyn ScratchSpace,
        non_linear: &[FusedSpec],
    ) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(MatMatMul);

impl PartialEq for Box<dyn MatMatMul> {
    fn eq(&self, other: &Box<dyn MatMatMul>) -> bool {
        self.name() == other.name()
    }
}
impl Eq for Box<dyn MatMatMul> {}

impl std::hash::Hash for Box<dyn MatMatMul> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

impl<K: MatMatMulKer> MatMatMul for K {
    fn name(&self) -> &str {
        self.name()
    }
    fn mr(&self) -> usize {
        self.mr()
    }
    fn nr(&self) -> usize {
        self.nr()
    }

    fn quality(&self) -> ImplementationQuality {
        MatMatMulKer::quality(self)
    }

    fn dynamic_boost(&self) -> isize {
        MatMatMulKer::dynamic_boost(self)
    }

    fn packings(&self) -> &[(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)] {
        self.packings()
    }

    fn internal_type(&self) -> DatumType {
        K::Acc::datum_type()
    }

    fn can_fuse(&self, spec: &FusedSpec) -> bool {
        self.can_fuse(spec)
    }

    unsafe fn c_view(&self, m_axis: Option<usize>, n_axis: Option<usize>) -> OutputStoreSpec {
        OutputStoreSpec::View { m_axis, n_axis, mr: self.mr(), nr: self.nr() }
    }

    unsafe fn c_from_data_and_strides(
        &self,
        item_size: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> OutputStoreSpec {
        OutputStoreSpec::Strides {
            row_byte_stride: row_stride * item_size as isize,
            col_byte_stride: col_stride * item_size as isize,
            mr: self.mr(),
            nr: self.nr(),
        }
    }

    fn stores(&self) -> Cow<'_, [DatumType]> {
        self.stores()
    }

    unsafe fn allocate_scratch_space(&self) -> Box<dyn ScratchSpace> {
        Box::<ScratchSpaceImpl<K::Acc>>::default()
    }

    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool {
        scratch.downcast_ref::<ScratchSpaceImpl<K::Acc>>().is_some()
    }

    unsafe fn run_with_scratch_space(
        &self,
        m: usize,
        n: usize,
        scratch: &mut dyn ScratchSpace,
        non_linear: &[FusedSpec],
    ) -> TractResult<()> {
        unsafe {
            let scratch = scratch
                .downcast_mut::<ScratchSpaceImpl<K::Acc>>()
                .context("Wrong scratch space type")?;
            scratch.prepare(self, m, n, non_linear)?;
            if n == 1 && self.nr() == 1 {
                run_with_scratch_space_vec(self, m, scratch, non_linear)
            } else {
                let (mut prefer_col, mut prefer_row) = (0, 0);
                for uop in non_linear.iter() {
                    if let Some(col) = uop.prefer_col_outer() {
                        prefer_col = col as usize;
                        prefer_row = (!col) as usize;
                    }
                }
                if prefer_col > prefer_row {
                    run_with_scratch_space_col_outer(self, m, n, scratch, non_linear)
                } else {
                    run_with_scratch_space_row_outer(self, m, n, scratch, non_linear)
                }
            }
        }
    }
}

unsafe fn run_with_scratch_space_vec<K: MatMatMulKer>(
    ker: &K,
    m: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        match crate::multithread::current_tract_executor() {
            Executor::SingleThread => scratch.run_in_tls_scope(|scratch, tls| {
                for ia in 0..m.divceil(ker.mr()) {
                    scratch.run_one_tile(ker, non_linear, tls, ia, 0)?;
                }
                TractResult::Ok(())
            }),
            #[cfg(feature = "multithread-mm")]
            Executor::MultiThread(pool) => chunked_dispatch_rayon(
                Some(&pool),
                m.divceil(ker.mr()),
                1,
                add_mat_mul_panel_pair_bytes(ker, non_linear),
                |ia_start, ia_end, _, _| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ia in ia_start..ia_end {
                            scratch.run_one_tile(ker, non_linear, tls, ia, 0)?;
                        }
                        TractResult::Ok(())
                    })
                },
            ),
            #[cfg(feature = "multithread-mm")]
            Executor::RayonGlobal => chunked_dispatch_rayon(
                None,
                m.divceil(ker.mr()),
                1,
                add_mat_mul_panel_pair_bytes(ker, non_linear),
                |ia_start, ia_end, _, _| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ia in ia_start..ia_end {
                            scratch.run_one_tile(ker, non_linear, tls, ia, 0)?;
                        }
                        TractResult::Ok(())
                    })
                },
            ),
        }
    }
}

unsafe fn run_with_scratch_space_col_outer<K: MatMatMulKer>(
    ker: &K,
    m: usize,
    n: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        match crate::multithread::current_tract_executor() {
            Executor::SingleThread => scratch.run_in_tls_scope(|scratch, tls| {
                for ib in 0..n.divceil(ker.nr()) {
                    for ia in 0..m.divceil(ker.mr()) {
                        scratch.run_one_tile(ker, non_linear, tls, ia, ib)?;
                    }
                }
                TractResult::Ok(())
            }),
            #[cfg(feature = "multithread-mm")]
            Executor::MultiThread(pool) => chunked_dispatch_rayon(
                Some(&pool),
                m.divceil(ker.mr()),
                n.divceil(ker.nr()),
                add_mat_mul_panel_pair_bytes(ker, non_linear),
                |ia_start, ia_end, ib_start, ib_end| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ib in ib_start..ib_end {
                            for ia in ia_start..ia_end {
                                scratch.run_one_tile(ker, non_linear, tls, ia, ib)?;
                            }
                        }
                        TractResult::Ok(())
                    })
                },
            ),
            #[cfg(feature = "multithread-mm")]
            Executor::RayonGlobal => chunked_dispatch_rayon(
                None,
                m.divceil(ker.mr()),
                n.divceil(ker.nr()),
                add_mat_mul_panel_pair_bytes(ker, non_linear),
                |ia_start, ia_end, ib_start, ib_end| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ib in ib_start..ib_end {
                            for ia in ia_start..ia_end {
                                scratch.run_one_tile(ker, non_linear, tls, ia, ib)?;
                            }
                        }
                        TractResult::Ok(())
                    })
                },
            ),
        }
    }
}

unsafe fn run_with_scratch_space_row_outer<K: MatMatMulKer>(
    ker: &K,
    m: usize,
    n: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        match crate::multithread::current_tract_executor() {
            Executor::SingleThread => scratch.run_in_tls_scope(|scratch, tls| {
                for ia in 0..m.divceil(ker.mr()) {
                    for ib in 0..n.divceil(ker.nr()) {
                        scratch.run_one_tile(ker, non_linear, tls, ia, ib)?;
                    }
                }
                TractResult::Ok(())
            }),
            #[cfg(feature = "multithread-mm")]
            Executor::MultiThread(pool) => chunked_dispatch_rayon(
                Some(&pool),
                m.divceil(ker.mr()),
                n.divceil(ker.nr()),
                add_mat_mul_panel_pair_bytes(ker, non_linear),
                |ia_start, ia_end, ib_start, ib_end| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ia in ia_start..ia_end {
                            for ib in ib_start..ib_end {
                                scratch.run_one_tile(ker, non_linear, tls, ia, ib)?;
                            }
                        }
                        TractResult::Ok(())
                    })
                },
            ),
            #[cfg(feature = "multithread-mm")]
            Executor::RayonGlobal => chunked_dispatch_rayon(
                None,
                m.divceil(ker.mr()),
                n.divceil(ker.nr()),
                add_mat_mul_panel_pair_bytes(ker, non_linear),
                |ia_start, ia_end, ib_start, ib_end| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ia in ia_start..ia_end {
                            for ib in ib_start..ib_end {
                                scratch.run_one_tile(ker, non_linear, tls, ia, ib)?;
                            }
                        }
                        TractResult::Ok(())
                    })
                },
            ),
        }
    }
}

/// Per-(M+N)-panel working-set bytes for the first `AddMatMul`:
/// `K * (mr * a_elt + nr * b_elt)`, i.e. the bytes of one A panel plus one B
/// panel. Used to size cache-fitting MT chunks. Returns 0 when there is no
/// `AddMatMul` (caller keeps the base chunk size).
#[cfg(feature = "multithread-mm")]
fn add_mat_mul_panel_pair_bytes<K: MatMatMulKer>(ker: &K, non_linear: &[FusedSpec]) -> usize {
    for spec in non_linear {
        if let FusedSpec::AddMatMul { a, b, .. } = spec {
            let elt = |f: &dyn MMMInputFormat| match f.precursor() {
                crate::WeightType::Plain(dt) => dt.size_of(),
                _ => std::mem::size_of::<K::Acc>(),
            };
            return a.k() * (ker.mr() * elt(a.format()) + ker.nr() * elt(b.format()));
        }
    }
    0
}

/// L2 working-set budget (bytes) used to cap MT chunk sizes so a chunk's reused
/// A+B panel block stays cache-resident. Lazily initialised from
/// `TRACT_MM_CHUNK_L2_BYTES` (0/invalid → default), and runtime-tunable via
/// [`set_mm_chunk_l2_budget`]. `0` in the atomic means "not yet initialised".
#[cfg(feature = "multithread-mm")]
static MM_CHUNK_L2_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "multithread-mm")]
fn mm_chunk_l2_budget() -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    let v = MM_CHUNK_L2_BYTES.load(Relaxed);
    if v != 0 {
        return v;
    }
    let init = std::env::var("TRACT_MM_CHUNK_L2_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(512 * 1024);
    MM_CHUNK_L2_BYTES.store(init, Relaxed);
    init
}

/// Set the L2 working-set budget (bytes) for cache-adaptive MT chunk sizing.
/// Larger values keep ggml-style chunks (less shrinking); a very large value
/// disables the cache cap. Mirrors [`crate::multithread::set_threading_panel_threshold`].
#[cfg(feature = "multithread-mm")]
pub fn set_mm_chunk_l2_budget(bytes: usize) {
    MM_CHUNK_L2_BYTES.store(bytes.max(1), std::sync::atomic::Ordering::Relaxed);
}

/// Chunk grid for the 2D dispatch.
///
/// Mirrors ggml's `mul_mat` heuristic (`ggml/src/ggml-cpu/ggml-cpu.c:1378-1398`):
///  * 16-tile panel chunks by default;
///  * 64-tile chunks when one dimension is 1 (vec / vec-mat);
///  * fallback to "block-per-thread along the longer axis" when the natural
///    grid would have fewer than `4·nth` chunks.
///
/// Cache-adaptive refinement: `panel_pair_bytes` (one A + one B panel) lets us
/// shrink the chunk below the ggml base when K is large enough that a base-sized
/// chunk's reused panel block would exceed `mm_chunk_l2_budget()`. For small/medium
/// K this is identical to ggml (clamped to the base); it never grows the chunk, so
/// load-balancing across workers is unaffected. `panel_pair_bytes == 0` disables it.
///
/// Returns `(nchunks_m, nchunks_n, dr_m, dr_n)`.
#[cfg(feature = "multithread-mm")]
fn chunk_grid(
    n_panels_m: usize,
    n_panels_n: usize,
    nth: usize,
    panel_pair_bytes: usize,
) -> (usize, usize, usize, usize) {
    let base = if n_panels_m == 1 || n_panels_n == 1 { 64 } else { 16 };
    // panel_pair_bytes == 0 (no AddMatMul) → checked_div is None → keep base.
    // Floor at 4: never collapse to tiny chunks (over-shrinking at very large K
    // both thrashes per-chunk amortization and regresses — measured).
    let floor = 4.min(base);
    let chunk_size =
        mm_chunk_l2_budget().checked_div(panel_pair_bytes).map_or(base, |c| c.clamp(floor, base));
    let mut nchunks_m = n_panels_m.div_ceil(chunk_size);
    let mut nchunks_n = n_panels_n.div_ceil(chunk_size);
    if nchunks_m * nchunks_n < 4 * nth {
        if n_panels_m > n_panels_n {
            nchunks_m = nth;
            nchunks_n = 1;
        } else {
            nchunks_m = 1;
            nchunks_n = nth;
        }
    }
    let dr_m = n_panels_m.div_ceil(nchunks_m).max(1);
    let dr_n = n_panels_n.div_ceil(nchunks_n).max(1);
    (nchunks_m, nchunks_n, dr_m, dr_n)
}

/// 2D chunked dispatcher across the (m_panels × n_panels) grid for the
/// rayon path. Replaces a 1D `into_par_iter` over a single panel axis.
/// Better-utilises threads on small/skewed shapes where one dimension has
/// fewer panels than there are workers.
///
/// The closure receives **chunk bounds** (`ia_start, ia_end, ib_start, ib_end`),
/// not per-tile indices. This lets the caller amortise per-worker setup
/// (e.g. `ScratchSpaceImpl::run_in_tls_scope`) across all tiles in the
/// chunk, mirroring #2206 for the multi-threaded path. The closure is
/// invoked exactly once per rayon work item (and once total when the
/// small-graph fallback path is taken).
///
/// `pool`:
///   * `Some(p)` with `p.current_num_threads() > 1` → scoped via `p.install`
///     (native, custom pool path).
///   * `Some(p)` with single-thread pool, or `None` → dispatched via
///     `into_par_iter` directly, which uses rayon's GLOBAL pool. This is
///     the only working path on `wasm32-unknown-unknown` via
///     `wasm_bindgen_rayon::init_thread_pool`.
#[cfg(feature = "multithread-mm")]
unsafe fn chunked_dispatch_rayon<F>(
    pool: Option<&rayon::ThreadPool>,
    n_panels_m: usize,
    n_panels_n: usize,
    panel_pair_bytes: usize,
    run_chunk: F,
) -> TractResult<()>
where
    F: Fn(usize, usize, usize, usize) -> TractResult<()> + Sync,
{
    use rayon::prelude::*;
    if n_panels_m == 0 || n_panels_n == 0 {
        return Ok(());
    }
    if n_panels_m * n_panels_n < crate::multithread::current_threading_panel_threshold() {
        // Below the threading threshold: run the whole grid as a single chunk
        // on the calling thread. Closure handles its own TLS scope.
        return run_chunk(0, n_panels_m, 0, n_panels_n);
    }
    let use_global = pool.is_none_or(|p| p.current_num_threads() <= 1);
    let body = || {
        let nth = rayon::current_num_threads();
        let (nchunks_m, nchunks_n, dr_m, dr_n) =
            chunk_grid(n_panels_m, n_panels_n, nth, panel_pair_bytes);
        let total = nchunks_m * nchunks_n;
        (0..total).into_par_iter().try_for_each(|idx| {
            let im = idx % nchunks_m;
            let in_ = idx / nchunks_m;
            let ia_start = im * dr_m;
            let ia_end = (ia_start + dr_m).min(n_panels_m);
            let ib_start = in_ * dr_n;
            let ib_end = (ib_start + dr_n).min(n_panels_n);
            run_chunk(ia_start, ia_end, ib_start, ib_end)
        })
    };
    if use_global { body() } else { pool.unwrap().install(body) }
}
