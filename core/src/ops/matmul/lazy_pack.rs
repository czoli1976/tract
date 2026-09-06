//! Pack an activation panel by panel inside the matmul, rather than
//! materialising the whole packed tensor in a node of its own.
//!
//! `OptMatMulPack` writes a packed copy of its input and the matmul reads it
//! straight back. Where the input is a constant weight that costs nothing --
//! const propagation folds the pack away. Where it is an activation it is paid
//! every run, and where the matmul walks each panel once, which is what a
//! huge-N skinny-K convolution GEMM does, the packed buffer has no reuse to
//! amortise the round trip against. PP-OCRv6 detection spends 28 ms/i there on
//! an i7-12700K, about 15% of the run, and it did not move when the rest of the
//! network got faster.
//!
//! `PackedFormat::pack_t` already packs an arbitrary `mn` range, so one panel is
//! that range narrowed to `r`, written into the scratch buffer the kernel
//! already carries for `LazyIm2col`.

use crate::internal::*;
use crate::ops::matmul::pack::DynPackedExoticFact;
use std::fmt::{Debug, Display};
use tract_linalg::mmm::{MMMInputFormat, MMMInputValue, PackedExoticFact, PackedMatrixStorage};
use tract_linalg::pack::PackedFormat;

/// The packer, worn as a format of its own so the matmul can tell a lazily
/// gathered operand from a materialised one. `FusedSpec::prefer_col_outer`
/// orders the loops from exactly that distinction: with A lazy and B eager it
/// puts A outer, so `AddMatMulTemp`'s panel cache gathers each A panel once
/// rather than once per B pass. Handing back the inner `PackedFormat` here
/// would read as eager and lose that.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct LazyPackFormat(pub PackedFormat);

impl Display for LazyPackFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Lazy{}", self.0)
    }
}

impl MMMInputFormat for LazyPackFormat {
    fn r(&self) -> usize {
        self.0.r
    }
    fn precursor(&self) -> tract_linalg::WeightType {
        self.0.precursor()
    }
    fn k_alignment(&self) -> usize {
        self.0.k_alignment()
    }
    fn mem_size(&self, k: TDim, mn: TDim) -> TDim {
        // Report the operand as if it were materialised, as LazyIm2col does.
        // Only one panel is ever live, so this overstates -- but mem_size feeds
        // the const-folding budget and the memory profile, and overstating
        // declines a fold that understating would wrongly allow.
        k * mn * self.0.dt.size_of()
    }
    fn prepare_tensor(&self, _t: &Tensor, _k: usize, _mn: usize) -> TractResult<Tensor> {
        bail!("Unexpected call to prepare_tensor on LazyMatMulPack")
    }
    fn prepare_one_view(
        &self,
        _t: &TensorView,
        _k: usize,
        _mn: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        bail!("Unexpected call to prepare_one_view on LazyMatMulPack")
    }
    fn extract_at_mn_f16(
        &self,
        _d: &tract_linalg::mmm::EagerPackedInput,
        _mn: usize,
        _s: &mut [f16],
    ) -> TractResult<()> {
        unimplemented!()
    }
    fn extract_at_mn_f32(
        &self,
        _d: &tract_linalg::mmm::EagerPackedInput,
        _mn: usize,
        _s: &mut [f32],
    ) -> TractResult<()> {
        unimplemented!()
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct LazyMatMulPack {
    pub(crate) packer: PackedFormat,
    pub(crate) k_axis: usize,
    pub(crate) mn_axis: usize,
}

impl Op for LazyMatMulPack {
    fn name(&self) -> StaticName {
        "LazyMatMulPack".into()
    }

    op_as_typed_op!();
}

impl EvalOp for LazyMatMulPack {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs.remove(0);
        let dt = input.datum_type();
        let k = input.shape()[self.k_axis];
        let mn = input.shape()[self.mn_axis];
        let output_shape: TVec<usize> = self.output_shape(input.shape());
        ensure!(
            output_shape.iter().all(|d| *d == 1),
            "LazyMatMulPack is wired only for a single packed value"
        );
        let value: Box<dyn MMMInputValue> = Box::new(LazyPackedInput {
            tensor: input.clone(),
            fact: PackedExoticFact {
                format: Box::new(LazyPackFormat(self.packer.clone())),
                mn: mn.to_dim(),
                k,
            },
            packer: self.packer.clone(),
            lazy_format: LazyPackFormat(self.packer.clone()),
            k_axis: self.k_axis,
            mn_axis: self.mn_axis,
            mn,
        });
        let stores = PackedMatrixStorage::new_batched(&output_shape, tvec![value]).into_tensor(dt);
        Ok(tvec!(stores.into_tvalue()))
    }
}

impl LazyMatMulPack {
    pub fn output_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let mut shape: TVec<D> = input.into();
        shape.remove(self.mn_axis.max(self.k_axis));
        shape.remove(self.mn_axis.min(self.k_axis));
        shape
    }
}

impl TypedOp for LazyMatMulPack {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let k = inputs[0].shape[self.k_axis].clone();
        let mn = inputs[0].shape[self.mn_axis].clone();
        let exotic_fact = DynPackedExoticFact {
            k,
            mn,
            packers: vec![Box::new(LazyPackFormat(self.packer.clone())) as Box<dyn MMMInputFormat>],
        };
        Ok(tvec!(
            inputs[0]
                .datum_type
                .fact(self.output_shape(&inputs[0].shape))
                .with_exotic_fact(exotic_fact)
        ))
    }

    as_op!();
}

/// One activation, packed a panel at a time on demand.
#[derive(Clone, Debug)]
struct LazyPackedInput {
    tensor: TValue,
    fact: PackedExoticFact,
    packer: PackedFormat,
    lazy_format: LazyPackFormat,
    k_axis: usize,
    mn_axis: usize,
    mn: usize,
}

impl Display for LazyPackedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "LazyPacked({})", self.packer)
    }
}

impl std::hash::Hash for LazyPackedInput {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.fact.hash(state);
    }
}

impl PartialEq for LazyPackedInput {
    fn eq(&self, other: &Self) -> bool {
        self.fact == other.fact && self.tensor == other.tensor
    }
}
impl Eq for LazyPackedInput {}

unsafe impl Send for LazyPackedInput {}
unsafe impl Sync for LazyPackedInput {}

impl LazyPackedInput {
    fn do_panel<T: Datum + Copy>(&self, i: usize, buffer: Option<*mut u8>) -> *const u8 {
        let r = self.packer.r;
        let mn_start = i * r;
        let mn_end = (mn_start + r).min(self.mn);
        let k = self.fact.k;
        let strides = self.tensor.strides();
        let packed = buffer.unwrap() as *mut T;
        unsafe {
            self.packer.pack_t::<T>(
                packed,
                self.tensor.as_ptr_unchecked::<T>(),
                self.mn,
                strides[self.k_axis],
                strides[self.mn_axis],
                0..k,
                mn_start..mn_end,
            );
        }
        packed as *const u8
    }
}

impl MMMInputValue for LazyPackedInput {
    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        Some(self.packer.single_panel_layout(self.fact.k, self.tensor.datum_type().size_of()))
    }

    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8> {
        Ok(dispatch_copy!(Self::do_panel(self.tensor.datum_type())(self, i, buffer)))
    }

    fn k(&self) -> usize {
        self.fact.k
    }

    fn mn(&self) -> usize {
        self.mn
    }

    fn format(&self) -> &dyn MMMInputFormat {
        &self.lazy_format
    }

    fn exotic_fact(&self) -> &dyn ExoticFact {
        &self.fact
    }

    fn extract_at_mn_f16(&self, _mn: usize, _slice: &mut [f16]) -> TractResult<()> {
        bail!("LazyPackedInput does not serve panel extraction")
    }

    fn extract_at_mn_f32(&self, _mn: usize, _slice: &mut [f32]) -> TractResult<()> {
        bail!("LazyPackedInput does not serve panel extraction")
    }
}
