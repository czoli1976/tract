use crate::internal::*;
use crate::ops::cnn::Patch;
use crate::ops::cnn::patches::{Zone, ZoneScanner};
use crate::ops::nn::DataShape;
use num_traits::Zero;

/// Per-element post-op fused into the DepthWise inner loop after bias add.
///
/// Mirrors the FusedKerSpec scalar variants exposed by the matmul tile
/// (`linalg/src/frame/mmm/fuse.rs`). Operands are stored as `f32` for the
/// scalar variants; `f16` and other element types fall through to the
/// unfused path because the post-op walker only matches when the fused
/// kernel datum type is f32 (matching the conservative coverage of
/// `OptMatMul::fuse_binary` for hand-coded conv ops).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DepthWisePostOp {
    /// `x.max(c)` — covers ReLU at c=0 and Relu6's lower bound.
    ScalarMax(f32),
    /// `x.min(c)` — covers Relu6's upper bound at c=6.
    ScalarMin(f32),
    /// `x + c`.
    ScalarAdd(f32),
    /// `x * c`.
    ScalarMul(f32),
    /// `x - c`.
    ScalarSub(f32),
    /// `c - x` (sub flipped).
    ScalarSubF(f32),
}

// Hash and Eq treat the f32 payload via bit pattern so the op is hashable
// even though f32 isn't Hash/Eq by default.
impl std::hash::Hash for DepthWisePostOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            DepthWisePostOp::ScalarMax(c) => {
                0u8.hash(state);
                c.to_bits().hash(state)
            }
            DepthWisePostOp::ScalarMin(c) => {
                1u8.hash(state);
                c.to_bits().hash(state)
            }
            DepthWisePostOp::ScalarAdd(c) => {
                2u8.hash(state);
                c.to_bits().hash(state)
            }
            DepthWisePostOp::ScalarMul(c) => {
                3u8.hash(state);
                c.to_bits().hash(state)
            }
            DepthWisePostOp::ScalarSub(c) => {
                4u8.hash(state);
                c.to_bits().hash(state)
            }
            DepthWisePostOp::ScalarSubF(c) => {
                5u8.hash(state);
                c.to_bits().hash(state)
            }
        }
    }
}
impl Eq for DepthWisePostOp {}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DepthWise {
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
    /// Post-ops to apply after bias add, before store. When empty (the default),
    /// the inner loop is bit-identical to the original unfused path.
    pub post_ops: TVec<DepthWisePostOp>,
}

impl DepthWise {
    pub fn new(patch: Patch, input_shape: DataShape, output_shape: DataShape) -> Self {
        DepthWise { patch, input_shape, output_shape, post_ops: tvec!() }
    }

    /// Apply post_ops in order. Caller guarantees T: Datum + Copy + Float-like.
    /// Generic over T using two casts (T → f32 → T) per op so f16 inner loops
    /// still work. The empty-post_ops path is short-circuited at the call site
    /// to avoid disturbing autovectorisation when no fusion is in play.
    #[inline(always)]
    fn apply_post_ops_f32(&self, x: f32) -> f32 {
        let mut x = x;
        for op in &self.post_ops {
            x = match op {
                DepthWisePostOp::ScalarMax(c) => x.max(*c),
                DepthWisePostOp::ScalarMin(c) => x.min(*c),
                DepthWisePostOp::ScalarAdd(c) => x + *c,
                DepthWisePostOp::ScalarMul(c) => x * *c,
                DepthWisePostOp::ScalarSub(c) => x - *c,
                DepthWisePostOp::ScalarSubF(c) => *c - x,
            };
        }
        x
    }
}

impl Op for DepthWise {
    fn name(&self) -> StaticName {
        "DepthWiseConv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.patch)])
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for DepthWise {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let dt = inputs[0].datum_type();
        #[cfg(target_arch = "aarch64")]
        if dt == f16::datum_type() && tract_linalg::arm64::has_fp16() {
            return unsafe {
                eval_t_aarch64fp16::<f16>(
                    self,
                    inputs,
                    |a, b| tract_linalg::arm64::add_f16(a, b),
                    |a, b| tract_linalg::arm64::mul_f16(a, b),
                )
            };
        }
        let out = dispatch_floatlike!(Self::eval_gen(dt)(self, inputs))?;
        // Apply post-ops absorbed by fuse(). The inner loop is left bit-identical
        // to the original unfused path; post-ops are applied as a single output-
        // buffer pass after the inner loop completes. This avoids disturbing
        // autovectorisation in the hot inner loop while still saving the op
        // dispatch + tensor handoff cost of a separate `OptBinByScalar` post-pass.
        if !self.post_ops.is_empty() && dt == f32::datum_type() {
            let mut out_tensor = out.into_iter().next().unwrap().into_tensor();
            // SAFETY: dt was just confirmed to be f32 above.
            unsafe {
                let slice = out_tensor.as_slice_mut_unchecked::<f32>();
                for v in slice.iter_mut() {
                    *v = self.apply_post_ops_f32(*v);
                }
            }
            return Ok(tvec!(out_tensor.into_tvalue()));
        }
        Ok(out)
    }
}

impl DepthWise {
    fn eval_gen<T: Datum + Copy + num_traits::Zero + ndarray::LinalgScalar>(
        &self,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        unsafe { eval_t_generic::<T>(self, inputs, |a, b| a + b, |a, b| a * b) }
    }
}

impl TypedOp for DepthWise {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(inputs.len() == 3);
        anyhow::ensure!(
            self.input_shape.c() == self.output_shape.c(),
            "DepthWiseConv must have same input and output channels"
        );
        anyhow::ensure!(
            self.input_shape.c().to_dim() == inputs[2].shape.volume(),
            "DepthWiseConv data has {} channels, bias has {}",
            self.input_shape.c(),
            inputs[2].shape.len()
        );
        Ok(tvec!(inputs[0].datum_type.fact(&self.output_shape.shape)))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let [_input, kernel, _bias] = inputs else {
            bail!("Depthwise expects three inputs");
        };
        let n_output_points = self.patch.output_shape.iter().cloned().product::<usize>();
        Ok(tvec!((
            Cost::FMA(inputs[0].datum_type),
            kernel.shape.volume() * self.input_shape.n().unwrap_or(&1) * n_output_points
        )))
    }

    /// Absorb a scalar Min/Max/Add/Mul/Sub/SubF successor as a post-op.
    ///
    /// Mirrors `OptMatMul::fuse` for the BinScalar case. Only fires for f32
    /// outputs (matching the conservative scope of the post-op apply path)
    /// and only when the scalar operand is a statically-known const so the
    /// value can be baked into the `DepthWisePostOp` payload.
    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops;
        // Only one output, one successor, and not a graph output.
        rule_if!(node.outputs.len() == 1);
        rule_if!(node.outputs[0].successors.len() == 1);
        rule_if!(!model.output_outlets()?.contains(&node.id.into()));
        // Restrict to f32; matches the conservative scope of apply_post_ops_f32.
        rule_if!(self.output_shape.shape.iter().product::<usize>() > 0);
        let out_dt = model.outlet_fact(node.id.into())?.datum_type;
        rule_if!(out_dt == f32::datum_type());

        let succ = model.node(node.outputs[0].successors[0].node);

        // Try to recognize a Min/Max/Add/Mul/Sub/SubF binary successor.
        let (binop_kind, scalar_value, flipped): (tract_linalg::BinOp, f32, bool) =
            if let Some(op) = succ.op_as::<ops::binary::TypedBinOp>() {
                rule_if_some!(binop = op.0.as_linalg_binop());
                let flipped = succ.inputs[0].node == node.id;
                let other = succ.inputs[flipped as usize];
                let other_fact = model.outlet_fact(other)?;
                rule_if_some!(uniform = other_fact.uniform.as_ref());
                rule_if!(uniform.datum_type() == f32::datum_type());
                rule_if!(uniform.len() == 1);
                let v = uniform.cast_to_scalar::<f32>()?;
                (binop, v, flipped)
            } else if let Some(op) = succ.op_as::<ops::binary::OptBinByScalar>() {
                rule_if_some!(binop = op.binop.as_linalg_binop());
                let flipped = succ.inputs[0].node == node.id;
                let other = succ.inputs[flipped as usize];
                let other_fact = model.outlet_fact(other)?;
                rule_if_some!(uniform = other_fact.uniform.as_ref());
                rule_if!(uniform.datum_type() == f32::datum_type());
                rule_if!(uniform.len() == 1);
                let v = uniform.cast_to_scalar::<f32>()?;
                (binop, v, flipped)
            } else {
                return Ok(None);
            };

        let post_op = match (binop_kind, flipped) {
            (tract_linalg::BinOp::Min, _) => DepthWisePostOp::ScalarMin(scalar_value),
            (tract_linalg::BinOp::Max, _) => DepthWisePostOp::ScalarMax(scalar_value),
            (tract_linalg::BinOp::Add, _) => DepthWisePostOp::ScalarAdd(scalar_value),
            (tract_linalg::BinOp::Mul, _) => DepthWisePostOp::ScalarMul(scalar_value),
            (tract_linalg::BinOp::Sub, false) => DepthWisePostOp::ScalarSub(scalar_value),
            (tract_linalg::BinOp::Sub, true) => DepthWisePostOp::ScalarSubF(scalar_value),
            (tract_linalg::BinOp::SubF, false) => DepthWisePostOp::ScalarSubF(scalar_value),
            (tract_linalg::BinOp::SubF, true) => DepthWisePostOp::ScalarSub(scalar_value),
        };

        let mut new_op = self.clone();
        new_op.post_ops.push(post_op);
        TypedModelPatch::fuse_with_next(model, node, new_op).map(Some)
    }

    as_op!();
}

macro_rules! impl_eval {
    ($(#[$meta: meta])* $suffix: ident ) => {
        pastey::paste! {
            $(#[$meta])*
            unsafe fn [<eval_t_ $suffix>]<T: Datum + Copy + num_traits::Zero + ndarray::LinalgScalar>(
                dw: &DepthWise,
                inputs: TVec<TValue>,
                add: impl Fn(T, T) -> T + Copy + 'static,
                mul: impl Fn(T, T) -> T + Copy + 'static,
            ) -> TractResult<TVec<TValue>> {
                let (img, kernel, bias) = args_3!(inputs);
                let mut output = unsafe { Tensor::uninitialized::<T>(&dw.output_shape.shape)? };
                let iptr = img.as_ptr::<T>()?;
                let optr = output.as_ptr_mut::<T>()?;
                let k_stride_i = kernel.strides()[1];
                let n = *dw.input_shape.n().unwrap_or(&1);
                let n_stride_i = *dw.input_shape.n_stride().unwrap_or(&0) as isize;
                let n_stride_o = *dw.output_shape.n_stride().unwrap_or(&0) as isize;
                let c_stride_i = *dw.input_shape.c_stride() as isize;
                let c_stride_o = *dw.output_shape.c_stride() as isize;
                let bias = bias.as_ptr::<T>()?;
                let kptr = kernel.as_ptr::<T>()?;
                unsafe {
                    for n in 0..n as isize {
                        let iptr = iptr.offset(n_stride_i * n);
                        let optr = optr.offset(n_stride_o * n);
                        for zone in &dw.patch.zones {
                            [<process_zone_ $suffix>](
                                dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr,
                                add, mul,
                            )
                        }
                    }
                }
                Ok(tvec!(output.into_tvalue()))
            }

            #[inline(never)]
            #[allow(clippy::too_many_arguments)]
            $(#[$meta])*
            unsafe fn [<process_zone_ $suffix>]<T: Datum + Copy + Zero>(
                dw: &DepthWise,
                zone: &Zone,
                c_stride_i: isize,
                c_stride_o: isize,
                k_stride_i: isize,
                iptr: *const T,
                kptr: *const T,
                bias: *const T,
                optr: *mut T,
                add: impl Fn(T, T) -> T + Copy + 'static,
                mul: impl Fn(T, T) -> T + Copy + 'static,
                ) { unsafe {
                /*
                   if zone.values_offsets.len() == 2 {
                   self.process_zone_n::<T, 2, 4>(
                   zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr,
                   )
                   } else if zone.values_offsets.len() == 3 {
                   dw.process_zone_n::<T, 3, 4>(
                   zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr,
                   )
                   } else */
                if zone.values_offsets.len() == 4 {
                    [<process_zone_n_ $suffix>]::<T, 4, 4>(
                        dw, zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr, add, mul,
                        )
                        /*
                           } else if zone.values_offsets.len() == 5 {
                           dw.process_zone_n::<T, 5, 2>(
                           zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr,
                           )
                           } else if zone.values_offsets.len() == 9 {
                           dw.process_zone_n::<T, 9, 1>(
                           zone, c_stride_i, c_stride_o, k_stride_i, iptr, kptr, bias, optr,
                           )
                           */
                } else {
                    zone.visit_output(&dw.patch, |visitor| {
                        for c in 0..*dw.input_shape.c() as isize {
                            let iptr = iptr.offset(c_stride_i * c);
                            let optr = optr.offset(c_stride_o * c);
                            let kptr = kptr.offset(k_stride_i * c);
                            [<inner_loop_ $suffix>]::<T>(iptr, kptr, bias, optr, c, visitor, add, mul)
                        }
                    })
                }
            }}

            #[inline(never)]
            #[allow(clippy::too_many_arguments)]
            $(#[$meta])*
            unsafe fn [<process_zone_n_ $suffix>]<T: Datum + Copy + Zero, const N: usize, const UNROLL: usize>(
                dw: &DepthWise,
                zone: &Zone,
                c_stride_i: isize,
                c_stride_o: isize,
                k_stride_i: isize,
                iptr: *const T,
                kptr: *const T,
                bias: *const T,
                optr: *mut T,
                add: impl Fn(T, T) -> T,
                mul: impl Fn(T, T) -> T,
                ) { unsafe {
                let mut visitor = ZoneScanner::new(zone, &dw.patch);
                let mut ioffset = [0isize; N];
                for i in 0..N {
                    ioffset[i] = zone.values_offsets[i].1;
                }
                let mut k = [T::zero(); N];
                for c in 0..*dw.input_shape.c() as isize {
                    visitor.reset();
                    let iptr = iptr.offset(c_stride_i * c);
                    let optr = optr.offset(c_stride_o * c);
                    for n in 0..N {
                        k[n] = *kptr.offset(k_stride_i * c).add(zone.values_offsets[n].0);
                    }
                    let bias = *bias.offset(c);
                    while !visitor.done {
                        let iptr = iptr.offset(visitor.input_center_offset);
                        let optr = optr.offset(visitor.output_offset);
                        let mut i = 0isize;
                        while i + (UNROLL as isize) < visitor.inner_loop_len as isize {
                            let iptr = iptr.offset(visitor.inner_loop_input_full_stride * i);
                            let optr = optr.offset(visitor.inner_loop_output_stride * i);
                            let mut iptrs = [std::ptr::null(); UNROLL];
                            for u in 0..UNROLL {
                                iptrs[u] = iptr.offset(visitor.inner_loop_input_full_stride * u as isize);
                            }
                            let mut optrs = [std::ptr::null_mut(); UNROLL];
                            for u in 0..UNROLL {
                                optrs[u] = optr.offset(visitor.inner_loop_output_stride * u as isize);
                            }
                            let mut is = [[T::zero(); N]; UNROLL];
                            for u in 0..UNROLL {
                                for n in 0..N {
                                    is[u][n] = *iptrs[u].offset(ioffset[n]);
                                }
                            }
                            let mut ps = [[T::zero(); N]; UNROLL];
                            for u in 0..UNROLL {
                                for n in 0..N {
                                    ps[u][n] = mul(is[u][n], k[n]);
                                }
                            }
                            for u in 0..UNROLL {
                                let mut sum = bias;
                                for n in 0..N {
                                    sum = add(sum, ps[u][n]);
                                }
                                *optrs[u] = sum;
                            }
                            i += UNROLL as isize;
                        }
                        while i < visitor.inner_loop_len as isize {
                            let iptr = iptr.offset(visitor.inner_loop_input_full_stride * i);
                            let optr = optr.offset(visitor.inner_loop_output_stride * i);
                            let mut is = [T::zero(); N];
                            for n in 0..N {
                                is[n] = *iptr.offset(ioffset[n]);
                            }
                            let mut p = [T::zero(); N];
                            for n in 0..N {
                                p[n] = mul(is[n], k[n]);
                            }
                            let mut sum = bias;
                            for n in 0..N {
                                sum = add(sum, p[n]);
                            }
                            *optr = sum;
                            i += 1;
                        }
                        visitor.next_non_inner_axis()
                    }
                }
            }}

            #[inline(never)]
            #[allow(clippy::too_many_arguments)]
            $(#[$meta])*
            unsafe fn [<inner_loop_ $suffix>]<T: Datum + Copy>(
                iptr: *const T,
                kptr: *const T,
                bias: *const T,
                optr: *mut T,
                c: isize,
                visitor: &ZoneScanner,
                add: impl Fn(T, T) -> T,
                mul: impl Fn(T, T) -> T,
                ) { unsafe {
                let mut sum = *bias.offset(c);
                let mut iter = visitor.valid_offsets_ker_in();
                if iter.size_hint() == (3, Some(3)) {
                    let (ix, v) = iter.next().unwrap();
                    let k0 = *kptr.add(ix);
                    let i0 = *iptr.offset(v);
                    let (ix, v) = iter.next().unwrap();
                    let k1 = *kptr.add(ix);
                    let i1 = *iptr.offset(v);
                    let (ix, v) = iter.next().unwrap();
                    let k2 = *kptr.add(ix);
                    let i2 = *iptr.offset(v);
                    sum = add(add(add(sum, mul(k0, i0)), mul(k1, i1)), mul(k2, i2));
                } else {
                    for (ix, v) in iter {
                        let k = *kptr.add(ix);
                        let i = *iptr.offset(v);
                        sum = add(sum, mul(k, i));
                    }
                }
                let optr = optr.offset(visitor.output_offset);
                *optr = sum;
            }}
        }
    }
}

impl_eval!(generic);
impl_eval! {
#[target_feature(enable = "fp16")]
#[cfg(target_arch = "aarch64")]
aarch64fp16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::cnn::{KernelFormat, PaddingSpec, PoolSpec};
    use crate::ops::cnn::conv::Conv;
    use crate::ops::math::max;
    use crate::ops::nn::DataFormat;

    /// Build a model: input → Conv (depthwise) → Max(0) and return the optimized
    /// model alongside the input/kernel/bias tensors. The Conv is configured for
    /// group == channels (depthwise).
    fn build_depthwise_with_relu_model(
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
    ) -> TractResult<(TypedModel, Tensor, Tensor, Tensor)> {
        // NCHW input
        let mut model = TypedModel::default();
        let input_shape = tvec![n, c, h, w];
        let input_fact = f32::datum_type()
            .fact(ShapeFact::from_dims(input_shape.iter().map(|d| TDim::Val(*d as i64))));
        let x = model.add_source("x", input_fact)?;

        // Depthwise: kernel shape OIHW = (c, 1, kh, kw)
        let mut k_data = vec![0f32; c * 1 * kh * kw];
        for (i, v) in k_data.iter_mut().enumerate() {
            *v = ((i as f32) * 0.13 - 0.5).sin();
        }
        let kernel_t = tract_ndarray::Array4::from_shape_vec((c, 1, kh, kw), k_data)?;
        let kernel = model.add_const("kernel", Tensor::from(kernel_t.clone()))?;

        // Bias [c]
        let bias_data: Vec<f32> = (0..c).map(|i| (i as f32) * 0.05 - 0.1).collect();
        let bias_t = tract_ndarray::Array1::from_vec(bias_data.clone());
        let bias = model.add_const("bias", Tensor::from(bias_t.clone()))?;

        // Build Conv with group=c (depthwise).
        let pool_spec = PoolSpec::new(
            DataFormat::NCHW,
            tvec!(kh, kw),
            PaddingSpec::Valid,
            None,
            None,
            c, // input_channels
            c, // output_channels
        );
        let conv = Conv {
            pool_spec,
            kernel_fmt: KernelFormat::OIHW,
            group: c,
            q_params: None,
        };
        let conv_out = model.wire_node("conv", conv, &[x, kernel, bias])?[0];

        // Max(0) — broadcast scalar 0 across the conv output (rank 4).
        let zero = model.add_const(
            "zero",
            tensor0(0.0_f32).broadcast_into_rank(4)?,
        )?;
        let relu_out =
            model.wire_node("relu", max(), &[conv_out, zero])?[0];
        model.select_output_outlets(&[relu_out])?;

        // Build a sample input.
        let mut x_data = vec![0f32; n * c * h * w];
        for (i, v) in x_data.iter_mut().enumerate() {
            *v = ((i as f32) * 0.07).sin() * 1.5;
        }
        let x_tensor = Tensor::from(
            tract_ndarray::Array4::from_shape_vec((n, c, h, w), x_data)?,
        );
        let kernel_tensor = Tensor::from(kernel_t);
        let bias_tensor = Tensor::from(bias_t);
        Ok((model, x_tensor, kernel_tensor, bias_tensor))
    }

    /// Run the model with the given input, return the f32 slice as a Vec.
    fn run_model_collect_f32(model: TypedModel, input: &Tensor) -> TractResult<Vec<f32>> {
        let runnable = model.into_runnable()?;
        let outputs = runnable.run(tvec![input.clone().into_tvalue()])?;
        let out = outputs.into_iter().next().unwrap();
        let tensor = out.into_tensor();
        // SAFETY: the test wires only f32 paths; the output tensor is f32.
        let slice = unsafe { tensor.as_slice_unchecked::<f32>() };
        Ok(slice.to_vec())
    }

    /// Bit-identity: applying the fused post-op Max(0) inside DepthWise.eval()
    /// is bit-identical to running DepthWise.eval() with empty post_ops and
    /// applying f32::max(_, 0.0) externally to its output.
    ///
    /// Apples-to-apples comparison: same DepthWise instance, same eval inner
    /// loop, just toggling whether the Max is applied via apply_post_ops_f32
    /// or after-the-fact. This isolates the post-op apply path from the
    /// (irrelevant) Conv-vs-DepthWise codegen-routing question that the
    /// model-level decluttered-vs-optimized comparison would conflate.
    #[test]
    fn depthwise_with_fused_max_matches_separate_max() -> TractResult<()> {
        let (model, x, _kernel, _bias) =
            build_depthwise_with_relu_model(1, 8, 6, 6, 3, 3)?;

        // Optimize once → fuse fires, the resulting model has DepthWise with
        // post_ops=[ScalarMax(0.0)].
        let optimized_with_max = model.into_optimized()?;

        // Confirm fuse fired: at least one DepthWise has post_ops, no
        // OptMaxByScalar lingers as a separate op.
        let mut depthwise_node_id: Option<usize> = None;
        for n in optimized_with_max.nodes() {
            if let Some(dw) = n.op_as::<DepthWise>() {
                if !dw.post_ops.is_empty() {
                    depthwise_node_id = Some(n.id);
                    break;
                }
            }
        }
        let dw_node_id = depthwise_node_id
            .expect("expected DepthWise with post_ops after fuse pass");
        assert!(
            !optimized_with_max
                .nodes()
                .iter()
                .any(|n| n.op.name() == "OptMaxByScalar"),
            "expected OptMaxByScalar to be absorbed into DepthWise.post_ops"
        );

        // Run the with-postop graph → candidate (DepthWise.eval applies Max(0)
        // inline via apply_post_ops_f32).
        let candidate = run_model_collect_f32(optimized_with_max.clone(), &x)?;

        // Build a sibling graph: same optimized graph, but with the DepthWise's
        // post_ops cleared. Running this gives raw DepthWise output (no Max),
        // which we then clamp externally with f32::max — bit-identical operation
        // to apply_post_ops_f32, just applied at a different point in control flow.
        let mut optimized_no_postop = optimized_with_max;
        optimized_no_postop
            .node_mut(dw_node_id)
            .op_as_mut::<DepthWise>()
            .expect("DepthWise present at known node id")
            .post_ops
            .clear();
        let raw = run_model_collect_f32(optimized_no_postop, &x)?;

        // Apply Max(0) externally — same f32::max as apply_post_ops_f32.
        let baseline: Vec<f32> = raw.iter().map(|v| v.max(0.0)).collect();

        assert_eq!(baseline.len(), candidate.len());
        for (i, (a, b)) in baseline.iter().zip(candidate.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit-identity mismatch at output index {i}: \
                 baseline (raw_DepthWise.eval[{i}].max(0.0))={a} bits=0x{:x} \
                 candidate (DepthWise_with_post_ops.eval[{i}])={b} bits=0x{:x}",
                a.to_bits(),
                b.to_bits()
            );
        }
        Ok(())
    }
}
//#[target_feature(enable = "fp16")] impl_eval!(aarch64fp16);

/* partial alternative impl that may be relevant when simd gets better */

/*
#[inline(never)]
unsafe fn process_zone_4_f32(
&self,
zone: &Zone,
c_stride_i: isize,
c_stride_o: isize,
k_stride_i: isize,
iptr: *const f32,
kptr: *const f32,
bias: *const f32,
optr: *mut f32,
) {
use std::simd::*;
let mut visitor = ZoneScanner::new(zone, &self.patch);
let ioffset0 = zone.values_offsets[0].1;
let ioffset1 = zone.values_offsets[1].1;
let ioffset2 = zone.values_offsets[2].1;
let ioffset3 = zone.values_offsets[3].1;
for c in 0..*self.input_shape.c() as isize {
visitor.reset();
let kptr = kptr.offset(k_stride_i * c);
let iptr = iptr.offset(c_stride_i * c);
let optr = optr.offset(c_stride_o * c);
let k0 = *kptr.offset(zone.values_offsets[0].0 as isize);
let k1 = *kptr.offset(zone.values_offsets[1].0 as isize);
let k2 = *kptr.offset(zone.values_offsets[2].0 as isize);
let k3 = *kptr.offset(zone.values_offsets[3].0 as isize);
let k0 = f32x4::splat(k0);
let k1 = f32x4::splat(k1);
let k2 = f32x4::splat(k2);
let k3 = f32x4::splat(k3);
let bias = f32x4::splat(*bias.offset(c));
while !visitor.done {
let iptr = iptr.offset(visitor.input_center_offset);
let optr = optr.offset(visitor.output_offset);
let mut i  = 0;
while i + 4 <
for i in 0..visitor.inner_loop_len as isize {
let iptr = iptr.offset(visitor.inner_loop_input_full_stride * i);
let optr = optr.offset(visitor.inner_loop_output_stride * i);
let i0 = *iptr.offset(ioffset0);
let i1 = *iptr.offset(ioffset1);
let i2 = *iptr.offset(ioffset2);
let i3 = *iptr.offset(ioffset3);
let i = f32x4::from_array([i0, i1, i2, i3]);
let p = (i * k).reduce_sum();
let sum = bias + p;
     *optr = sum
     }
     visitor.next_non_inner_axis()
     }
     }
     }
     */

/*
#[inline(never)]
unsafe fn process_zone_4_f32(
&self,
zone: &Zone,
c_stride_i: isize,
c_stride_o: isize,
k_stride_i: isize,
iptr: *const f32,
kptr: *const f32,
bias: *const f32,
optr: *mut f32,
) {
use std::simd::*;
let mut visitor = ZoneScanner::new(zone, &self.patch);
let ioffset0 = zone.values_offsets[0].1;
let ioffset1 = zone.values_offsets[1].1;
let ioffset2 = zone.values_offsets[2].1;
let ioffset3 = zone.values_offsets[3].1;
for c in 0..*self.input_shape.c() as isize {
visitor.reset();
let kptr = kptr.offset(k_stride_i * c);
let iptr = iptr.offset(c_stride_i * c);
let optr = optr.offset(c_stride_o * c);
let k0 = *kptr.offset(zone.values_offsets[0].0 as isize);
let k1 = *kptr.offset(zone.values_offsets[1].0 as isize);
let k2 = *kptr.offset(zone.values_offsets[2].0 as isize);
let k3 = *kptr.offset(zone.values_offsets[3].0 as isize);
let k = f32x4::from_array([k0, k1, k2, k3]);
let bias = *bias.offset(c);
while !visitor.done {
let iptr = iptr.offset(visitor.input_center_offset);
let optr = optr.offset(visitor.output_offset);
for i in 0..visitor.inner_loop_len as isize {
let iptr = iptr.offset(visitor.inner_loop_input_full_stride * i);
let optr = optr.offset(visitor.inner_loop_output_stride * i);
let i0 = *iptr.offset(ioffset0);
let i1 = *iptr.offset(ioffset1);
let i2 = *iptr.offset(ioffset2);
let i3 = *iptr.offset(ioffset3);
let i = f32x4::from_array([i0, i1, i2, i3]);
let p = (i * k).reduce_sum();
let sum = bias + p;
     *optr = sum
     }
     visitor.next_non_inner_axis()
     }
     }
     }
     */

/*
#[inline(never)]
unsafe fn process_zone_4<T: Datum + Copy + ndarray::LinalgScalar>(
&self,
zone: &Zone,
c_stride_i: isize,
c_stride_o: isize,
k_stride_i: isize,
iptr: *const T,
kptr: *const T,
bias: *const T,
optr: *mut T,
) {
let mut visitor = ZoneScanner::new(zone, &self.patch);
let ioffset0 = zone.values_offsets[0].1;
let ioffset1 = zone.values_offsets[1].1;
let ioffset2 = zone.values_offsets[2].1;
let ioffset3 = zone.values_offsets[3].1;
for c in 0..*self.input_shape.c() as isize {
visitor.reset();
let kptr = kptr.offset(k_stride_i * c);
let iptr = iptr.offset(c_stride_i * c);
let optr = optr.offset(c_stride_o * c);
let k0 = *kptr.offset(zone.values_offsets[0].0 as isize);
let k1 = *kptr.offset(zone.values_offsets[1].0 as isize);
let k2 = *kptr.offset(zone.values_offsets[2].0 as isize);
let k3 = *kptr.offset(zone.values_offsets[3].0 as isize);
let bias = *bias.offset(c);
while !visitor.done {
let iptr = iptr.offset(visitor.input_center_offset);
let optr = optr.offset(visitor.output_offset);
for i in 0..visitor.inner_loop_len as isize {
let iptr = iptr.offset(visitor.inner_loop_input_full_stride * i);
let optr = optr.offset(visitor.inner_loop_output_stride * i);
let i0 = *iptr.offset(ioffset0);
let i1 = *iptr.offset(ioffset1);
let i2 = *iptr.offset(ioffset2);
let i3 = *iptr.offset(ioffset3);
let p0 = i0 * k0;
let p1 = i1 * k1;
let p2 = i2 * k2;
let p3 = i3 * k3;
let sum = bias + p0 + p1 + p2 + p3;
     *optr = sum
     }
     visitor.next_non_inner_axis()
     }
     }
     }
     */
