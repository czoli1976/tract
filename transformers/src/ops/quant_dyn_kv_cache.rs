//! Storage-only quantized KV cache: a drop-in replacement for `DynKeyValueCache` that keeps the
//! resident cache in int4/int8 (KIVI layout — Keys per-channel, Values per-token) and dequantizes
//! back to the input dtype on read. Because it stays a cache-shaped op, it still resolves the
//! past-length symbol `P` (via `init_tensor_fact` / `resolve_symbols`), so all downstream
//! position/RoPE/mask/attention math is untouched — unlike fusing attention, which orphans `P`.

use std::str::FromStr;

use tract_nnef::internal::*;
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::ser::{datum_type, tdims};
use tract_nnef::tract_core::ops::array::MultiBroadcastTo;
use tract_nnef::tract_core::ops::cast::Cast;
use tract_nnef::tract_core::ops::change_axes::AxisOp;
use tract_nnef::tract_core::ops::{FrozenOpState, OpStateFreeze};
use tract_nnef::tract_core::transform::ModelTransform;
use tract_nnef::tract_ndarray::{Array2, Ix4, s};

use crate::ops::apply_rope::ApplyRope;
use crate::ops::dyn_kv_cache::DynKeyValueCache;
use crate::ops::kv_quant::{BlockQuantKeyCache, QuantValueCache};
use crate::ops::sdpa::Sdpa;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_quant_dyn_kv_cache);
    registry.register_primitive(
        "tract_transformers_quantized_dyn_kv_cache",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::String.named("name"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("bits"),
            TypeName::Integer.named("per_channel"),
            TypeName::String.named("datum_type"),
            TypeName::Integer.array().named("past_sequence_shape"),
            TypeName::Integer.array().named("input_sequence_shape"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_quant_dyn_kv_cache,
    );
}

fn ser_quant_dyn_kv_cache(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &QuantizedDynKeyValueCache,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_transformers_quantized_dyn_kv_cache",
        &[input],
        &[
            ("name", string(&op.name)),
            ("axis", numeric(op.axis)),
            ("bits", numeric(op.bits)),
            ("per_channel", numeric(op.per_channel as i64)),
            ("datum_type", datum_type(op.past_sequence_fact.datum_type)),
            ("past_sequence_shape", tdims(op.past_sequence_fact.shape.dims())),
            ("input_sequence_shape", tdims(op.input_sequence_fact.shape.dims())),
        ],
    )))
}

fn de_quant_dyn_kv_cache(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let name: String = invocation.named_arg_as(builder, "name")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let bits: i64 = invocation.named_arg_as(builder, "bits")?;
    let per_channel: i64 = invocation.named_arg_as(builder, "per_channel")?;
    let dt = DatumType::from_str(&invocation.named_arg_as::<String>(builder, "datum_type")?)?;
    let past_sequence_shape: TVec<TDim> = builder
        .allowing_new_symbols(|builder| invocation.named_arg_as(builder, "past_sequence_shape"))?;
    let input_sequence_shape: TVec<TDim> = builder
        .allowing_new_symbols(|builder| invocation.named_arg_as(builder, "input_sequence_shape"))?;
    builder.wire(
        QuantizedDynKeyValueCache {
            name,
            axis,
            bits: bits as u32,
            per_channel: per_channel != 0,
            past_sequence_fact: dt.fact(&*past_sequence_shape),
            input_sequence_fact: dt.fact(&*input_sequence_shape),
        },
        &[input],
    )
}

/// Per-head quantized store, dispatched on the KIVI layout for this cache (K per-channel,
/// V per-token). Reuses the packed int4/int8 stores + NEON dequant from `kv_quant`.
#[derive(Clone, Debug)]
enum HeadCache {
    Key(BlockQuantKeyCache),
    Value(QuantValueCache),
}

impl HeadCache {
    fn new(per_channel: bool, d: usize, bits: u32) -> Self {
        if per_channel {
            HeadCache::Key(BlockQuantKeyCache::with_bits(d, bits))
        } else {
            HeadCache::Value(QuantValueCache::with_bits(d, bits))
        }
    }
    fn push(&mut self, row: &[f32]) {
        match self {
            HeadCache::Key(c) => c.push_token(row),
            HeadCache::Value(c) => c.push_token(row),
        }
    }
    fn dequant_all(&self) -> Array2<f32> {
        match self {
            HeadCache::Key(c) => c.dequant_all(),
            HeadCache::Value(c) => c.dequant_all(),
        }
    }
    fn len(&self) -> usize {
        match self {
            HeadCache::Key(c) => c.len(),
            HeadCache::Value(c) => c.len(),
        }
    }
    fn memory_bytes(&self) -> usize {
        match self {
            HeadCache::Key(c) => c.memory_bytes(),
            HeadCache::Value(c) => c.memory_bytes(),
        }
    }
}

/// Quantized replacement for `DynKeyValueCache`. `per_channel` selects the KIVI layout
/// (Keys: true, Values: false); `bits` is 4 or 8. Requires the sequence axis to be `rank-2`
/// with the head dim last (the standard `[.., H, S, D]` decode-cache layout).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedDynKeyValueCache {
    pub name: String,
    pub axis: usize,
    pub bits: u32,
    pub per_channel: bool,
    pub past_sequence_fact: TypedFact,
    pub input_sequence_fact: TypedFact,
}

impl Op for QuantizedDynKeyValueCache {
    fn name(&self) -> StaticName {
        "QuantizedDynKeyValueCache".to_string().into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "bits={}, per_channel={}, axis={}",
            self.bits, self.per_channel, self.axis
        )])
    }
    op_as_typed_op!();
}

impl EvalOp for QuantizedDynKeyValueCache {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(QuantizedDynKvCacheState {
            name: self.name.clone(),
            axis: self.axis,
            bits: self.bits,
            per_channel: self.per_channel,
            past_sequence_fact: self.past_sequence_fact.clone(),
            caches: Vec::new(),
            lead_shape: tvec!(),
            d: 0,
            len: 0,
        })))
    }
}

impl TypedOp for QuantizedDynKeyValueCache {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 1);
        let mut fact = inputs[0].without_value();
        fact.shape.set(
            self.axis,
            self.past_sequence_fact.shape.dims()[self.axis].clone()
                + self.input_sequence_fact.shape.dims()[self.axis].clone(),
        );
        Ok(tvec!(fact))
    }
    as_op!();
}

#[derive(Clone, Debug)]
pub struct QuantizedDynKvCacheState {
    name: String,
    axis: usize,
    bits: u32,
    per_channel: bool,
    past_sequence_fact: TypedFact,
    caches: Vec<HeadCache>,
    lead_shape: TVec<usize>, // dims before the seq axis (e.g. [batch, heads])
    d: usize,                // head dim (last axis)
    len: usize,              // accumulated sequence length
}

impl QuantizedDynKvCacheState {
    /// Bind the single unresolved symbol in `past_sequence_fact` (the past length `P`) to the
    /// current accumulated length. Mirrors `DynKeyValueCacheState::resolve_symbols`.
    fn bind_past(&self, state: &mut TurnState, len: usize) -> TractResult<()> {
        let unresolved = self
            .past_sequence_fact
            .shape
            .iter()
            .filter_map(|symb| match symb {
                TDim::Sym(s) if state.resolved_symbols.get(&s).is_none() => Some(s),
                _ => None,
            })
            .collect_vec();
        if unresolved.is_empty() {
            return Ok(());
        }
        ensure!(unresolved.len() == 1);
        let sym = &unresolved[0];
        state.resolved_symbols.set(sym, len as i64);
        if state.scenario.is_none() {
            state.scenario = sym.scope().unwrap().guess_scenario(&state.resolved_symbols)?;
        }
        Ok(())
    }
}

impl OpState for QuantizedDynKvCacheState {
    fn init_tensor_fact(&self) -> Option<(String, TypedFact)> {
        Some((self.name.clone(), self.past_sequence_fact.clone()))
    }
    fn has_init_tensor_fact(&self) -> bool {
        true
    }

    fn load_from(
        &mut self,
        state: &mut TurnState,
        states: &mut dyn Iterator<Item = TValue>,
    ) -> TractResult<()> {
        let init = states.next().context("Not enough state initializers")?;
        self.bind_past(state, init.shape()[self.axis])?;
        // Quantize the (usually empty) initializer into the packed caches.
        let f32init = init.cast_to::<f32>()?;
        self.ingest(&f32init)?;
        Ok(())
    }

    fn save_to(&self, states: &mut Vec<TValue>) -> TractResult<()> {
        states.push(self.dequantized()?.into_tvalue());
        Ok(())
    }

    fn resolve_symbols(&mut self, state: &mut TurnState) -> TractResult<()> {
        self.bind_past(state, self.len)
    }

    fn eval(
        &mut self,
        _state: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input_dt = input.datum_type();
        let f32in = input.cast_to::<f32>()?;
        self.ingest(&f32in)?;
        Ok(tvec!(self.dequantized()?.cast_to_dt(input_dt)?.into_owned().into_tvalue()))
    }
}

impl QuantizedDynKvCacheState {
    /// Append every new token of `f32in` (`[B, H, S, D]`, seq axis = rank-2) to the packed caches.
    fn ingest(&mut self, f32in: &Tensor) -> TractResult<()> {
        let view = f32in.to_plain_array_view::<f32>()?;
        ensure!(view.ndim() == 4, "quantized KV cache supports rank-4 [B, H, S, D] caches");
        ensure!(self.axis == 2, "seq axis must be 2 ([B, H, S, D])");
        let view = view.into_dimensionality::<Ix4>()?;
        let (b, h, s, d) = view.dim();
        if self.caches.is_empty() {
            self.d = d;
            self.lead_shape = tvec!(b, h);
            self.caches =
                (0..b * h).map(|_| HeadCache::new(self.per_channel, d, self.bits)).collect();
        }
        ensure!(
            d == self.d && self.lead_shape.as_slice() == [b, h],
            "cache shape changed between steps"
        );
        for bi in 0..b {
            for hi in 0..h {
                let idx = bi * h + hi;
                for t in 0..s {
                    self.caches[idx].push(view.slice(s![bi, hi, t, ..]).as_slice().unwrap());
                }
            }
        }
        self.len += s;
        Ok(())
    }

    /// Reconstruct the full `[B, H, T, D]` f32 cache from the packed per-head stores.
    fn dequantized(&self) -> TractResult<Tensor> {
        let (t, d) = (self.len, self.d);
        let leading: usize = self.lead_shape.iter().product();
        let mut data = vec![0f32; leading * t * d];
        for (idx, cache) in self.caches.iter().enumerate() {
            let deq = cache.dequant_all(); // [T, D]
            let base = idx * t * d;
            for ti in 0..t {
                for di in 0..d {
                    data[base + ti * d + di] = deq[(ti, di)];
                }
            }
        }
        let mut shape: Vec<usize> = self.lead_shape.to_vec();
        shape.push(t);
        shape.push(d);
        Tensor::from_shape(&shape, &data)
    }
}

#[derive(Clone, Debug)]
struct FrozenQuantizedDynKvCacheState {
    name: String,
    axis: usize,
    bits: u32,
    per_channel: bool,
    past_sequence_fact: TypedFact,
    caches: Vec<HeadCache>,
    lead_shape: TVec<usize>,
    d: usize,
    len: usize,
}

impl OpStateFreeze for QuantizedDynKvCacheState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenQuantizedDynKvCacheState {
            name: self.name.clone(),
            axis: self.axis,
            bits: self.bits,
            per_channel: self.per_channel,
            past_sequence_fact: self.past_sequence_fact.clone(),
            caches: self.caches.clone(),
            lead_shape: self.lead_shape.clone(),
            d: self.d,
            len: self.len,
        })
    }
}

impl FrozenOpState for FrozenQuantizedDynKvCacheState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(QuantizedDynKvCacheState {
            name: self.name.clone(),
            axis: self.axis,
            bits: self.bits,
            per_channel: self.per_channel,
            past_sequence_fact: self.past_sequence_fact.clone(),
            caches: self.caches.clone(),
            lead_shape: self.lead_shape.clone(),
            d: self.d,
            len: self.len,
        })
    }
}

/// Total resident bytes across all per-head packed caches (for reporting/tests).
impl QuantizedDynKvCacheState {
    pub fn resident_bytes(&self) -> usize {
        self.caches.iter().map(|c| c.memory_bytes()).sum()
    }
}

// ── Storage-quantization transform ──────────────────────────────────────────────────────────────

/// Walk an Sdpa K/V input back through cache-read plumbing (broadcast / reshape / cast / on-read
/// RoPE) to the `DynKeyValueCache` node; return its id. Every hop must be single-consumer.
fn walk_to_cache_node(model: &TypedModel, start: OutletId) -> Option<usize> {
    let mut outlet = start;
    loop {
        let n = model.node(outlet.node);
        if n.outputs[outlet.slot].successors.len() != 1 {
            return None;
        }
        if n.op_is::<DynKeyValueCache>() {
            return Some(n.id);
        } else if n.op_is::<ApplyRope>()
            || n.op_is::<MultiBroadcastTo>()
            || n.op_is::<AxisOp>()
            || n.op_is::<Cast>()
        {
            outlet = n.inputs[0];
        } else {
            return None;
        }
    }
}

fn replace_with_quant_cache(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    cache_node_id: usize,
    per_channel: bool,
    bits: u32,
) -> TractResult<()> {
    let cnode = model.node(cache_node_id);
    let dkv = cnode.op_as::<DynKeyValueCache>().context("expected DynKeyValueCache")?;
    let tap = patch.taps(model, &[cnode.inputs[0]])?;
    let new = patch.wire_node(
        format!("{}.quant", cnode.name),
        QuantizedDynKeyValueCache {
            name: dkv.name.clone(),
            axis: dkv.axis,
            bits,
            per_channel,
            past_sequence_fact: dkv.past_sequence_fact.clone(),
            input_sequence_fact: dkv.input_sequence_fact.clone(),
        },
        &tap,
    )?;
    patch.shunt_outside(model, cache_node_id.into(), new[0])?;
    Ok(())
}

/// For each `Sdpa`, quantize its two upstream KV caches in place: the K cache (input 1) per-channel
/// and the V cache (input 2) per-token (KIVI). Attention/RoPE/mask downstream are left untouched.
pub fn quantize_kv_storage_rule(
    ctx: &u32,
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    _op: &Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    if node.inputs.len() != 3 && node.inputs.len() != 4 {
        return Ok(None);
    }
    let (Some(k_cache), Some(v_cache)) =
        (walk_to_cache_node(model, node.inputs[1]), walk_to_cache_node(model, node.inputs[2]))
    else {
        return Ok(None);
    };
    if k_cache == v_cache {
        return Ok(None);
    }
    let mut patch = TypedModelPatch::default();
    // KIVI layout: Keys per-channel (outlier-channel robust, via block-wise finalized scales),
    // Values per-token. Both dequantize consistently as the cache grows.
    replace_with_quant_cache(&mut patch, model, k_cache, true, *ctx)?;
    replace_with_quant_cache(&mut patch, model, v_cache, false, *ctx)?;
    Ok(Some(patch))
}

/// Replace the KV caches feeding each attention with storage-quantized ones (int4/int8). Keeps a
/// cache-shaped op so the past-length symbol still resolves; attention math is unchanged.
#[derive(Debug, Clone, Copy)]
pub struct QuantizeKvStorageTransform {
    pub bits: u32,
}

impl Default for QuantizeKvStorageTransform {
    fn default() -> Self {
        QuantizeKvStorageTransform { bits: 8 }
    }
}

impl ModelTransform for QuantizeKvStorageTransform {
    fn name(&self) -> StaticName {
        "quantize_kv_storage".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        ensure!(self.bits == 4 || self.bits == 8, "KV quantization bits must be 4 or 8");
        crate::rewriter::ApplyRopeTransform.transform(model)?;
        crate::rewriter::KeyValueCacheTransform.transform(model)?;
        Rewriter::default()
            .with_rule_for("quantize-kv-storage", quantize_kv_storage_rule)
            .rewrite(&self.bits, model)?;
        model.compact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_core::ops::array::TypedConcat;

    fn model_with_cache(bits: u32, per_channel: bool) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let p = model.sym("P");
        let (b, h, d) = (1usize, 2usize, 8usize);
        let inf: TVec<TDim> = tvec![b.to_dim(), h.to_dim(), s.into(), d.to_dim()];
        let pf: TVec<TDim> = tvec![b.to_dim(), h.to_dim(), p.into(), d.to_dim()];
        let input = model.add_source("input", f32::fact(&inf))?;
        let op = QuantizedDynKeyValueCache {
            name: "kv0".to_string(),
            axis: 2,
            bits,
            per_channel,
            past_sequence_fact: f32::fact(&pf),
            input_sequence_fact: f32::fact(&inf),
        };
        let out = model.wire_node("kv", op, &[input])?;
        model.select_output_outlets(&out)?;
        Ok(model)
    }

    // The quantized cache accumulates like a plain concat and dequantizes near-losslessly at int8.
    #[test]
    fn quant_cache_accumulates_near_lossless() -> TractResult<()> {
        let mut rt = model_with_cache(8, false)?.into_runnable()?.spawn()?;
        let (b, h, d) = (1usize, 2usize, 8usize);
        let mut st = 3u64;
        let mut nf = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let mut acc: Option<Tensor> = None;
        for _ in 0..10 {
            let step = Tensor::from_shape(
                &[b, h, 1, d],
                &(0..b * h * d).map(|_| nf()).collect::<Vec<f32>>(),
            )?;
            let out = rt.run(tvec![step.clone().into()])?.remove(0).into_tensor();
            acc = Some(match acc.take() {
                None => step,
                Some(a) => TypedConcat { axis: 2 }
                    .eval(tvec![a.into(), step.into()])?
                    .remove(0)
                    .into_tensor(),
            });
            acc.as_ref().unwrap().close_enough(&out, Approximation::SuperApproximate)?;
        }
        Ok(())
    }

    #[test]
    fn quant_cache_nnef_round_trip() -> TractResult<()> {
        use crate::WithTractTransformers;
        let model = model_with_cache(4, true)?;
        let nnef = tract_nnef::nnef().with_tract_transformers();
        let mut buffer = vec![];
        nnef.write_to_tar(&model, &mut buffer)?;
        let reloaded = nnef.model_for_read(&mut &*buffer)?;
        let n = reloaded
            .nodes()
            .iter()
            .find_map(|n| n.op_as::<QuantizedDynKeyValueCache>())
            .context("QuantizedDynKeyValueCache missing after round-trip")?;
        assert_eq!(n.bits, 4);
        assert!(n.per_channel);
        assert_eq!(n.axis, 2);
        Ok(())
    }
}

// Real-model decode harness: baseline f16 vs int8/int4 storage-quant KV cache on OpenELM-270M.
//   cargo test --release -p tract-transformers quant_dyn_kv_cache::tests::decode_openelm -- --ignored --nocapture
#[cfg(test)]
mod harness {
    use super::*;
    use crate::WithTractTransformers;
    use std::sync::Arc;
    use std::time::Instant;
    use tract_nnef::tract_core::model::typed::TypedRunnableModel;
    use tract_nnef::tract_core::transform::ModelTransform;

    #[test]
    #[ignore]
    fn decode_openelm() -> TractResult<()> {
        let path = "/Users/CZoli/coding/pocket-tts-tract/openelm.nnef.tgz";
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: {path} missing");
            return Ok(());
        }
        let nnef = tract_nnef::nnef().with_tract_transformers();
        let base = nnef.model_for_path(path)?;

        let build = |quant: Option<u32>| -> TractResult<_> {
            let mut m = base.clone().into_decluttered()?;
            match quant {
                None => {
                    crate::rewriter::ApplyRopeTransform.transform(&mut m)?;
                    crate::rewriter::KeyValueCacheTransform.transform(&mut m)?;
                }
                Some(bits) => super::QuantizeKvStorageTransform { bits }.transform(&mut m)?,
            }
            m.into_optimized()?.into_runnable()
        };
        let f32b = build(None)?;
        let q8 = build(Some(8))?;
        let q4 = build(Some(4))?;

        let steps = 96usize;
        let mut tok = 1234u64;
        let mut next_tok = || {
            tok = tok.wrapping_mul(6364136223846793005).wrapping_add(1);
            (tok >> 33) % 32000
        };
        let toks: Vec<i64> = (0..steps).map(|_| next_tok() as i64).collect();

        let run_all = |rt: &Arc<TypedRunnableModel>| -> TractResult<(Vec<Vec<f32>>, Vec<f64>)> {
            let mut st = rt.spawn()?;
            let mut logits = Vec::new();
            let mut times = Vec::new();
            for &t in &toks {
                let inp = tensor2(&[[t]]);
                let s = Instant::now();
                let o = st.run(tvec![inp.into()])?.remove(0).into_tensor();
                times.push(s.elapsed().as_secs_f64() * 1e3);
                let of = o.cast_to::<f32>()?;
                let v = of.to_plain_array_view::<f32>()?.iter().copied().collect::<Vec<_>>();
                logits.push(v);
            }
            Ok((logits, times))
        };

        let (lb, tb) = run_all(&f32b)?;
        let (l8, t8) = run_all(&q8)?;
        let (l4, t4) = run_all(&q4)?;

        let argmax = |v: &[f32]| {
            v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        };
        let relerr = |a: &[f32], b: &[f32]| {
            let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
            let den: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            num / den
        };
        let summarize = |name: &str, lq: &[Vec<f32>]| {
            let mut re = 0f32;
            let mut top1 = 0usize;
            for i in 0..steps {
                re += relerr(&lb[i], &lq[i]);
                if argmax(&lb[i]) == argmax(&lq[i]) {
                    top1 += 1;
                }
            }
            println!(
                "  {name}: mean logit rel-err={:.4}  top-1 match={}/{} ({:.0}%)",
                re / steps as f32,
                top1,
                steps,
                100.0 * top1 as f32 / steps as f32
            );
        };
        let med = |t: &[f64], lo: usize, hi: usize| {
            let mut w: Vec<f64> = t[lo..hi.min(t.len())].to_vec();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            w[w.len() / 2]
        };
        println!("\n  OpenELM-270M decode, {steps} steps — accuracy vs f16 baseline:");
        summarize("int8", &l8);
        summarize("int4", &l4);
        println!("\n  per-token latency (median over window) by cache depth:");
        println!("     depth   f16(ms)  int8(ms)  int8/f16  int4(ms)  int4/f16");
        for &(lo, hi) in &[(4usize, 12usize), (28, 36), (60, 68), (88, 96)] {
            let (b, e8, e4) = (med(&tb, lo, hi), med(&t8, lo, hi), med(&t4, lo, hi));
            println!(
                "  {:>6}   {b:>7.1}  {e8:>7.1}  {:>6.2}x  {e4:>7.1}  {:>6.2}x",
                (lo + hi) / 2,
                e8 / b,
                e4 / b
            );
        }
        Ok(())
    }
}
