use tract_data::itertools::Itertools;
use tract_num_traits::Zero;

use crate::internal::*;

use super::Slice;

#[derive(new, Debug, Clone, Hash, PartialEq, Eq)]
pub struct TypedConcat {
    pub axis: usize,
}

impl TypedConcat {
    pub fn offsets(&self, inputs: &[&TypedFact]) -> TractResult<Vec<TDim>> {
        let mut offsets = vec![0.to_dim()];
        for slice in inputs {
            let len = slice.shape[self.axis].clone();
            let offset = len + offsets.last().unwrap();
            offsets.push(offset)
        }
        Ok(offsets)
    }
}

impl Op for TypedConcat {
    fn name(&self) -> StaticName {
        "Concat".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis)])
    }

    op_as_typed_op!();
}

impl TypedOp for TypedConcat {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() > 0);
        let mut fact = inputs[0].without_value();
        for input in inputs {
            if input.rank() != fact.rank()
                || input
                    .shape
                    .iter()
                    .zip(fact.shape.iter())
                    .enumerate()
                    .filter(|(ax, _)| *ax != self.axis)
                    .any(|(_, (i, f))| i != f)
            {
                bail!("Inconsistent concat {:?} inputs: {:?}", self, inputs);
            }
        }
        fact.shape.set(self.axis, self.offsets(inputs)?.pop().unwrap());
        Ok(tvec!(fact))
    }

    fn input_roi(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TVec<Option<TDim>>>> {
        let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
        rule_if_some!(roi = &output_fact.region_of_interest);
        let input_facts: TVec<&TypedFact> =
            node.inputs.iter().map(|i| model.outlet_fact(*i)).collect::<TractResult<_>>()?;
        let inputs_ref: Vec<&TypedFact> = input_facts.iter().copied().collect();
        let offsets = self.offsets(&inputs_ref)?;

        // Find the coordinate symbol that designates the concat axis in the
        // output ROI expression (if any). If absent, the ROI is invariant on
        // the concat axis and passes through to every input unchanged.
        let axis_sym = roi
            .symbols()
            .into_iter()
            .find(|s| crate::ops::logic::sym_to_coord_axis(s) == Some(self.axis));

        let input_rois: TVec<Option<TDim>> = (0..node.inputs.len())
            .map(|ix| {
                let shift = &offsets[ix];
                match &axis_sym {
                    None => Some(roi.clone()),
                    Some(sym) if shift.is_zero() => Some(roi.clone()),
                    Some(sym) => {
                        // Remap output 🎯axis → input 🎯axis + offset[ix], so
                        // that the i-th input receives the slice of the output
                        // ROI that corresponds to its [offset[i], offset[i+1])
                        // range, expressed in the input's local coordinates.
                        let shifted = TDim::Sym(sym.clone()) + shift.clone();
                        roi.substitute(sym, &shifted).ok().or_else(|| Some(roi.clone()))
                    }
                }
            })
            .collect();
        Ok(Some(input_rois))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut axes = AxesMapping::disconnected(inputs, outputs)?;
        for ax in 0..outputs[0].rank() {
            if ax != self.axis {
                for i in 0..inputs.len() {
                    axes = axes.linking((InOut::Out(0), ax), (InOut::In(i), ax))?;
                }
            }
        }
        Ok(axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        rule_if_some!(axis = change.transform_axis(self.axis));
        let op = TypedConcat { axis };
        Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if node.inputs.len() == 1 {
            return TypedModelPatch::shunt_one_op(model, node);
        }
        let inputs = model.node_input_facts(node.id)?;
        if let Some(pos) = inputs.iter().position(|f| f.shape.volume().is_zero()) {
            let mut inputs = node.inputs.clone();
            inputs.remove(pos);
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &inputs,
                self.clone(),
            )?));
        }
        Ok(None)
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        _node: &TypedNode,
        prefix: &str,
        inputs: &[OutletId],
        output_axis: usize,
        start: &TDim,
        end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        if output_axis != self.axis {
            return Ok(Some(patch.wire_node(prefix, self.clone(), inputs)?));
        }
        let facts =
            inputs.iter().map(|o| patch.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
        let offsets = self.offsets(&facts)?;
        std::mem::drop(facts);
        for (ix, (slice_start, slice_end)) in offsets.iter().tuple_windows().enumerate() {
            if (start.clone() - slice_start).prove_positive_or_zero()
                && (slice_end.clone() - end).prove_positive_or_zero()
            {
                return patch
                    .wire_node(
                        format!("{prefix}.slice-{output_axis}.{start}..{end}"),
                        Slice {
                            axis: output_axis,
                            start: (start.clone() - slice_start),
                            end: (end.clone() - slice_start),
                        },
                        &[inputs[ix]],
                    )
                    .map(Some);
            }
        }
        Ok(None)
    }
}

impl EvalOp for TypedConcat {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let result = Tensor::stack_tensors(self.axis, &inputs)?;
        Ok(tvec![result.into_tvalue()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// When the output has no `region_of_interest`, `input_roi` returns `None`
    /// (the pass moves on without recording demands for this node's inputs).
    #[test]
    fn input_roi_with_no_output_roi_returns_none() {
        let mut model = TypedModel::default();
        let fact = TypedFact::dt_shape(f32::datum_type(), [1.to_dim(), 4.to_dim()]);
        let i0 = model.add_source("a", fact.clone()).unwrap();
        let i1 = model.add_source("b", fact.clone()).unwrap();
        let outlets = model.wire_node("concat", TypedConcat { axis: 1 }, &[i0, i1]).unwrap();

        let node = &model.nodes()[outlets[0].node];
        let op = node.op.downcast_ref::<TypedConcat>().unwrap();
        assert!(op.input_roi(&model, node).unwrap().is_none());
    }

    /// When the output ROI mentions the concat axis, each input branch gets
    /// the ROI translated by its branch offset (input 0 unchanged, input 1
    /// shifted by +len_0, etc.) — mirrors `Slice::input_roi`'s shift but per
    /// branch.
    #[test]
    fn input_roi_translates_concat_axis_per_branch_offset() {
        let mut model = TypedModel::default();
        let fact = TypedFact::dt_shape(f32::datum_type(), [1.to_dim(), 4.to_dim()]);
        let i0 = model.add_source("a", fact.clone()).unwrap();
        let i1 = model.add_source("b", fact.clone()).unwrap();
        let i2 = model.add_source("c", fact.clone()).unwrap();
        let outlets =
            model.wire_node("concat", TypedConcat { axis: 1 }, &[i0, i1, i2]).unwrap();

        // Manually annotate the output with a ROI mentioning the axis-1 coord
        // symbol. The expression itself is opaque; only the substitution
        // matters for this test.
        let axis_sym = model.symbols.coord_sym(1);
        let roi_expr = TDim::Sym(axis_sym.clone());
        model.nodes_mut()[outlets[0].node].outputs[0].fact.region_of_interest =
            Some(roi_expr.clone());

        let node = &model.nodes()[outlets[0].node];
        let op = node.op.downcast_ref::<TypedConcat>().unwrap();
        let result = op.input_roi(&model, node).unwrap().expect("input_roi");

        assert_eq!(result.len(), 3);
        // Input 0 (offset 0): unchanged
        assert_eq!(result[0], Some(roi_expr.clone()));
        // Input 1 (offset 4): 🎯1 + 4
        assert_eq!(result[1], Some(TDim::Sym(axis_sym.clone()) + 4.to_dim()));
        // Input 2 (offset 8): 🎯1 + 8
        assert_eq!(result[2], Some(TDim::Sym(axis_sym) + 8.to_dim()));
    }

    /// When the output ROI does not mention the concat axis (e.g., only
    /// references a non-concat axis), every input branch gets the same ROI
    /// passed through unchanged.
    #[test]
    fn input_roi_without_concat_axis_passes_through_unchanged() {
        let mut model = TypedModel::default();
        let fact = TypedFact::dt_shape(f32::datum_type(), [1.to_dim(), 4.to_dim()]);
        let i0 = model.add_source("a", fact.clone()).unwrap();
        let i1 = model.add_source("b", fact.clone()).unwrap();
        let outlets = model.wire_node("concat", TypedConcat { axis: 1 }, &[i0, i1]).unwrap();

        // ROI mentions axis 0, not the concat axis 1 — should pass through.
        let other_axis_sym = model.symbols.coord_sym(0);
        let roi_expr = TDim::Sym(other_axis_sym);
        model.nodes_mut()[outlets[0].node].outputs[0].fact.region_of_interest =
            Some(roi_expr.clone());

        let node = &model.nodes()[outlets[0].node];
        let op = node.op.downcast_ref::<TypedConcat>().unwrap();
        let result = op.input_roi(&model, node).unwrap().expect("input_roi");

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], Some(roi_expr.clone()));
        assert_eq!(result[1], Some(roi_expr));
    }
}
