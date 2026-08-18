use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use tract_core::internal::*;

pub type DispatchPowConstFn = fn(f32, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

#[derive(Clone, new)]
pub struct GpuPowConst {
    pub exponent: f32,
    pub backend_name: &'static str,
    pub dispatch: DispatchPowConstFn,
}

impl std::fmt::Debug for GpuPowConst {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}PowConst(exponent: {})", self.backend_name, self.exponent)
    }
}

impl PartialEq for GpuPowConst {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.exponent == other.exponent
    }
}

impl Eq for GpuPowConst {}

impl std::hash::Hash for GpuPowConst {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.exponent.to_bits().hash(state);
    }
}

impl Op for GpuPowConst {
    fn name(&self) -> StaticName {
        format!("{}PowConst", self.backend_name).into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuPowConst {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            input.shape(),
        )?;
        (self.dispatch)(self.exponent, input, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuPowConst {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let dt = facts[0].datum_type;
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
