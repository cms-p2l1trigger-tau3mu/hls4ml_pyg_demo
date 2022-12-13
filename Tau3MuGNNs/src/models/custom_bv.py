"""
debugging BatchNorm1dToQuantScaleBias
"""
from typing import Union, Type, Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from brevitas.function.ops_ste import ceil_ste
from brevitas.function.ops import max_int
from brevitas.quant_tensor import QuantTensor
from brevitas.inject.defaults import Int8WeightPerTensorFloat
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn.quant_layer import WeightQuantType, BiasQuantType, ActQuantType
from abc import ABC
from typing import Optional, Tuple

from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.core.scaling import PowerOfTwoIntScaling
from brevitas.core.restrict_val import PowerOfTwoRestrictValue
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.base import *
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import ScalingImplType, StatsOp, RestrictValueType
from brevitas.core.scaling import ConstScaling

import brevitas.config as config
from brevitas.nn.utils import mul_add_from_bn




class ScaleBias(Module):

    def __init__(self, num_features: int, bias: bool, runtime_shape=(1, -1, 1, 1)):
        super(ScaleBias, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features)) if bias else None
        self.runtime_shape = runtime_shape
        # print(f"ScaleBias runtime_shape: {runtime_shape}")
        # print(f"ScaleBias weight shape: {self.weight.shape}")
        # print(f"ScaleBias bias shape: {self.bias.shape}")
        # print(f"self.weight.view(self.runtime_shape) shape: {self.weight.view(self.runtime_shape).shape}")
        # print(f"self.bias.view(self.runtime_shape) shape: {self.bias.view(self.runtime_shape).shape}")

    def forward(self, input):
        output = input * self.weight.view(self.runtime_shape) + self.bias.view(self.runtime_shape)
        # print(f"ScaleBias output shape: {output.shape}")
        return output


class QuantScaleBias(QuantWBIOL, ScaleBias):

    def __init__(
            self,
            num_features: int,
            bias: bool,
            runtime_shape=(1, -1, 1, 1),
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        ScaleBias.__init__(self, num_features, bias, runtime_shape=runtime_shape)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        # print(f"QuantScaleBias weight_quant bit_width: {weight_quant.bit_width}")
        # print(f"QuantScaleBias bias_quant bit_width: {bias_quant.bit_width}")

    @property
    def per_elem_ops(self):
        return 2

    @property
    def output_channel_dim(self):
        return 0

    @property
    def out_channels(self):
        return self.num_features

    @property
    def channelwise_separable(self) -> bool:
        return True

    def quant_weight(self):
        return self.weight_quant(self.weight.view(-1, 1))  # TODO check if the view is needed

    def forward(self, inp: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        # print("forward")
        return self.forward_impl(inp)

    def inner_forward_impl(self, input: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        # print(f"self.runtime_shape: {self.runtime_shape}")
        quant_weight = quant_weight.view(self.runtime_shape)
        # print(f"quant_weight shape: {quant_weight.shape}")
        quant_bias = quant_bias.view(self.runtime_shape)
        # print(f"quant_bias shape: {quant_bias.shape}")
        output_tensor = input * quant_weight + quant_bias
        # print(f"output_tensor shape: {output_tensor.shape}")
        return output_tensor

    def max_acc_bit_width(self, input_bit_width, weight_bit_width):
        max_input_val = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
        max_weight_val = self.weight_quant.max_uint_value(weight_bit_width)
        max_output_val = max_input_val * max_weight_val
        output_bit_width = ceil_ste(torch.log2(max_output_val))
        return output_bit_width





class _BatchNormToQuantScaleBias(QuantScaleBias, ABC):

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        running_mean_key = prefix + 'running_mean'
        running_var_key = prefix + 'running_var'
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if running_mean_key in state_dict and running_var_key in state_dict:
            weight_init, bias_init = mul_add_from_bn(
                bn_bias=state_dict[bias_key],
                bn_weight=state_dict[weight_key],
                bn_mean=state_dict[running_mean_key],
                bn_var=state_dict[running_var_key],
                bn_eps=self.eps)
            self.weight.data = weight_init
            self.bias.data = bias_init
            del state_dict[bias_key]
            del state_dict[weight_key]
            del state_dict[running_mean_key]
            del state_dict[running_var_key]
            del state_dict[num_batches_tracked_key]
        super(_BatchNormToQuantScaleBias, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and bias_key in missing_keys:
            missing_keys.remove(bias_key)
        if config.IGNORE_MISSING_KEYS and weight_key in missing_keys:
            missing_keys.remove(weight_key)
        if num_batches_tracked_key in unexpected_keys:
            unexpected_keys.remove(num_batches_tracked_key)


class BatchNorm1dToQuantScaleBias(_BatchNormToQuantScaleBias):

    def __init__(
            self,
            num_features,
            eps: float = 1e-5,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        super(BatchNorm1dToQuantScaleBias, self).__init__(
            num_features,
            bias=True,
            # runtime_shape=(1, -1, 1),
            runtime_shape=(1, -1),
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self.eps = eps



"""
conversion function from target ap_fixed to parameters for UAQ
(Uniform Affine Quantization)
"""
def ConvAp_FixedToUAQ(int_bitwidth, fract_bitwidth) -> Tuple[int,float,float]:
  """
  parameters:
  int_bitwidth: int 
  fract_bitwidth: int

  return:
  bitwidth: int
  scale_factor: float
  zero_point: float
  """
  bitwidth = int_bitwidth + fract_bitwidth
  scale_factor = 2**(-fract_bitwidth)
  zero_point = 0 # we assume int representation is signed
  
  return (bitwidth, scale_factor, zero_point)

"""
Reinitialize the custom quantizers for quantized batchnorm seperately
"""


def getCustomQuantizer(int_bitwidth, fract_bitwidth) -> dict:
    # zero point (the third variable) is ignored since
    # it's zero by default
    custom_bit_width, custom_scale, _ = ConvAp_FixedToUAQ(int_bitwidth, fract_bitwidth)
    print(f"scale: {custom_scale}")
    custom_scaling_impl = custom_scale*2**(custom_bit_width-1)

    class PerTensorActPoTScalingCustombit(ExtendedInjector):
        """
        """
        scaling_per_output_channel = False
        restrict_scaling_type = RestrictValueType.POWER_OF_TWO
        bit_width = custom_bit_width
        restrict_value_float_to_int_impl = CeilSte

    class PerTensorWeightPoTScalingCustombit(ExtendedInjector):
        """
        """
        scaling_per_output_channel = False
        restrict_scaling_type = RestrictValueType.POWER_OF_TWO
        bit_width = custom_bit_width
        restrict_value_float_to_int_impl = CeilSte
        scaling_impl = ConstScaling(custom_scaling_impl)

    class CustomMaxScaling(ExtendedInjector):
        """
        """
        scaling_impl_type = ScalingImplType.CONST
        scaling_stats_op = StatsOp.MAX

    class ParamFromRuntimePercentileScaling(ExtendedInjector):
        """
        """
        scaling_impl_type = ScalingImplType.CONST
        scaling_stats_op = StatsOp.PERCENTILE
        percentile_q = 99.999
        collect_stats_steps = 300

    class IntCustomWeightPerTensorFixedPoint(
        NarrowIntQuant, CustomMaxScaling, PerTensorWeightPoTScalingCustombit, WeightQuantSolver):
        """
        8-bit narrow per-tensor signed fixed-point weight quantizer with the radix point
        computed from backpropagated statistics of the weight tensor.

        Examples:
            >>> from brevitas.nn import QuantLinear
            >>> fc = QuantLinear(10, 5, bias=False, weight_quant=Int8WeightPerTensorFixedPoint)
            >>> fc.quant_weight()
        """
        pass


    class IntCustomActPerTensorFixedPoint(
        IntQuant, ParamFromRuntimePercentileScaling, PerTensorActPoTScalingCustombit, ActQuantSolver):
        """
        8-bit per-tensor signed int activations fixed-point quantizer with learned radix point
        initialized from runtime statistics.

        Examples:
            >>> from brevitas.nn import QuantIdentity
            >>> act = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint)
        """
        min_val = -custom_scale*2**(custom_bit_width-1)
        max_val = custom_scale*(2**(custom_bit_width-1)-1)
        print(f"min_val: {min_val}")
        print(f"max_val: {max_val}")


    class IntCustomBiasPerTensorFixedPointInternalScaling(
        IntQuant, MaxStatsScaling, PerTensorWeightPoTScalingCustombit, BiasQuantSolver):
        """
        8-bit per-tensor signed fixed-point bias quantizer with the radix point computed
        from backpropagated statistics of the bias tensor.

        Examples:
            >>> from brevitas.nn import QuantLinear
            >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int8BiasPerTensorFixedPointInternalScaling)
        """
        requires_input_scale = False
        requires_input_bit_width = False

    return_dict = {}
    return_dict["weight"] = IntCustomWeightPerTensorFixedPoint
    return_dict["bias"] = IntCustomBiasPerTensorFixedPointInternalScaling
    return_dict["act"] = IntCustomActPerTensorFixedPoint
    return return_dict