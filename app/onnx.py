from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTOptimizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import numpy as np

################ make file to onnx #################################################
model_checkpoint = "yeombora/kobart_r3f"
save_directory = "./kobart_r3f_onnx/"

# Load a model from transformers and export it to ONNX
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
ort_model = ORTModelForSeq2SeqLM.from_pretrained(model_checkpoint, from_transformers=True)

# Save the ONNX model and tokenizer
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print('model to onnx! finish!')
###################################################################################

################## quantization ###################################################
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

save_directory = "./kobart_r3f_onnx/"

encoder_quantizer = ORTQuantizer.from_pretrained(save_directory, file_name="encoder_model.onnx")
decoder_quantizer = ORTQuantizer.from_pretrained(save_directory, file_name="decoder_model.onnx")
decoder_wp_quantizer = ORTQuantizer.from_pretrained(save_directory, file_name="decoder_with_past_model.onnx")
quantizer = [encoder_quantizer, decoder_quantizer, decoder_wp_quantizer]

# dqconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

for q in quantizer:
    q.quantize(save_dir="./kobart_r3f_onnx_quantization",quantization_config=dqconfig)

print('model quantization! finish!')    
##################################################################################

###################### optimum ###################################################
from optimum.onnxruntime.configuration import OptimizationConfig

print("optimum start!")

path = './kobart_r3f_onnx_quantization/'
model = ORTModelForSeq2SeqLM.from_pretrained(path, use_cache=True)
tokenizer = PreTrainedTokenizerFast.from_pretrained(path, use_fast=True)
optimizer = ORTOptimizer.from_pretrained(model)
save_directory = './kobart_r3f_onnx_quantization_optimizer/'

# Here the optimization level is selected to be 1, enabling basic optimizations such as redundant node eliminations and constant folding. Higher optimization level will result in a hardware dependent optimized graph.
optimization_config = OptimizationConfig(optimization_level=2)

optimizer.optimize(save_dir=save_directory, optimization_config=optimization_config)

print("optimum finish!")
##################################################################################

################################## onnx inference ################################
# path = './kobart_r3f_onnx/'
# summary_model = ORTModelForSeq2SeqLM.from_pretrained(path, use_cache=True)
# summary_tokenizer = PreTrainedTokenizerFast.from_pretrained(path, use_fast=True)
###################################################################################

##################################### quantization inference test #################
# path = './kobart_r3f_onnx_quantization/'
# summary_model = ORTModelForSeq2SeqLM.from_pretrained(path, file_name="decoder_with_past_model_quantized_optimized.onnx", use_cache=True)
# summary_tokenizer = PreTrainedTokenizerFast.from_pretrained(path, use_fast=True)
###################################################################################

############################# optimum inference test ##############################
# path = './kobart_r3f_onnx_quantization_optimizer/'
# summary_model = ORTModelForSeq2SeqLM.from_pretrained(path, use_cache=True)
# summary_tokenizer = PreTrainedTokenizerFast.from_pretrained(path, use_fast=True)
###################################################################################
