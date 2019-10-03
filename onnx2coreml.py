import sys
from onnx import onnx_pb
from onnx_coreml import convert
import coremltools

model_in = sys.argv[1]
model_out = sys.argv[2]

model_file = open(model_in, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
coreml_model = convert(model_proto,
                       preprocessing_args={'red_bias': 127, 'blue_bias': 127, 'green_bias': 127, 'scale': 128},
                       image_input_names=['image']) #image_output_names=['scores', 'boxes'])
coreml_model.save(model_out)