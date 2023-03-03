from onnxsim import simplify
import onnx
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--output", required=True, help="output model")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    onnx_model = onnx.load(args.input)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.output)
    print('finished exporting onnxsim')
