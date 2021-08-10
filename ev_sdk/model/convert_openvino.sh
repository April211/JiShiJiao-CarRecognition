source /opt/intel/openvino/deployment_tools/model_optimizer/venv/bin/activate
/opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model /project/train/models/Yolov3_55.onnx --output_dir /project/ev_sdk/model --model_name Yolov3 --batch 1
# /usr/local/ev_sdk/model/
# bash /project/ev_sdk/model/convert_openvino.sh