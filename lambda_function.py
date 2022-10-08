import json
from run import model_predict_masks
import numpy as np
import boto3


def lambda_handler(event, context):
    model_name = event["model_name"]
    image = np.asarray(event["image"], dtype=np.float32)

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket='your-bucket-with-models', Key=model_name)
    response = obj['Body'].read()
    
    if model_name == "SegmentationModel.onnx":
        masks = model_predict_masks(response, image)

        return {"statusCode": 200, "body": json.dumps(masks)}

