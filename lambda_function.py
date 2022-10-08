import json
from run import model_predict_masks, model_predict_classes 
import numpy as np
import boto3


def lambda_handler(event, context):
    model_name = event["model_name"]
    image = np.asarray(event["image"], dtype=np.float32)

    S3 = boto3.client("s3")
    bucket = 'ml-web-app-models'
    key = model_name
    obj = S3.get_object(Bucket=bucket, Key=key)
    response = obj['Body'].read()
    
    if key == "SegmentationModel.onnx":
        masks = model_predict_masks(response, image)
        return {"statusCode": 200, "body": json.dumps(masks)}
    elif key == "classificator.onnx":
        labels = model_predict_classes(response, image)
        return {"statusCode": 200, "body": json.dumps(labels)}

