import base64
import json
import tempfile

from image import make_sticker


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode("utf-8")


def base64_to_image(base64_string, output_file_path):
    with open(output_file_path, "wb") as image_file:
        decoded_image = base64.b64decode(base64_string)
        image_file.write(decoded_image)


def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        image = body["image"]
        boxes = body.get("boxes") if body.get("boxes") else None
        points = body.get("points") if body.get("points") else None
        with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
            if not boxes and not points:
                return json.dumps({"image": ""})
            input_path = f"{tmpdirname}/input.jpeg"
            output_path = f"{tmpdirname}/output.png"
            base64_to_image(image, input_path)
            is_segmented = make_sticker(input_path, output_path, boxes, points)
            if is_segmented:
                return json.dumps({"image": image_to_base64(output_path)})
            return json.dumps({"image": ""})

    except Exception as e:
        print(e)
        raise e
