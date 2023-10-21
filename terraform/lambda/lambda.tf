resource "aws_cloudwatch_log_group" "samhq" {
  name = "/aws/lambda/${aws_lambda_function.samhq.function_name}"

  retention_in_days = 7
}

resource "aws_lambda_function" "samhq" {
  function_name = "samhq"
  memory_size   = 10240
  timeout       = 900
  package_type  = "Image"
  architectures = ["x86_64"]
  image_uri     = "${data.terraform_remote_state.ecr.outputs.repository_url_samhq}:${var.image_tag_samhq}"

  ephemeral_storage {
    size = 5000
  }

  environment {
    variables = {
      HOME = "/tmp"
    }
  }

  role = aws_iam_role.lambda.arn
}

resource "aws_lambda_function_url" "samhq" {
  function_name      = aws_lambda_function.samhq.function_name
  authorization_type = "NONE"
}

data "aws_iam_policy_document" "lambda" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "samhq"
  assume_role_policy = data.aws_iam_policy_document.lambda.json
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
  ]
}
