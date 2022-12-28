# How to create serverless applications using AWS SAM
The AWS Serverless model allows us to easily create and manage resources used in our serverless 
app using AWS cloudFormation.
SAM template: JSON or YAML configuration file that describes Lambda function , API endpoints 
[nifty commands](https://github.com/aws/aws-sam-cli), you upload this template to CloudFormation , which in turn creates all the individual resources and groups them into a CloudFormation stack for ease of management.
updating the SAM template, will redeploy the changes to this stack.
The remainder of this document explains how to write SAM templates and deploy them via AWS CloudFormation.
## Getting started with SAM Template
[aws-sam-cli](https://github.com/aws/aws-sam-cli) to get started
```linux
$ sam init --runtime python3.7

