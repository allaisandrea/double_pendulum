#!/bin/bash
# One-time AWS setup for cloud training, run from the Mac:
#
#   bash cloud/setup.sh
#
# Creates the IAM role and instance profile double-pendulum-trainer, which
# the training instances run as: read and write under the bucket's
# double_pendulum/ prefix, read the W&B key's secret, and Systems Manager,
# for a shell on a running instance without SSH. Then stores the W&B API
# key from ~/.netrc as the secret double_pendulum/wandb_api_key. Safe to run
# again: it skips what exists.
set -euo pipefail
P="--profile ${AWS_PROFILE:-andrea-personal}"
ROLE=double-pendulum-trainer
SECRET=double_pendulum/wandb_api_key
BUCKET=allais-andrea-store
ACCOUNT=$(aws $P sts get-caller-identity --query Account --output text)
REGION=$(aws $P configure get region)

if ! aws $P iam get-role --role-name $ROLE > /dev/null 2>&1; then
    aws $P iam create-role --role-name $ROLE --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }' > /dev/null
    echo "created role $ROLE"
fi
aws $P iam put-role-policy --role-name $ROLE --policy-name training --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
        {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\", \"s3:PutObject\", \"s3:DeleteObject\"],
         \"Resource\": \"arn:aws:s3:::$BUCKET/double_pendulum/*\"},
        {\"Effect\": \"Allow\", \"Action\": \"s3:ListBucket\", \"Resource\": \"arn:aws:s3:::$BUCKET\",
         \"Condition\": {\"StringLike\": {\"s3:prefix\": [\"double_pendulum/*\", \"double_pendulum\"]}}},
        {\"Effect\": \"Allow\", \"Action\": \"secretsmanager:GetSecretValue\",
         \"Resource\": \"arn:aws:secretsmanager:$REGION:$ACCOUNT:secret:$SECRET-*\"}
    ]
}"
aws $P iam attach-role-policy --role-name $ROLE --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
if ! aws $P iam get-instance-profile --instance-profile-name $ROLE > /dev/null 2>&1; then
    aws $P iam create-instance-profile --instance-profile-name $ROLE > /dev/null
    aws $P iam add-role-to-instance-profile --instance-profile-name $ROLE --role-name $ROLE
    echo "created instance profile $ROLE"
fi

if ! aws $P secretsmanager describe-secret --secret-id $SECRET > /dev/null 2>&1; then
    KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
    [ -n "$KEY" ] || { echo "no W&B key for api.wandb.ai in ~/.netrc"; exit 1; }
    aws $P secretsmanager create-secret --name $SECRET --secret-string "$KEY" > /dev/null
    echo "stored the W&B key as $SECRET"
fi
echo "ready"
