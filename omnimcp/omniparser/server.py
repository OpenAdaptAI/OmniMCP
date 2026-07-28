# omnimcp/omniparser/server.py

"""Deployment module for OmniParser on AWS EC2 with on-demand startup and ALARM-BASED auto-shutdown."""

import datetime
import io
import json
import os
import subprocess
import time
import zipfile
from collections.abc import Sequence

import boto3
import fire
import paramiko
from botocore.exceptions import ClientError
from loguru import logger

# Assuming config is imported correctly from omnimcp.config
from omnimcp.config import config

# Constants for AWS resource names
LAMBDA_FUNCTION_NAME = f"{config.PROJECT_NAME}-auto-shutdown"
IAM_ROLE_NAME = (
    f"{config.PROJECT_NAME}-lambda-role"  # Role for the auto-shutdown Lambda
)

CLEANUP_ON_FAILURE = False  # Set to True to attempt cleanup even if start fails


def create_key_pair(
    key_name: str = config.AWS_EC2_KEY_NAME, key_path: str = config.AWS_EC2_KEY_PATH
) -> str | None:
    """Create an EC2 key pair."""
    ec2_client = boto3.client("ec2", region_name=config.AWS_REGION)
    try:
        logger.info(f"Attempting to create key pair: {key_name}")
        key_pair = ec2_client.create_key_pair(KeyName=key_name)
        private_key = key_pair["KeyMaterial"]

        # Ensure directory exists if key_path includes directories
        os.makedirs(os.path.dirname(key_path), exist_ok=True)

        with open(key_path, "w") as key_file:
            key_file.write(private_key)
        os.chmod(key_path, 0o400)  # Set read-only permissions

        logger.info(f"Key pair {key_name} created and saved to {key_path}")
        return key_name
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidKeyPair.Duplicate":
            logger.warning(
                f"Key pair '{key_name}' already exists in AWS. Attempting to delete and recreate."
            )
            try:
                ec2_client.delete_key_pair(KeyName=key_name)
                logger.info(f"Deleted existing key pair '{key_name}' from AWS.")
                # Retry creation
                return create_key_pair(key_name, key_path)
            except ClientError as e_del:
                logger.error(
                    f"Failed to delete existing key pair '{key_name}': {e_del}"
                )
                return None
        else:
            logger.error(f"Error creating key pair {key_name}: {e}")
            return None


# Evaluated once at import, exactly as the old mutable default argument was,
# but as an immutable tuple so a caller cannot mutate every later call's default.
_DEFAULT_SECURITY_GROUP_PORTS: tuple[int, ...] = (22, config.PORT)


def get_or_create_security_group_id(
    ports: Sequence[int] | None = None,
) -> str | None:
    """Get existing security group or create a new one."""
    if ports is None:
        ports = _DEFAULT_SECURITY_GROUP_PORTS
    ec2_client = boto3.client("ec2", region_name=config.AWS_REGION)
    sg_name = config.AWS_EC2_SECURITY_GROUP

    ip_permissions = [
        {
            "IpProtocol": "tcp",
            "FromPort": port,
            "ToPort": port,
            "IpRanges": [
                {"CidrIp": "0.0.0.0/0"}
            ],  # Allows access from any IP, adjust if needed
        }
        for port in ports
    ]

    try:
        response = ec2_client.describe_security_groups(GroupNames=[sg_name])
        security_group_id = response["SecurityGroups"][0]["GroupId"]
        logger.info(f"Security group '{sg_name}' already exists: {security_group_id}")

        # Ensure desired rules exist (idempotent check)
        existing_permissions = response["SecurityGroups"][0].get("IpPermissions", [])
        current_ports_open = set()
        for perm in existing_permissions:
            if perm.get("IpProtocol") == "tcp" and any(
                ip_range == {"CidrIp": "0.0.0.0/0"}
                for ip_range in perm.get("IpRanges", [])
            ):
                current_ports_open.add(perm.get("FromPort"))

        for required_perm in ip_permissions:
            port_to_open = required_perm["FromPort"]
            if port_to_open not in current_ports_open:
                try:
                    logger.info(
                        f"Attempting to add inbound rule for port {port_to_open}..."
                    )
                    ec2_client.authorize_security_group_ingress(
                        GroupId=security_group_id, IpPermissions=[required_perm]
                    )
                    logger.info(f"Added inbound rule for port {port_to_open}")
                except ClientError as e_auth:
                    # Handle race condition or other errors
                    if (
                        e_auth.response["Error"]["Code"]
                        == "InvalidPermission.Duplicate"
                    ):
                        logger.info(
                            f"Rule for port {port_to_open} likely added concurrently or already exists."
                        )
                    else:
                        logger.error(
                            f"Error adding rule for port {port_to_open}: {e_auth}"
                        )
            else:
                logger.info(f"Rule for port {port_to_open} already exists.")

        return security_group_id

    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidGroup.NotFound":
            logger.info(f"Security group '{sg_name}' not found. Creating...")
            try:
                response = ec2_client.create_security_group(
                    GroupName=sg_name,
                    Description=f"Security group for {config.PROJECT_NAME} deployment",
                    TagSpecifications=[
                        {
                            "ResourceType": "security-group",
                            "Tags": [{"Key": "Name", "Value": config.PROJECT_NAME}],
                        }
                    ],
                )
                security_group_id = response["GroupId"]
                logger.info(
                    f"Created security group '{sg_name}' with ID: {security_group_id}"
                )

                # Add rules after creation
                time.sleep(5)  # Brief wait for SG propagation
                ec2_client.authorize_security_group_ingress(
                    GroupId=security_group_id, IpPermissions=ip_permissions
                )
                logger.info(f"Added inbound rules for ports {ports}")
                return security_group_id
            except ClientError as e_create:
                logger.error(f"Error creating security group '{sg_name}': {e_create}")
                return None
        else:
            logger.error(f"Error describing security group '{sg_name}': {e}")
            return None


def deploy_ec2_instance(
    ami: str = config.AWS_EC2_AMI,
    instance_type: str = config.AWS_EC2_INSTANCE_TYPE,
    project_name: str = config.PROJECT_NAME,
    key_name: str = config.AWS_EC2_KEY_NAME,
    disk_size: int = config.AWS_EC2_DISK_SIZE,
) -> tuple[str | None, str | None]:
    """
    Deploy a new EC2 instance or start/return an existing usable one.
    Ignores instances that are shutting-down or terminated.

    Args:
        ami: AMI ID to use for the instance.
        instance_type: EC2 instance type.
        project_name: Name tag for the instance.
        key_name: Name of the key pair to use.
        disk_size: Size of the root volume in GB.

    Returns:
        Tuple[str | None, str | None]: Instance ID and public IP if successful, otherwise (None, None).
    """
    ec2 = boto3.resource("ec2", region_name=config.AWS_REGION)
    ec2_client = boto3.client("ec2", region_name=config.AWS_REGION)
    key_path = config.AWS_EC2_KEY_PATH  # Local path for the key

    instance_id = None
    instance_ip = None
    usable_instance_found = False

    try:
        logger.info(
            f"Checking for existing usable EC2 instance tagged: Name={project_name}"
        )
        # Filter for states we can potentially reuse or wait for
        instances = ec2.instances.filter(
            Filters=[
                {"Name": "tag:Name", "Values": [project_name]},
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopped"],
                },
            ]
        )

        # Find the most recently launched instance in a usable state
        sorted_instances = sorted(instances, key=lambda i: i.launch_time, reverse=True)

        if sorted_instances:
            candidate_instance = sorted_instances[0]
            instance_id = candidate_instance.id
            state = candidate_instance.state["Name"]
            logger.info(
                f"Found most recent potentially usable instance {instance_id} in state: {state}"
            )

            # Check if local key file exists before trying to use/start instance
            if not os.path.exists(key_path):
                logger.error(
                    f"Local SSH key file {key_path} not found for existing instance {instance_id}."
                )
                logger.error(
                    "Cannot proceed with existing instance without the key. Will attempt to create a new instance."
                )
                # Force creation of a new instance by setting usable_instance_found to False
                usable_instance_found = False
                # Reset instance_id/ip as we cannot use this one
                instance_id = None
                instance_ip = None
            else:
                # Key exists, proceed with state handling
                if state == "running":
                    instance_ip = candidate_instance.public_ip_address
                    if not instance_ip:
                        logger.warning(
                            f"Instance {instance_id} is running but has no public IP. Waiting briefly..."
                        )
                        try:
                            # Short wait, maybe IP assignment is delayed
                            waiter = ec2_client.get_waiter("instance_running")
                            waiter.wait(
                                InstanceIds=[instance_id],
                                WaiterConfig={"Delay": 5, "MaxAttempts": 6},
                            )  # Wait up to 30s
                            candidate_instance.reload()
                            instance_ip = candidate_instance.public_ip_address
                            if not instance_ip:
                                raise RuntimeError(
                                    "Instance running but failed to get Public IP."
                                )
                            logger.info(
                                f"Successfully obtained Public IP for running instance: {instance_ip}"
                            )
                            usable_instance_found = True
                        except Exception as e_wait_ip:
                            logger.error(
                                f"Failed to get Public IP for running instance {instance_id}: {e_wait_ip}"
                            )
                            # Fall through to create new instance
                    else:
                        logger.info(
                            f"Reusing running instance: ID={instance_id}, IP={instance_ip}"
                        )
                        usable_instance_found = True

                elif state == "stopped":
                    logger.info(
                        f"Attempting to start existing stopped instance: ID={instance_id}"
                    )
                    try:
                        ec2_client.start_instances(InstanceIds=[instance_id])
                        waiter = ec2_client.get_waiter("instance_running")
                        logger.info("Waiting for instance to reach 'running' state...")
                        waiter.wait(
                            InstanceIds=[instance_id],
                            WaiterConfig={"Delay": 15, "MaxAttempts": 40},
                        )  # Standard wait
                        candidate_instance.reload()
                        instance_ip = candidate_instance.public_ip_address
                        if not instance_ip:
                            raise RuntimeError(
                                f"Instance {instance_id} started but has no public IP."
                            )
                        logger.info(
                            f"Instance started successfully: ID={instance_id}, IP={instance_ip}"
                        )
                        usable_instance_found = True
                    except Exception as e_start:
                        logger.error(
                            f"Failed to start or wait for stopped instance {instance_id}: {e_start}"
                        )
                        # Fall through to create new instance

                elif state == "pending":
                    logger.info(
                        f"Instance {instance_id} is pending. Waiting until running..."
                    )
                    try:
                        waiter = ec2_client.get_waiter("instance_running")
                        waiter.wait(
                            InstanceIds=[instance_id],
                            WaiterConfig={"Delay": 15, "MaxAttempts": 40},
                        )  # Standard wait
                        candidate_instance.reload()
                        instance_ip = candidate_instance.public_ip_address
                        if not instance_ip:
                            raise RuntimeError(
                                "Instance reached running state but has no public IP"
                            )
                        logger.info(
                            f"Instance now running: ID={instance_id}, IP={instance_ip}"
                        )
                        usable_instance_found = True
                    except Exception as e_wait:
                        logger.error(
                            f"Error waiting for pending instance {instance_id}: {e_wait}"
                        )
                        # Fall through to create new instance

        # --- If usable instance found and prepared, return its details ---
        if usable_instance_found and instance_id and instance_ip:
            logger.info(f"Using existing/started instance {instance_id}")
            return instance_id, instance_ip

        # --- No usable existing instance found, proceed to create a new one ---
        logger.info(
            "No usable existing instance found or prepared. Creating a new instance..."
        )
        instance_id = None  # Reset in case candidate failed
        instance_ip = None

        security_group_id = get_or_create_security_group_id()
        if not security_group_id:
            logger.error("Unable to get/create security group ID. Aborting deployment.")
            return None, None

        # Create new key pair (delete old local file and AWS key pair first)
        try:
            key_name_to_use = key_name  # Use function arg or config default
            if os.path.exists(key_path):
                logger.info(f"Removing existing local key file {key_path}")
                os.remove(key_path)
            try:
                logger.info(
                    f"Attempting to delete key pair '{key_name_to_use}' from AWS (if exists)..."
                )
                ec2_client.delete_key_pair(KeyName=key_name_to_use)
                logger.info(f"Deleted existing key pair '{key_name_to_use}' from AWS.")
            except ClientError as e:
                # Ignore if key not found, log other errors
                if e.response["Error"]["Code"] != "InvalidKeyPair.NotFound":
                    logger.warning(
                        f"Could not delete key pair '{key_name_to_use}' from AWS: {e}"
                    )
                else:
                    logger.info(f"Key pair '{key_name_to_use}' not found in AWS.")
            # Create the new key pair
            if not create_key_pair(key_name_to_use, key_path):
                raise RuntimeError("Failed to create new key pair")
        except Exception as e:
            logger.error(f"Error managing key pair: {e}")
            return None, None

        # Create new EC2 instance
        try:
            ebs_config = {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": disk_size,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                    "Iops": 3000,
                    "Throughput": 125,
                },
            }
            logger.info(
                f"Launching new EC2 instance (AMI: {ami}, Type: {instance_type})..."
            )
            new_instance_resource = ec2.create_instances(
                ImageId=ami,
                MinCount=1,
                MaxCount=1,
                InstanceType=instance_type,
                KeyName=key_name_to_use,
                SecurityGroupIds=[security_group_id],
                BlockDeviceMappings=[ebs_config],
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [{"Key": "Name", "Value": project_name}],
                    },
                    {
                        "ResourceType": "volume",
                        "Tags": [{"Key": "Name", "Value": f"{project_name}-root-vol"}],
                    },
                ],
            )[0]

            instance_id = new_instance_resource.id
            logger.info(f"New instance {instance_id} created. Waiting until running...")
            new_instance_resource.wait_until_running(
                WaiterConfig={"Delay": 15, "MaxAttempts": 40}
            )
            new_instance_resource.reload()
            instance_ip = new_instance_resource.public_ip_address
            if not instance_ip:
                raise RuntimeError(
                    f"Instance {instance_id} started but has no public IP."
                )
            logger.info(f"New instance running: ID={instance_id}, IP={instance_ip}")
            return instance_id, instance_ip  # Return new instance details
        except Exception as e:
            logger.error(f"Failed to create or wait for new EC2 instance: {e}")
            if instance_id:  # If instance was created but failed later
                try:
                    logger.warning(
                        f"Attempting to terminate partially created/failed instance {instance_id}"
                    )
                    ec2_client.terminate_instances(InstanceIds=[instance_id])
                    logger.info(f"Issued terminate for {instance_id}")
                except Exception as term_e:
                    logger.error(
                        f"Failed to terminate failed instance {instance_id}: {term_e}"
                    )
            return None, None  # Return failure

    except Exception as outer_e:
        # Catch any unexpected errors in the overall logic
        logger.error(
            f"Unexpected error during instance deployment/discovery: {outer_e}",
            exc_info=True,
        )
        return None, None


# TODO: Wait for Unattended Upgrades: Add an explicit wait or a loop checking
# for the lock file (/var/lib/dpkg/lock-frontend) before running apt-get
# install. E.g., while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1;
# do echo 'Waiting for apt lock...'; sleep 10; done. This is more robust.


def configure_ec2_instance(
    instance_id: str,
    instance_ip: str,
    max_ssh_retries: int = 20,
    ssh_retry_delay: int = 20,
    max_cmd_retries: int = 20,
    cmd_retry_delay: int = 20,
) -> bool:
    """Configure the specified EC2 instance (install Docker, etc.)."""

    logger.info(f"Starting configuration for instance {instance_id} at {instance_ip}")
    try:
        key_path = config.AWS_EC2_KEY_PATH
        if not os.path.exists(key_path):
            logger.error(
                f"Key file not found at {key_path}. Cannot configure instance."
            )
            return False
        key = paramiko.RSAKey.from_private_key_file(key_path)
    except Exception as e:
        logger.error(f"Failed to load SSH key {key_path}: {e}")
        return False

    ssh_client = None  # Initialize to None
    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # --- SSH Connection Logic ---
        logger.info("Attempting SSH connection...")
        ssh_retries = 0
        while ssh_retries < max_ssh_retries:
            try:
                ssh_client.connect(
                    hostname=instance_ip,
                    username=config.AWS_EC2_USER,
                    pkey=key,
                    timeout=20,
                )
                logger.success("SSH connection established.")
                break  # Exit loop on success
            except Exception as e:
                ssh_retries += 1
                logger.warning(
                    f"SSH connection attempt {ssh_retries}/{max_ssh_retries} failed: {e}"
                )
                if ssh_retries < max_ssh_retries:
                    logger.info(
                        f"Retrying SSH connection in {ssh_retry_delay} seconds..."
                    )
                    time.sleep(ssh_retry_delay)
                else:
                    logger.error(
                        "Maximum SSH connection attempts reached. Configuration aborted."
                    )
                    return False  # Return failure

        # --- Instance Setup Commands ---
        commands = [
            "sudo apt-get update -y",
            "sudo apt-get install -y ca-certificates curl gnupg apt-transport-https",  # Ensure https transport
            "sudo install -m 0755 -d /etc/apt/keyrings",
            # Use non-deprecated method for adding Docker GPG key with non-interactive flags
            "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --batch --yes -o /etc/apt/keyrings/docker.gpg",
            "sudo chmod a+r /etc/apt/keyrings/docker.gpg",
            (  # Use lsb_release for codename reliably
                'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] '
                'https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | '
                "sudo tee /etc/apt/sources.list.d/docker.list > /dev/null"
            ),
            "sudo apt-get update -y",
            # Install specific components needed
            "sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin",
            "sudo systemctl start docker",
            "sudo systemctl enable docker",
            # Add user to docker group - requires new login/session to take effect for user directly, but sudo works
            f"sudo usermod -aG docker {config.AWS_EC2_USER}",
        ]

        for command in commands:
            # logger.info(f"Executing: {command}") # execute_command already logs
            # Use execute_command helper for better output handling and retries
            execute_command(
                ssh_client,
                command,
                max_retries=max_cmd_retries,
                retry_delay=cmd_retry_delay,
            )
        logger.success("Instance OS configuration commands completed.")
        return True  # Configuration successful

    except Exception as e:
        logger.error(f"Failed during instance configuration: {e}", exc_info=True)
        return False  # Configuration failed
    finally:
        if ssh_client:
            ssh_client.close()
            logger.info("SSH connection closed during configure_ec2_instance.")


def execute_command(
    ssh_client: paramiko.SSHClient,
    command: str,
    max_retries: int = 20,
    retry_delay: int = 10,
    timeout: int = config.COMMAND_TIMEOUT,  # Use timeout from config
) -> tuple[int, str, str]:  # Return status, stdout, stderr
    """Execute a command via SSH with retries for specific errors."""
    logger.info(
        f"Executing SSH command: {command[:100]}{'...' if len(command) > 100 else ''}"
    )
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            _stdin, stdout, stderr = ssh_client.exec_command(
                command,
                timeout=timeout,
                get_pty=False,  # Try without PTY first
            )
            # It's crucial to wait for the command to finish *before* reading streams fully
            exit_status = stdout.channel.recv_exit_status()

            # Read output streams completely after command exit
            stdout_output = stdout.read().decode("utf-8", errors="replace").strip()
            stderr_output = stderr.read().decode("utf-8", errors="replace").strip()

            if stdout_output:
                logger.debug(f"STDOUT:\n{stdout_output}")
            if stderr_output:
                if exit_status == 0:
                    logger.warning(f"STDERR (Exit Status 0):\n{stderr_output}")
                else:
                    logger.error(
                        f"STDERR (Exit Status {exit_status}):\n{stderr_output}"
                    )

            # Check exit status and potential retry conditions
            if exit_status == 0:
                logger.success(
                    f"Command successful (attempt {attempt}): {command[:50]}..."
                )
                return exit_status, stdout_output, stderr_output  # Success

            # Specific Retry Condition: dpkg lock
            if (
                "Could not get lock" in stderr_output
                or "dpkg frontend is locked" in stderr_output
            ):
                logger.warning(
                    f"Command failed due to dpkg lock (attempt {attempt}/{max_retries}). Retrying in {retry_delay}s..."
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue  # Go to next attempt
                else:
                    # Max retries reached for lock
                    error_msg = f"Command failed after {max_retries} attempts due to dpkg lock: {command}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)  # Final failure after retries
            else:
                # Other non-zero exit status, fail immediately
                error_msg = f"Command failed with exit status {exit_status} (attempt {attempt}): {command}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)  # Final failure

        except Exception as e:
            # Catch other potential errors like timeouts
            logger.error(f"Exception during command execution (attempt {attempt}): {e}")
            if attempt < max_retries:
                logger.info(f"Retrying command after exception in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Command failed after {max_retries} attempts due to exception: {command}"
                )
                raise  # Reraise the last exception

    # This line should not be reachable if logic is correct
    raise RuntimeError(f"Command failed after exhausting retries: {command}")


def create_auto_shutdown_infrastructure(instance_id: str) -> None:
    """
    Create CloudWatch Alarm and Lambda function for CPU inactivity based auto-shutdown,
    including granting necessary permissions.
    """
    # Initialize necessary clients
    lambda_client = boto3.client("lambda", region_name=config.AWS_REGION)
    iam_client = boto3.client("iam", region_name=config.AWS_REGION)
    cloudwatch_client = boto3.client("cloudwatch", region_name=config.AWS_REGION)
    sts_client = boto3.client(
        "sts", region_name=config.AWS_REGION
    )  # Needed for Account ID

    # Use constants defined at module level
    role_name = IAM_ROLE_NAME
    lambda_function_name = LAMBDA_FUNCTION_NAME
    alarm_name = f"{config.PROJECT_NAME}-CPU-Low-Alarm-{instance_id}"  # Unique alarm name per instance

    logger.info("Setting up auto-shutdown infrastructure (Alarm-based)...")

    # --- Create or Get IAM Role ---
    role_arn = None
    try:
        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        logger.info(f"Attempting to create/get IAM role: {role_name}")
        try:
            response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
            )
            role_arn = response["Role"]["Arn"]
            logger.info(f"Created IAM role {role_name}. Attaching policies...")
            # Attach policies needed by Lambda
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            )
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonEC2ReadOnlyAccess",
            )
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonEC2FullAccess",
            )  # Consider reducing scope later
            logger.info(f"Attached policies to IAM role {role_name}")
            logger.info("Waiting for IAM role propagation...")
            time.sleep(15)  # Increased wait time for IAM propagation
        except ClientError as e:
            if e.response["Error"]["Code"] == "EntityAlreadyExists":
                logger.info(f"IAM role {role_name} already exists, retrieving ARN...")
                response = iam_client.get_role(RoleName=role_name)
                role_arn = response["Role"]["Arn"]
                # Optional: Add logic here to verify/attach required policies if the role already existed
            else:
                raise  # Reraise other creation errors
    except Exception as e:
        logger.error(f"Failed to create/get IAM role {role_name}: {e}")
        logger.error("Cannot proceed with auto-shutdown setup without IAM role.")
        return  # Stop setup

    if not role_arn:
        logger.error("Failed to obtain IAM role ARN. Aborting auto-shutdown setup.")
        return

    # --- Define Updated Lambda Function Code ---
    # (Contains fix to remove AWS_REGION env var usage and rely on default boto3 region)
    lambda_code = """
import boto3
import os
import json

INSTANCE_ID = os.environ.get('INSTANCE_ID')
# AWS_REGION = os.environ.get('AWS_REGION') # No longer needed

print(f"Lambda invoked. Checking instance: {INSTANCE_ID}") # Removed region here

def lambda_handler(event, context):
    if not INSTANCE_ID:
        print("Error: INSTANCE_ID environment variable not set.")
        return {'statusCode': 500, 'body': json.dumps('Configuration error')}

    # boto3 automatically uses the Lambda execution region if not specified
    ec2 = boto3.client('ec2') # Removed region_name
    print(f"Inactivity Alarm triggered for instance: {INSTANCE_ID}. Checking state...")

    try:
        response = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
        if not response.get('Reservations') or not response['Reservations'][0].get('Instances'):
             print(f"Instance {INSTANCE_ID} not found (already terminated?). No action needed.")
             return {'statusCode': 404, 'body': json.dumps('Instance not found')}

        instance_data = response['Reservations'][0]['Instances'][0]
        state = instance_data['State']['Name']

        if state == 'running':
            print(f"Instance {INSTANCE_ID} is running. Stopping due to inactivity alarm.")
            try:
                 ec2.stop_instances(InstanceIds=[INSTANCE_ID])
                 print(f"Stop command issued for {INSTANCE_ID}.")
                 return {'statusCode': 200, 'body': json.dumps('Instance stop initiated')}
            except Exception as stop_err:
                 print(f"Failed to issue stop command for {INSTANCE_ID}: {str(stop_err)}")
                 return {'statusCode': 500, 'body': json.dumps(f'Failed to stop instance: {str(stop_err)}')}
        else:
            print(f"Instance {INSTANCE_ID} is already in state '{state}'. No action taken.")
            return {'statusCode': 200, 'body': json.dumps('Instance not running, no action')}
    except Exception as e:
        print(f"Error interacting with EC2 for instance {INSTANCE_ID}: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps(f'Error: {str(e)}')}
"""

    # --- Create or Update Lambda Function ---
    lambda_arn = None
    try:
        logger.info(f"Preparing Lambda function code for {lambda_function_name}...")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("lambda_function.py", lambda_code.encode("utf-8"))
        zip_content = zip_buffer.getvalue()

        env_vars = {"Variables": {"INSTANCE_ID": instance_id}}  # Only pass instance ID

        try:
            logger.info(
                f"Checking for existing Lambda function: {lambda_function_name}"
            )
            func_config = lambda_client.get_function_configuration(
                FunctionName=lambda_function_name
            )
            lambda_arn = func_config["FunctionArn"]
            logger.info("Found existing Lambda. Updating code and configuration...")
            lambda_client.update_function_code(
                FunctionName=lambda_function_name, ZipFile=zip_content
            )
            # Add waiter after code update
            logger.info(
                f"Waiting for Lambda function code update on {lambda_function_name} to complete..."
            )
            waiter_update = lambda_client.get_waiter("function_updated_v2")
            waiter_update.wait(
                FunctionName=lambda_function_name,
                WaiterConfig={"Delay": 5, "MaxAttempts": 12},
            )  # Wait up to 60s
            logger.info("Lambda function code update complete.")
            # Now update configuration
            lambda_client.update_function_configuration(
                FunctionName=lambda_function_name,
                Role=role_arn,
                Environment=env_vars,
                Timeout=30,
                MemorySize=128,
            )
            logger.info(
                f"Updated Lambda function configuration: {lambda_function_name}"
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.info(
                    f"Lambda function {lambda_function_name} not found. Creating..."
                )
                response = lambda_client.create_function(
                    FunctionName=lambda_function_name,
                    Runtime="python3.9",
                    Role=role_arn,
                    Handler="lambda_function.lambda_handler",
                    Code={"ZipFile": zip_content},
                    Timeout=30,
                    MemorySize=128,
                    Description=f"Auto-shutdown function for {config.PROJECT_NAME} instance {instance_id}",
                    Environment=env_vars,
                    Tags={"Project": config.PROJECT_NAME},
                )
                lambda_arn = response["FunctionArn"]
                logger.info(f"Created Lambda function: {lambda_arn}")
                logger.info("Waiting for Lambda function to become active...")
                waiter_create = lambda_client.get_waiter("function_active_v2")
                waiter_create.wait(
                    FunctionName=lambda_function_name,
                    WaiterConfig={"Delay": 2, "MaxAttempts": 15},
                )
                logger.info("Lambda function is active.")
            else:
                raise  # Reraise other ClientErrors

        if not lambda_arn:
            raise RuntimeError("Failed to get Lambda Function ARN after create/update.")

        # --- Remove Old CloudWatch Events Rule and Permissions (Idempotent) ---
        try:
            events_client = boto3.client("events", region_name=config.AWS_REGION)
            old_rule_name = f"{config.PROJECT_NAME}-inactivity-monitor"
            logger.info(
                f"Attempting to cleanup old Event rule/targets for: {old_rule_name}"
            )
            try:
                events_client.remove_targets(Rule=old_rule_name, Ids=["1"], Force=True)
            except ClientError as e_rem:
                logger.debug(f"Ignoring error removing targets: {e_rem}")
            try:
                events_client.delete_rule(Name=old_rule_name)
            except ClientError as e_del:
                logger.debug(f"Ignoring error deleting rule: {e_del}")
            logger.info(
                f"Cleaned up old CloudWatch Events rule: {old_rule_name} (if it existed)"
            )
        except Exception as e_ev_clean:
            logger.warning(f"Issue during old Event rule cleanup: {e_ev_clean}")
        try:
            logger.info(
                "Attempting to remove old CloudWatch Events Lambda permission..."
            )
            lambda_client.remove_permission(
                FunctionName=lambda_function_name,
                StatementId=f"{config.PROJECT_NAME}-cloudwatch-trigger",
            )  # Old Statement ID
            logger.info("Removed old CloudWatch Events permission from Lambda.")
        except ClientError as e_perm:
            if e_perm.response["Error"]["Code"] != "ResourceNotFoundException":
                logger.warning(f"Could not remove old Lambda permission: {e_perm}")
            else:
                logger.info("Old Lambda permission not found.")

        # --- Create New CloudWatch Alarm ---
        evaluation_periods = max(1, config.INACTIVITY_TIMEOUT_MINUTES // 5)
        threshold_cpu = 5.0
        logger.info(
            f"Setting up CloudWatch alarm '{alarm_name}' for CPU < {threshold_cpu}% over {evaluation_periods * 5} minutes."
        )
        alarm_arn = None  # Initialize alarm ARN
        try:
            # Delete existing alarm first for idempotency
            try:
                cloudwatch_client.delete_alarms(AlarmNames=[alarm_name])
                logger.info(
                    f"Deleted potentially existing CloudWatch alarm: {alarm_name}"
                )
            except ClientError as e:
                if e.response["Error"]["Code"] != "ResourceNotFound":
                    logger.warning(
                        f"Could not delete existing alarm {alarm_name} before creation: {e}"
                    )

            # Get Account ID for constructing Alarm ARN
            try:
                account_id = sts_client.get_caller_identity()["Account"]
                # Construct the ARN - verify region and partition if needed (assuming aws standard)
                alarm_arn = f"arn:aws:cloudwatch:{config.AWS_REGION}:{account_id}:alarm:{alarm_name}"
                logger.debug(f"Constructed Alarm ARN: {alarm_arn}")
            except Exception as sts_e:
                logger.error(
                    f"Could not get AWS Account ID via STS: {sts_e}. Cannot set Lambda permission."
                )
                # Proceed without setting permission if ARN cannot be constructed

            cloudwatch_client.put_metric_alarm(
                AlarmName=alarm_name,
                AlarmDescription=f"Stop EC2 instance {instance_id} if avg CPU < {threshold_cpu}% for {evaluation_periods * 5} mins",
                ActionsEnabled=True,
                AlarmActions=[lambda_arn],  # Trigger Lambda function ARN
                MetricName="CPUUtilization",
                Namespace="AWS/EC2",
                Statistic="Average",
                Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                Period=300,
                EvaluationPeriods=evaluation_periods,
                Threshold=threshold_cpu,
                ComparisonOperator="LessThanThreshold",
                TreatMissingData="breaching",
                Tags=[{"Key": "Project", "Value": config.PROJECT_NAME}],
            )
            logger.info(
                f"Created/Updated CloudWatch Alarm '{alarm_name}' triggering Lambda on low CPU."
            )

            # --- *** ADD LAMBDA PERMISSION FOR ALARM *** ---
            if alarm_arn and lambda_arn:  # Only proceed if we have both ARNs
                statement_id = (
                    f"AllowExecutionFromCloudWatchAlarm_{alarm_name}"  # Unique ID
                )
                logger.info(
                    f"Attempting to grant invoke permission to Lambda {lambda_function_name} from Alarm {alarm_name}"
                )
                try:
                    # Remove potentially existing permission with same ID first
                    try:
                        lambda_client.remove_permission(
                            FunctionName=lambda_function_name, StatementId=statement_id
                        )
                        logger.info(
                            f"Removed existing permission statement '{statement_id}' before adding new one."
                        )
                    except ClientError as e:
                        if e.response["Error"]["Code"] != "ResourceNotFoundException":
                            raise  # Reraise unexpected error

                    # Add permission for the CloudWatch Alarm service to invoke this Lambda
                    lambda_client.add_permission(
                        FunctionName=lambda_function_name,
                        StatementId=statement_id,
                        Action="lambda:InvokeFunction",
                        Principal="cloudwatch.amazonaws.com",  # Correct principal for CW Alarms
                        SourceArn=alarm_arn,  # ARN of the specific CloudWatch Alarm
                    )
                    logger.success(
                        f"Granted CloudWatch Alarm ({alarm_name}) permission to invoke Lambda ({lambda_function_name})."
                    )
                except ClientError as e:
                    if e.response["Error"]["Code"] == "ResourceConflictException":
                        logger.warning(
                            f"Lambda permission statement '{statement_id}' may already exist or a conflict occurred."
                        )
                    else:
                        logger.error(
                            f"Failed to add Lambda permission for CloudWatch Alarm: {e}"
                        )
                        # Log but maybe don't fail deployment? Auto-shutdown just won't work.
            else:
                logger.error(
                    "Skipping Lambda permission setup because Alarm ARN or Lambda ARN could not be determined."
                )
            # --- *** END PERMISSION FIX *** ---

        except Exception as e:
            logger.error(
                f"Failed to create/update CloudWatch alarm or set permissions: {e}"
            )

        logger.success(
            f"Auto-shutdown infrastructure setup completed for {instance_id=}"
        )

    except Exception as e:
        logger.error(
            f"Error setting up auto-shutdown infrastructure: {e}", exc_info=True
        )
        # Allow deployment to continue but log the failure


class Deploy:
    """Class handling deployment operations for OmniParser."""

    @staticmethod
    def start() -> tuple[str | None, str | None]:  # Added return type hint
        """
        Start or configure EC2 instance, setup auto-shutdown, deploy OmniParser container.
        Returns the public IP and instance ID on success, or (None, None) on failure.
        """
        instance_id = None
        instance_ip = None
        ssh_client = None
        key_path = config.AWS_EC2_KEY_PATH

        try:
            # 1. Deploy or find/start EC2 instance
            logger.info("Step 1: Deploying/Starting EC2 Instance...")
            instance_id, instance_ip = deploy_ec2_instance()
            if not instance_id or not instance_ip:
                # deploy_ec2_instance already logs the error
                raise RuntimeError("Failed to deploy or start EC2 instance")
            logger.success(f"EC2 instance ready: ID={instance_id}, IP={instance_ip}")

            # 2. Configure EC2 Instance (Docker etc.)
            logger.info("Step 2: Configuring EC2 Instance (Docker, etc.)...")
            if not os.path.exists(key_path):
                logger.error(
                    f"SSH Key not found at {key_path}. Cannot proceed with configuration."
                )
                raise RuntimeError(f"SSH Key missing: {key_path}")
            config_success = configure_ec2_instance(instance_id, instance_ip)
            if not config_success:
                # configure_ec2_instance already logs the error
                raise RuntimeError("Failed to configure EC2 instance")
            logger.success("EC2 instance configuration complete.")

            # 3. Set up Auto-Shutdown Infrastructure (Alarm-based)
            logger.info("Step 3: Setting up Auto-Shutdown Infrastructure...")
            # This function now handles errors internally and logs them but doesn't stop deployment
            create_auto_shutdown_infrastructure(instance_id)
            # Success/failure logged within the function

            # 4. Trigger Driver Installation via Non-Interactive SSH Login
            logger.info(
                "Step 4: Triggering potential driver install via SSH login (might cause temporary disconnect)..."
            )
            try:
                Deploy.ssh(non_interactive=True)
                logger.success("Non-interactive SSH login trigger completed.")
            except Exception as ssh_e:
                logger.warning(f"Non-interactive SSH step failed or timed out: {ssh_e}")
                logger.warning(
                    "Proceeding with Docker deployment, assuming instance is accessible."
                )

            # 5. Copy Dockerfile, .dockerignore
            logger.info("Step 5: Copying Docker related files...")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            files_to_copy = {
                "Dockerfile": os.path.join(current_dir, "Dockerfile"),
                ".dockerignore": os.path.join(current_dir, ".dockerignore"),
            }
            for filename, filepath in files_to_copy.items():
                if os.path.exists(filepath):
                    logger.info(f"Copying {filename} to instance {instance_ip}...")
                    scp_command = [
                        "scp",
                        "-i",
                        key_path,
                        "-o",
                        "StrictHostKeyChecking=no",
                        "-o",
                        "UserKnownHostsFile=/dev/null",
                        "-o",
                        "ConnectTimeout=30",
                        filepath,
                        f"{config.AWS_EC2_USER}@{instance_ip}:~/{filename}",
                    ]
                    result = subprocess.run(
                        scp_command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode != 0:
                        logger.error(
                            f"Failed to copy {filename}: {result.stderr or result.stdout}"
                        )
                        # Allow continuing even if copy fails? Or raise error? Let's allow for now.
                    else:
                        logger.info(f"Successfully copied {filename}.")
                else:
                    logger.warning(
                        f"Required file not found: {filepath}. Skipping copy."
                    )

            # 6. Connect SSH and Run Setup/Docker Commands
            logger.info(
                "Step 6: Connecting via SSH to run setup and Docker commands..."
            )
            key = paramiko.RSAKey.from_private_key_file(key_path)
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            try:
                logger.info(f"Attempting final SSH connection to {instance_ip}...")
                ssh_client.connect(
                    hostname=instance_ip,
                    username=config.AWS_EC2_USER,
                    pkey=key,
                    timeout=30,
                )
                logger.success("SSH connected for Docker setup.")

                setup_commands = [  # Ensure commands are safe and idempotent if possible
                    "rm -rf OmniParser",
                    f"git clone --depth 1 {config.REPO_URL}",
                    "if [ -f ~/Dockerfile ]; then cp ~/Dockerfile ~/OmniParser/; else echo 'Warning: Dockerfile not found in home dir'; fi",
                    "if [ -f ~/.dockerignore ]; then cp ~/.dockerignore ~/OmniParser/; else echo 'Warning: .dockerignore not found in home dir'; fi",
                ]
                for command in setup_commands:
                    execute_command(ssh_client, command)

                docker_commands = [
                    f"sudo docker rm -f {config.CONTAINER_NAME} || true",
                    f"sudo docker rmi {config.PROJECT_NAME} || true",
                    (
                        f"cd OmniParser && sudo docker build --progress=plain "
                        f"--no-cache -t {config.PROJECT_NAME} ."
                    ),
                    (
                        f"sudo docker run -d --restart always -p {config.PORT}:{config.PORT} --gpus all --name "
                        f"{config.CONTAINER_NAME} {config.PROJECT_NAME}"
                    ),
                ]
                for command in docker_commands:
                    execute_command(ssh_client, command)
                logger.success("Docker build and run commands executed.")

                # 7. Wait for Container/Server to Become Responsive
                logger.info(
                    "Step 7: Waiting for server inside container to become responsive..."
                )
                max_retries = 30
                retry_delay = 10
                server_ready = False
                check_command = (
                    f"curl -s --fail http://localhost:{config.PORT}/probe/ || exit 1"
                )
                for attempt in range(max_retries):
                    logger.info(
                        f"Checking server readiness via internal curl (attempt {attempt + 1}/{max_retries})..."
                    )
                    try:
                        execute_command(ssh_client, check_command, max_retries=1)
                        logger.success("Server is responsive inside instance!")
                        server_ready = True
                        break
                    except Exception as e:
                        logger.warning(f"Server not ready yet (internal check): {e}")
                        if attempt < max_retries - 1:
                            try:
                                logger.info("Checking Docker container status...")
                                execute_command(
                                    ssh_client,
                                    f"sudo docker ps -f name={config.CONTAINER_NAME}",
                                    max_retries=1,
                                )
                            except Exception as ps_e:
                                logger.error(f"Container check failed: {ps_e}")
                            logger.info(f"Waiting {retry_delay} seconds...")
                            time.sleep(retry_delay)
                if not server_ready:
                    try:
                        logger.error(
                            "Server failed to become responsive. Getting container logs..."
                        )
                        execute_command(
                            ssh_client, f"sudo docker logs {config.CONTAINER_NAME}"
                        )
                    except Exception as log_e:
                        logger.error(f"Could not retrieve container logs: {log_e}")
                    raise RuntimeError(
                        f"Server at localhost:{config.PORT} did not become responsive."
                    )

                # Final check
                execute_command(
                    ssh_client, f"sudo docker ps --filter name={config.CONTAINER_NAME}"
                )

            finally:
                if ssh_client:
                    ssh_client.close()
                    logger.info("SSH connection for Docker setup closed.")

            # 8. Deployment Successful
            server_url = f"http://{instance_ip}:{config.PORT}"
            logger.success(f"Deployment complete! Server running at: {server_url}")
            logger.info(
                f"Auto-shutdown configured for inactivity (approx {config.INACTIVITY_TIMEOUT_MINUTES} minutes of low CPU)."
            )

            # Optional: Verify external access
            try:
                import requests

                logger.info(f"Verifying external access to {server_url}/probe/ ...")
                response = requests.get(f"{server_url}/probe/", timeout=20)
                response.raise_for_status()
                logger.success(
                    "Successfully verified external access to /probe/ endpoint."
                )
            except Exception as e:
                logger.warning(f"Could not verify external access to server: {e}")

            # Return IP and ID on success
            return instance_ip, instance_id

        except Exception as e:
            logger.error(f"Deployment failed: {e}", exc_info=True)
            if CLEANUP_ON_FAILURE and instance_id:
                logger.warning("Attempting cleanup due to deployment failure...")
                try:
                    Deploy.stop(project_name=config.PROJECT_NAME)
                except Exception as cleanup_error:
                    logger.error(f"Cleanup after failure also failed: {cleanup_error}")
            # Return None on failure
            return None, None

    @staticmethod
    def stop(
        project_name: str = config.PROJECT_NAME,
        security_group_name: str = config.AWS_EC2_SECURITY_GROUP,
    ) -> None:
        """
        Initiates termination of EC2 instance(s) and deletion of associated resources
        (SG, Auto-Shutdown Lambda, CW Alarm, IAM Role). Returns before termination completes.
        Excludes Discovery API components cleanup.

        Args:
            project_name (str): The project name used to tag the instance.
            security_group_name (str): The name of the security group to delete.
        """
        # 1. Initialize clients
        ec2_resource = boto3.resource("ec2", region_name=config.AWS_REGION)
        ec2_client = boto3.client("ec2", region_name=config.AWS_REGION)
        lambda_client = boto3.client("lambda", region_name=config.AWS_REGION)
        cloudwatch_client = boto3.client("cloudwatch", region_name=config.AWS_REGION)
        iam_client = boto3.client("iam", region_name=config.AWS_REGION)

        logger.info("Starting cleanup initiation...")

        # 2. Initiate EC2 instance termination
        instances_to_terminate = []
        try:
            instances = ec2_resource.instances.filter(
                Filters=[
                    {"Name": "tag:Name", "Values": [project_name]},
                    {
                        "Name": "instance-state-name",
                        "Values": [
                            "pending",
                            "running",
                            "shutting-down",  # Include shutting-down just in case
                            "stopped",
                            "stopping",
                        ],
                    },
                ]
            )
            instance_list = list(instances)
            if not instance_list:
                logger.info(
                    f"No instances found with tag Name={project_name} to terminate."
                )
            else:
                logger.info(
                    f"Found {len(instance_list)} instance(s). Initiating termination..."
                )
                for instance in instance_list:
                    logger.info(
                        f"Initiating termination for instance: ID - {instance.id}"
                    )
                    instances_to_terminate.append(instance.id)
                    try:
                        instance.terminate()
                    except ClientError as term_error:
                        # Log error but don't stop overall cleanup
                        logger.warning(
                            f"Could not issue terminate for {instance.id}: {term_error}"
                        )

                if instances_to_terminate:
                    logger.info(
                        f"Termination initiated for instance(s): {instances_to_terminate}. AWS will complete this in the background."
                    )
                # --- REMOVED WAITER BLOCK ---
                # logger.info(f"Waiting for instance(s) {instances_terminated} to terminate...")
                # try:
                #     waiter = ec2_client.get_waiter('instance_terminated')
                #     waiter.wait(...)
                #     logger.info(f"Instance(s) {instances_terminated} confirmed terminated.")
                # except Exception as wait_error:
                #     logger.warning(f"Error or timeout waiting for instance termination: {wait_error}")
                #     logger.warning("Proceeding with cleanup...")

        except Exception as e:
            logger.error(f"Error during instance discovery/termination initiation: {e}")
            # Continue cleanup attempt anyway

        # 3. Delete CloudWatch Alarms
        try:
            alarm_prefix = f"{config.PROJECT_NAME}-CPU-Low-Alarm-"
            paginator = cloudwatch_client.get_paginator("describe_alarms")
            alarms_to_delete = []
            logger.info(f"Searching for CloudWatch alarms with prefix: {alarm_prefix}")
            for page in paginator.paginate(AlarmNamePrefix=alarm_prefix):
                for alarm in page.get("MetricAlarms", []):
                    alarms_to_delete.append(alarm["AlarmName"])
            alarms_to_delete = list(set(alarms_to_delete))
            if alarms_to_delete:
                logger.info(f"Deleting CloudWatch alarms: {alarms_to_delete}")
                for i in range(0, len(alarms_to_delete), 100):
                    chunk = alarms_to_delete[i : i + 100]
                    try:
                        cloudwatch_client.delete_alarms(AlarmNames=chunk)
                        logger.info(f"Deleted alarm chunk: {chunk}")
                    except ClientError as delete_alarm_err:
                        logger.error(
                            f"Failed to delete alarm chunk {chunk}: {delete_alarm_err}"
                        )
            else:
                logger.info("No matching CloudWatch alarms found to delete.")
        except Exception as e:
            logger.error(f"Error searching/deleting CloudWatch alarms: {e}")

        # 4. Delete Lambda function
        lambda_function_name = LAMBDA_FUNCTION_NAME
        try:
            logger.info(f"Attempting to delete Lambda function: {lambda_function_name}")
            lambda_client.delete_function(FunctionName=lambda_function_name)
            logger.info(f"Deleted Lambda function: {lambda_function_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.info(f"Lambda function {lambda_function_name} does not exist.")
            else:
                logger.error(
                    f"Error deleting Lambda function {lambda_function_name}: {e}"
                )

        # 5. Delete IAM Role
        role_name = IAM_ROLE_NAME
        try:
            logger.info(f"Attempting to delete IAM role: {role_name}")
            attached_policies = iam_client.list_attached_role_policies(
                RoleName=role_name
            ).get("AttachedPolicies", [])
            if attached_policies:
                logger.info(
                    f"Detaching {len(attached_policies)} managed policies from role {role_name}..."
                )
            for policy in attached_policies:
                try:
                    iam_client.detach_role_policy(
                        RoleName=role_name, PolicyArn=policy["PolicyArn"]
                    )
                    logger.debug(f"Detached policy {policy['PolicyArn']}")
                except ClientError as detach_err:
                    logger.warning(
                        f"Could not detach policy {policy['PolicyArn']}: {detach_err}"
                    )
            inline_policies = iam_client.list_role_policies(RoleName=role_name).get(
                "PolicyNames", []
            )
            if inline_policies:
                logger.info(
                    f"Deleting {len(inline_policies)} inline policies from role {role_name}..."
                )
            for policy_name in inline_policies:
                try:
                    iam_client.delete_role_policy(
                        RoleName=role_name, PolicyName=policy_name
                    )
                    logger.debug(f"Deleted inline policy {policy_name}")
                except ClientError as inline_err:
                    logger.warning(
                        f"Could not delete inline policy {policy_name}: {inline_err}"
                    )
            iam_client.delete_role(RoleName=role_name)
            logger.info(f"Deleted IAM role: {role_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                logger.info(f"IAM role {role_name} does not exist.")
            elif e.response["Error"]["Code"] == "DeleteConflict":
                logger.error(
                    f"Cannot delete IAM role {role_name} due to dependencies: {e}"
                )
            else:
                logger.error(f"Error deleting IAM role {role_name}: {e}")

        # 6. Delete Security Group
        # Might still fail if instance termination hasn't fully released ENIs,
        # but we don't wait for termination anymore. Manual cleanup might be needed sometimes.
        sg_delete_wait = 5  # Shorter wait now, as we aren't waiting for termination
        logger.info(
            f"Waiting {sg_delete_wait} seconds before attempting security group deletion..."
        )
        time.sleep(sg_delete_wait)
        try:
            logger.info(f"Attempting to delete security group: {security_group_name}")
            ec2_client.delete_security_group(GroupName=security_group_name)
            logger.info(f"Deleted security group: {security_group_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "InvalidGroup.NotFound":
                logger.info(f"Security group {security_group_name} not found.")
            elif e.response["Error"]["Code"] == "DependencyViolation":
                logger.warning(
                    f"Could not delete security group {security_group_name} due to existing dependencies (likely ENI from terminating instance). AWS will clean it up later, or run stop again after a few minutes. Error: {e}"
                )
            else:
                logger.error(
                    f"Error deleting security group {security_group_name}: {e}"
                )

        logger.info(
            "Cleanup initiation finished. Instance termination proceeds in background."
        )

    @staticmethod
    def status() -> None:
        """Check the status of deployed instances."""
        ec2 = boto3.resource("ec2", region_name=config.AWS_REGION)
        instances = ec2.instances.filter(
            Filters=[{"Name": "tag:Name", "Values": [config.PROJECT_NAME]}]
        )

        for instance in instances:
            public_ip = instance.public_ip_address
            if public_ip:
                server_url = f"http://{public_ip}:{config.PORT}"
                logger.info(
                    f"Instance ID: {instance.id}, State: {instance.state['Name']}, "
                    f"URL: {server_url}"
                )
            else:
                logger.info(
                    f"Instance ID: {instance.id}, State: {instance.state['Name']}, "
                    f"URL: Not available (no public IP)"
                )

        # Check auto-shutdown infrastructure
        lambda_client = boto3.client("lambda", region_name=config.AWS_REGION)

        try:
            lambda_response = lambda_client.get_function(
                FunctionName=LAMBDA_FUNCTION_NAME
            )
            logger.info(f"Auto-shutdown Lambda: {LAMBDA_FUNCTION_NAME} (Active)")
            logger.debug(f"{lambda_response=}")
        except ClientError:
            logger.info("Auto-shutdown Lambda: Not configured")

    @staticmethod
    def ssh(non_interactive: bool = False) -> None:
        # Get instance IP
        ec2 = boto3.resource("ec2", region_name=config.AWS_REGION)
        instances = ec2.instances.filter(
            Filters=[
                {"Name": "tag:Name", "Values": [config.PROJECT_NAME]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ]
        )

        instance = next(iter(instances), None)
        if not instance:
            logger.error("No running instance found")
            return

        ip = instance.public_ip_address
        if not ip:
            logger.error("Instance has no public IP")
            return

        # Check if key file exists
        if not os.path.exists(config.AWS_EC2_KEY_PATH):
            logger.error(f"Key file not found: {config.AWS_EC2_KEY_PATH}")
            return

        if non_interactive:
            # Trigger driver installation (this might cause reboot)
            ssh_command = [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-i",
                config.AWS_EC2_KEY_PATH,
                f"{config.AWS_EC2_USER}@{ip}",
                "-t",
                "-tt",
                "bash --login -c 'exit'",
            ]

            try:
                subprocess.run(ssh_command, check=True)
                logger.info("Initial SSH login completed successfully")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Initial SSH connection closed: {e}")

            # Wait for potential reboot to complete
            logger.info(
                "Waiting for instance to be fully available after potential reboot..."
            )
            max_attempts = 20
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                logger.info(f"SSH connection attempt {attempt}/{max_attempts}")
                try:
                    # Check if we can make a new SSH connection
                    test_ssh_cmd = [
                        "ssh",
                        "-o",
                        "StrictHostKeyChecking=no",
                        "-o",
                        "ConnectTimeout=5",
                        "-o",
                        "UserKnownHostsFile=/dev/null",
                        "-i",
                        config.AWS_EC2_KEY_PATH,
                        f"{config.AWS_EC2_USER}@{ip}",
                        "echo 'SSH connection successful'",
                    ]
                    result = subprocess.run(
                        test_ssh_cmd, capture_output=True, text=True, check=False
                    )
                    if result.returncode == 0:
                        logger.info("Instance is ready for SSH connections")
                        return
                except Exception as e:
                    # Expected while the instance is still booting; keep
                    # retrying, but do not swallow it silently -- a
                    # misconfigured key path looked identical to "not up yet".
                    logger.debug(f"SSH readiness probe failed: {e}")

                time.sleep(10)  # Wait 10 seconds between attempts

            logger.error("Failed to reconnect to instance after potential reboot")
        else:
            # Interactive SSH session
            ssh_command = f"ssh -i {config.AWS_EC2_KEY_PATH} -o StrictHostKeyChecking=no {config.AWS_EC2_USER}@{ip}"
            logger.info(f"Connecting with: {ssh_command}")
            os.system(ssh_command)
            return

    @staticmethod
    def stop_instance(instance_id: str) -> None:
        """Stop a specific EC2 instance."""
        ec2_client = boto3.client("ec2", region_name=config.AWS_REGION)
        try:
            ec2_client.stop_instances(InstanceIds=[instance_id])
            logger.info(f"Stopped instance {instance_id}")
        except ClientError as e:
            logger.error(f"Error stopping instance {instance_id}: {e}")

    @staticmethod
    def start_instance(instance_id: str) -> str:
        """Start a specific EC2 instance and return its public IP."""
        ec2_client = boto3.client("ec2", region_name=config.AWS_REGION)
        ec2_resource = boto3.resource("ec2", region_name=config.AWS_REGION)

        try:
            ec2_client.start_instances(InstanceIds=[instance_id])
            logger.info(f"Starting instance {instance_id}...")

            instance = ec2_resource.Instance(instance_id)
            instance.wait_until_running()
            instance.reload()

            logger.info(
                f"Instance {instance_id} started, IP: {instance.public_ip_address}"
            )
            return instance.public_ip_address
        except ClientError as e:
            logger.error(f"Error starting instance {instance_id}: {e}")
            return None

    @staticmethod
    def history(days: int = 7) -> None:
        """Display deployment and auto-shutdown history.

        Args:
            days: Number of days of history to retrieve (default: 7)
        """
        logger.info(f"Retrieving {days} days of deployment history...")

        # Calculate time range
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(days=days)

        # Initialize AWS clients
        cloudwatch_logs = boto3.client("logs", region_name=config.AWS_REGION)
        ec2_client = boto3.client("ec2", region_name=config.AWS_REGION)

        # Get instance information
        instances = []
        try:
            response = ec2_client.describe_instances(
                Filters=[{"Name": "tag:Name", "Values": [config.PROJECT_NAME]}]
            )
            for reservation in response["Reservations"]:
                instances.extend(reservation["Instances"])

            logger.info(
                f"Found {len(instances)} instances with name tag '{config.PROJECT_NAME}'"
            )
        except Exception as e:
            logger.error(f"Error retrieving instances: {e}")

        # Display instance state transition history
        logger.info("\n=== Instance State History ===")
        for instance in instances:
            instance_id = instance["InstanceId"]
            try:
                # Get instance state transition history
                response = ec2_client.describe_instance_status(
                    InstanceIds=[instance_id], IncludeAllInstances=True
                )

                state = instance["State"]["Name"]
                launch_time = instance.get("LaunchTime", "Unknown")

                logger.info(
                    f"Instance {instance_id}: Current state={state}, Launch time={launch_time}"
                )

                # Get instance console output if available
                try:
                    console = ec2_client.get_console_output(InstanceId=instance_id)
                    if console.get("Output"):
                        logger.info("Last console output (truncated):")
                        # Show last few lines of console output
                        lines = console["Output"].strip().split("\n")
                        for line in lines[-10:]:
                            logger.info(f"  {line}")
                except Exception as e:
                    logger.info(f"Console output not available: {e}")

            except Exception as e:
                logger.error(f"Error retrieving status for instance {instance_id}: {e}")

        # Check for Lambda auto-shutdown logs
        logger.info("\n=== Auto-shutdown Lambda Logs ===")
        try:
            # Check if log group exists
            log_group_name = f"/aws/lambda/{LAMBDA_FUNCTION_NAME}"

            log_streams = cloudwatch_logs.describe_log_streams(
                logGroupName=log_group_name,
                orderBy="LastEventTime",
                descending=True,
                limit=5,
            )

            if not log_streams.get("logStreams"):
                logger.info("No log streams found for auto-shutdown Lambda")
            else:
                # Process the most recent log streams
                for stream in log_streams.get("logStreams", [])[:5]:
                    stream_name = stream["logStreamName"]
                    logger.info(f"Log stream: {stream_name}")

                    logs = cloudwatch_logs.get_log_events(
                        logGroupName=log_group_name,
                        logStreamName=stream_name,
                        startTime=int(start_time.timestamp() * 1000),
                        endTime=int(end_time.timestamp() * 1000),
                        limit=100,
                    )

                    if not logs.get("events"):
                        logger.info("  No events in this stream")
                        continue

                    for event in logs.get("events", []):
                        timestamp = datetime.datetime.fromtimestamp(
                            event["timestamp"] / 1000, tz=datetime.timezone.utc
                        )
                        message = event["message"]
                        logger.info(f"  {timestamp}: {message}")

        except cloudwatch_logs.exceptions.ResourceNotFoundException:
            logger.info(
                "No logs found for auto-shutdown Lambda. It may not have been triggered yet."
            )
        except Exception as e:
            logger.error(f"Error retrieving Lambda logs: {e}")

        logger.info("\nHistory retrieval complete.")


@staticmethod
def discover() -> dict:
    """Discover instances by tag and optionally start them if stopped.

    Returns:
        dict: Information about the discovered instance including status and connection
            details
    """
    ec2 = boto3.resource("ec2", region_name=config.AWS_REGION)

    # Find instance with project tag
    instances = list(
        ec2.instances.filter(
            Filters=[
                {"Name": "tag:Name", "Values": [config.PROJECT_NAME]},
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopped"],
                },
            ]
        )
    )

    if not instances:
        logger.info("No instances found")
        return {"status": "not_found"}

    instance = instances[0]  # Get the first matching instance
    logger.info(f"Found instance {instance.id} in state {instance.state['Name']}")

    # If instance is stopped, start it
    if instance.state["Name"] == "stopped":
        logger.info(f"Starting stopped instance {instance.id}")
        instance.start()
        return {
            "instance_id": instance.id,
            "status": "starting",
            "message": "Instance is starting. Please try again in a few minutes.",
        }

    # Return info for running instance
    if instance.state["Name"] == "running":
        return {
            "instance_id": instance.id,
            "public_ip": instance.public_ip_address,
            "status": instance.state["Name"],
            "api_url": f"http://{instance.public_ip_address}:{config.PORT}",
        }

    # Instance is in another state (e.g., pending)
    return {
        "instance_id": instance.id,
        "status": instance.state["Name"],
        "message": f"Instance is {instance.state['Name']}. Please try again shortly.",
    }


if __name__ == "__main__":
    # Ensure boto3 clients use the region from config if set
    # Note: Boto3 usually picks region from env vars or ~/.aws/config first
    if config.AWS_REGION:
        boto3.setup_default_session(region_name=config.AWS_REGION)
    fire.Fire(Deploy)
