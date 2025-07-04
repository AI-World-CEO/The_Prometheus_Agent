# Prometheus_Agent/Cloud/terraform/main.tf

# --- Provider Configuration ---
# Specifies that we are using the AWS provider and sets the region.
provider "aws" {
  region = "us-east-1"
}

# --- Variables ---
# Defines variables for customization without changing the code.
variable "instance_name" {
  description = "The name for the EC2 instance."
  type        = string
  default     = "Prometheus-Agent-Instance"
}

variable "instance_type" {
  description = "The EC2 instance type. Should be a GPU instance for AI workloads."
  type        = string
  default     = "g4dn.xlarge" # A cost-effective GPU instance
}

# --- Data Sources ---
# Automatically find the latest Amazon Linux 2 AMI with GPU support.
data "aws_ami" "amazon_linux_gpu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-ecs-gpu-hvm-*-x86_64-ebs"]
  }
}

# --- Networking Resources ---
# Creates a new Virtual Private Cloud (VPC) for network isolation.
resource "aws_vpc" "prometheus_vpc" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "Prometheus-VPC"
  }
}

# Creates an internet gateway to allow the instance to connect to the internet.
resource "aws_internet_gateway" "prometheus_igw" {
  vpc_id = aws_vpc.prometheus_vpc.id
  tags = {
    Name = "Prometheus-IGW"
  }
}

# Creates a public subnet within our VPC.
resource "aws_subnet" "prometheus_subnet" {
  vpc_id     = aws_vpc.prometheus_vpc.id
  cidr_block = "10.0.1.0/24"
  map_public_ip_on_launch = true
  tags = {
    Name = "Prometheus-Public-Subnet"
  }
}

# Creates a route table to direct internet-bound traffic to the internet gateway.
resource "aws_route_table" "prometheus_rt" {
  vpc_id = aws_vpc.prometheus_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.prometheus_igw.id
  }

  tags = {
    Name = "Prometheus-Route-Table"
  }
}

# Associates the route table with our subnet.
resource "aws_route_table_association" "a" {
  subnet_id      = aws_subnet.prometheus_subnet.id
  route_table_id = aws_route_table.prometheus_rt.id
}

# --- Security Group (Firewall) ---
# Defines firewall rules for our instance.
resource "aws_security_group" "prometheus_sg" {
  name        = "prometheus-sg"
  description = "Allow SSH and potentially other app-specific traffic"
  vpc_id      = aws_vpc.prometheus_vpc.id

  # Allow inbound SSH traffic (Port 22) from anywhere.
  # For production, you should restrict this to your IP address.
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow SSH from anywhere (for development)"
  }

  # Allow all outbound traffic.
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "Prometheus-Security-Group"
  }
}

# --- EC2 Instance ---
# The core resource: a virtual server in the cloud.
resource "aws_instance" "prometheus_server" {
  ami           = data.aws_ami.amazon_linux_gpu.id
  instance_type = var.instance_type
  subnet_id     = aws_subnet.prometheus_subnet.id
  vpc_security_group_ids = [aws_security_group.prometheus_sg.id]

  # You would add your SSH key here to be able to connect to the instance.
  # key_name = "your-aws-key-pair-name"

  # User data script to bootstrap the instance on first launch.
  # This script installs Docker and runs our application.
  user_data = <<-EOF
              #!/bin/bash
              # Install Docker
              yum update -y
              yum install -y docker
              service docker start
              usermod -a -G docker ec2-user

              # Pull your Docker image from a registry (e.g., Docker Hub or AWS ECR)
              # For this example, you would build and push the Dockerfile first.
              # docker pull your-username/prometheus-agent:latest

              # Run the container
              # You would pass your .env file securely, e.g., using AWS Secrets Manager.
              # docker run -d --gpus all --env-file .env your-username/prometheus-agent:latest
              EOF

  tags = {
    Name = var.instance_name
  }
}

# --- Outputs ---
# Displays the public IP address of the instance after it's created.
output "instance_public_ip" {
  value       = aws_instance.prometheus_server.public_ip
  description = "The public IP address of the Prometheus Agent EC2 instance."
}