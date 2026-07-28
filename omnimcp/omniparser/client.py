# omnimcp/omniparser/client.py

"""Client module for interacting with the OmniParser server."""

import base64

import boto3  # Need boto3 for the initial check
import requests
from loguru import logger
from PIL import Image, ImageDraw

from ..config import config
from .server import Deploy


class OmniParserClient:
    """Client for interacting with the OmniParser server."""

    def __init__(self, server_url: str | None = None, auto_deploy: bool = True):
        """Initialize the OmniParser client.

        Args:
            server_url: URL of the OmniParser server. If None, will attempt to find
                or deploy a server.
            auto_deploy: Whether to automatically deploy a server if none is found.
        """
        self.server_url = server_url
        self.auto_deploy = auto_deploy
        self._ensure_server()

    def _ensure_server(self) -> None:
        """Ensure a server is available, deploying one if necessary."""
        if self.server_url:
            logger.info(f"Using provided server URL: {self.server_url}")
        else:
            logger.info("No server_url provided, attempting discovery/deployment...")
            # Try finding existing running instance first
            instance_ip = None
            instance_id = None
            try:
                ec2 = boto3.resource("ec2", region_name=config.AWS_REGION)
                instances = ec2.instances.filter(
                    Filters=[
                        {
                            "Name": "tag:Name",
                            "Values": [config.PROJECT_NAME],
                        },  # Use project name tag
                        {"Name": "instance-state-name", "Values": ["running"]},
                    ]
                )
                # Get the most recently launched running instance
                running_instances = sorted(
                    instances, key=lambda i: i.launch_time, reverse=True
                )
                instance = running_instances[0] if running_instances else None

                if instance and instance.public_ip_address:
                    instance_ip = instance.public_ip_address
                    instance_id = instance.id  # Store ID too for logging maybe
                    self.server_url = f"http://{instance_ip}:{config.PORT}"
                    logger.success(
                        f"Found existing running server instance {instance_id} at {self.server_url}"
                    )
                elif self.auto_deploy:
                    logger.info(
                        "No running server found, attempting auto-deployment via Deploy.start()..."
                    )
                    # Call start and get the result directly
                    deployer = Deploy()
                    # Deploy.start now returns IP and ID
                    instance_ip, instance_id = deployer.start()

                    if instance_ip and instance_id:
                        # Deployment succeeded, set the URL
                        self.server_url = f"http://{instance_ip}:{config.PORT}"
                        logger.success(
                            f"Auto-deployment successful. Server URL: {self.server_url} (Instance ID: {instance_id})"
                        )
                    else:
                        # deployer.start() failed and returned None
                        raise RuntimeError(
                            "Auto-deployment failed (Deploy.start did not return valid IP/ID). Check server logs."
                        )
                else:  # No running instance and auto_deploy is False
                    raise RuntimeError(
                        "No server URL provided, no running instance found, and auto_deploy is disabled."
                    )

            except Exception as e:
                logger.error(
                    f"Error during server discovery/deployment: {e}", exc_info=True
                )
                # Re-raise as a RuntimeError to be caught by the main script if needed
                raise RuntimeError(f"Server discovery/deployment failed: {e}") from e

        # Verify server is responsive (only if server_url is now set)
        if self.server_url:
            logger.info(f"Checking server responsiveness at {self.server_url}...")
            try:
                self._check_server()  # This probes the URL
                logger.success(f"Server at {self.server_url} is responsive.")
            except Exception as check_err:
                logger.error(f"Server check failed for {self.server_url}: {check_err}")
                # Raise error - if we have a URL it should be responsive after deployment/discovery
                raise RuntimeError(
                    f"Server at {self.server_url} failed responsiveness check."
                ) from check_err
        else:
            # Safety check - should not be reachable if logic above is correct
            raise RuntimeError("Critical error: Failed to obtain server URL.")

    def _check_server(self) -> None:
        """Check if the server is responsive."""
        if not self.server_url:
            raise RuntimeError(
                "Cannot check server responsiveness, server_url is not set."
            )
        try:
            # Increased timeout slightly
            response = requests.get(f"{self.server_url}/probe/", timeout=15)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            # Check content if needed: assert response.json().get("message") == "..."
        except requests.exceptions.Timeout:
            logger.error(
                f"Timeout connecting to server probe endpoint: {self.server_url}/probe/"
            )
            raise RuntimeError(f"Server probe timed out for {self.server_url}")
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Connection error reaching server probe endpoint: {self.server_url}/probe/"
            )
            raise RuntimeError(f"Server probe connection error for {self.server_url}")
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Error during server probe request for {self.server_url}: {e}"
            )
            raise RuntimeError(f"Server probe failed: {e}") from e

    def parse_image(self, image: Image.Image) -> dict:
        """Parse an image using the OmniParser server.

        Args:
            image: PIL Image to parse

        Returns:
            Dict containing parsing results
        """
        # Convert image to base64
        image_bytes = self._image_to_base64(image)

        # Make request
        try:
            response = requests.post(
                f"{self.server_url}/parse/",
                json={"base64_image": image_bytes},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to parse image: {e}"}

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import io

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def visualize_results(
        self, image: Image.Image, parsed_content: list[dict]
    ) -> Image.Image:
        """Visualize parsing results on the image.

        Args:
            image: Original PIL Image
            parsed_content: List of parsed content with bounding boxes

        Returns:
            PIL Image with visualizations
        """
        # Create copy of image
        viz_image = image.copy()
        draw = ImageDraw.Draw(viz_image)

        # Draw results
        for item in parsed_content:
            # Get coordinates
            x1, y1, x2, y2 = item["bbox"]
            x1 = int(x1 * image.width)
            y1 = int(y1 * image.height)
            x2 = int(x2 * image.width)
            y2 = int(y2 * image.height)

            # Draw box
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)

            # Draw label
            label = item["content"]
            bbox = draw.textbbox((x1, y1), label)
            draw.rectangle(bbox, fill="white")
            draw.text((x1, y1), label, fill="red")

        return viz_image


# Example usage:
if __name__ == "__main__":
    # Create client (will auto-deploy if needed)
    client = OmniParserClient()

    # Parse an image
    image = Image.open("../OpenAdapt/tests/assets/excel.png")
    results = client.parse_image(image)

    # Visualize results
    if "error" not in results:
        viz_image = client.visualize_results(image, results["parsed_content_list"])
        viz_image.show()
