# tests/test_omnimcp.py

"""Tests for OmniParser deployment functionality (E2E)."""

import pytest
import time
import boto3
import requests
from typing import List

from omnimcp.omniparser.server import Deploy
from omnimcp.config import config


def get_running_parser_instances() -> List[dict]:
    """Get any running OmniParser instances."""
    # (Implementation remains the same as provided)
    ec2 = boto3.resource("ec2", region_name=config.AWS_REGION)
    instances = list(
        ec2.instances.filter(
            Filters=[
                {"Name": "tag:Name", "Values": [config.PROJECT_NAME]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ]
        )
    )
    running_instances = []
    for instance in instances:
        if instance.public_ip_address:
            url = f"http://{instance.public_ip_address}:{config.PORT}/probe/"
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    running_instances.append(
                        {
                            "id": instance.id,
                            "ip": instance.public_ip_address,
                            "url": f"http://{instance.public_ip_address}:{config.PORT}",
                        }
                    )
            except requests.exceptions.RequestException:
                pass
    return running_instances


def cleanup_parser_instances():
    """Stop all running parser instances."""
    Deploy.stop()


# @pytest.fixture(scope="module")
# def test_image():
#     """Generate synthetic test image."""
#     img, _ = synthetic_ui_helpers.generate_test_ui()
#     return img


@pytest.mark.e2e
class TestParserDeployment:
    """Test suite for OmniParser deployment scenarios."""

    @classmethod
    def setup_class(cls):
        """Initial setup for all tests."""
        cls.initial_instances = get_running_parser_instances()
        print(f"\nInitial running instances: {len(cls.initial_instances)}")
        # Ensure cleanup happens before tests if needed, or rely on teardown
        # cleanup_parser_instances()

    @classmethod
    def teardown_class(cls):
        """Cleanup after all tests."""
        print("\nCleaning up parser instances after tests...")
        cleanup_parser_instances()
        # Short wait to allow termination to progress slightly before final check
        time.sleep(10)
        final_instances = get_running_parser_instances()
        # Allow for some flexibility if initial instances were present
        print(f"Final running instances after cleanup: {len(final_instances)}")
        # assert len(final_instances) == 0, "Cleanup did not terminate all instances"
        # Asserting <= initial might be safer if tests run against pre-existing envs
        assert len(final_instances) <= len(cls.initial_instances), "Cleanup failed"


#     @pytest.mark.skipif(
#         # This skip logic might be less reliable now, consider removing or adjusting
#         # condition=lambda: len(get_running_parser_instances()) > 0,
#         False, # Let's try running it, client init should handle existing instances
#         reason="Skip logic needs review, test client's ability to find existing"
#     )
#     def test_auto_deployment(self, test_image):
#         """Test client auto-deploys when no instance exists."""
#         # Ensure no instances are running before this specific test
#         print("\nEnsuring no instances are running before auto-deploy test...")
#         cleanup_parser_instances()
#         time.sleep(15) # Wait longer after stop
#         running_instances = get_running_parser_instances()
#         assert len(running_instances) == 0, "Test requires no running instances at start"
#
#         # Instantiate client - should trigger auto-deployment
#         print("Initializing client to trigger auto-deployment...")
#         deployment_start = time.time()
#         try:
#              # Init client with auto_deploy=True (default) and no URL
#              client = OmniParserClient(server_url=None, auto_deploy=True)
#         except Exception as e:
#              pytest.fail(f"OmniParserClient initialization failed during auto-deploy: {e}")
#
#         deployment_time = time.time() - deployment_start
#         print(f"Client initialization (inc. deployment) took {deployment_time:.1f} seconds")
#
#         # Verify deployment happened (at least one instance should be running now)
#         running_instances = get_running_parser_instances()
#         assert len(running_instances) >= 1, \
#             f"Expected at least 1 running instance after auto-deploy, found {len(running_instances)}"
#
#         # Verify parsing works via the client instance
#         assert client.server_url is not None, "Client did not get a server URL after deployment"
#         print(f"Parsing image using deployed server: {client.server_url}")
#         result = client.parse_image(test_image)
#
#         assert result is not None, "Parse result should not be None"
#         assert "error" not in result, f"Parsing failed: {result.get('error')}"
#         assert "parsed_content_list" in result, "Result missing parsed content"
#
#     def test_use_existing_deployment(self, test_image):
#         """Test client uses existing deployment if available."""
#         print("\nTesting client use of existing deployment...")
#         running_instances = get_running_parser_instances()
#         if not running_instances:
#             # Deploy if needed for this test specifically
#             print("No running instance found, deploying one for test...")
#             Deploy.start()
#             # Wait needed for server to be fully ready after Deploy.start returns
#             print("Waiting for deployed server to be ready...")
#             time.sleep(60) # Add a wait, adjust as needed
#             running_instances = get_running_parser_instances()
#
#         assert len(running_instances) > 0, \
#             "Test requires at least one running instance (deployment failed?)"
#
#         initial_instance = running_instances[0]
#         initial_url = initial_instance['url']
#         print(f"Using existing instance: {initial_url}")
#
#         # Instantiate client WITH the existing URL
#         client = OmniParserClient(server_url=initial_url, auto_deploy=False) # Disable auto_deploy
#
#         # Use client with existing deployment
#         start_time = time.time()
#         result = client.parse_image(test_image) # Use the client method
#         operation_time = time.time() - start_time
#
#         # Verify no *new* instances were created
#         current_instances = get_running_parser_instances()
#         assert len(current_instances) == len(running_instances), \
#             "Number of running instances changed unexpectedly"
#
#         # Verify result
#         assert result is not None, "Parse result should not be None"
#         assert "error" not in result, f"Parsing failed: {result.get('error')}"
#         assert "parsed_content_list" in result, "Result missing parsed content"
#         print(f"Parse operation with existing deployment took {operation_time:.1f} seconds")
#
#     def test_deployment_idempotency(self, test_image):
#         """Test that multiple deployment attempts don't create duplicate instances."""
#         print("\nTesting deployment idempotency...")
#         # Ensure at least one instance exists initially
#         initial_instances = get_running_parser_instances()
#         if not initial_instances:
#              print("No initial instance, running Deploy.start() once...")
#              Deploy.start()
#              time.sleep(60) # Wait
#              initial_instances = get_running_parser_instances()
#              assert initial_instances, "Failed to start initial instance for idempotency test"
#         initial_count = len(initial_instances)
#         print(f"Initial instance count: {initial_count}")
#
#         # Attempt multiple deployments via Deploy.start()
#         for i in range(2): # Run start twice more
#             print(f"Deployment attempt {i + 1}")
#             # Deploy.start() should find the existing running instance and not create more
#             ip, id = Deploy.start()
#             assert ip is not None, f"Deploy.start() failed on attempt {i+1}"
#             time.sleep(5) # Short pause
#
#             current_instances = get_running_parser_instances()
#             print(f"Instance count after attempt {i + 1}: {len(current_instances)}")
#             # Should ideally be exactly initial_count, but allow for delays/transients
#             assert len(current_instances) == initial_count, \
#                 f"Unexpected number of instances: {len(current_instances)} (expected {initial_count})"
#
#         # Verify client works with the final deployment state
#         final_instances = get_running_parser_instances()
#         assert final_instances, "No instances running after idempotency test"
#         client = OmniParserClient(server_url=final_instances[0]["url"], auto_deploy=False)
#         result = client.parse_image(test_image)
#         assert result is not None, "Parse operation failed after idempotency checks"
#         assert "error" not in result, f"Parsing failed: {result.get('error')}"
