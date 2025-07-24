"""
Unit-tests for OpenAIServingChat — rewritten to use only the std-lib 'unittest'.
Run with either:
    python tests/test_serving_chat_unit.py -v
or
    python -m unittest discover -s tests -p "test_*unit.py" -v
"""

import unittest
import uuid
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingChat."""

    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
        )
        self.chat_template_name: Optional[str] = "llama-3"

        # tokenizer stub
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # async generator stub for generate_request
        async def _mock_generate():
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.1, 1, "Test"), (0.2, 2, "response")],
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None


class ServingChatTestCase(unittest.TestCase):
    # ------------- common fixtures -------------
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)

        # frequently reused requests
        self.basic_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )
        self.stream_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    # ------------- conversion tests -------------
    def test_convert_to_internal_request_single(self):
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock, patch.object(self.chat, "_process_messages") as proc_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = ["</s>"]
            conv_mock.return_value = conv_ins

            proc_mock.return_value = MessageProcessingResult(
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted, processed = self.chat._convert_to_internal_request(self.basic_req)
            self.assertIsInstance(adapted, GenerateReqInput)
            self.assertFalse(adapted.stream)
            self.assertEqual(processed, self.basic_req)

    def test_stop_str_isolation_between_requests(self):
        """Test that stop strings from one request don't affect subsequent requests.

        This tests the fix for the bug where conv.stop_str was being mutated globally,
        causing stop strings from one request to persist in subsequent requests.
        """
        # Mock conversation template with initial stop_str
        initial_stop_str = ["\n"]

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            # Create a mock conversation object that will be returned by generate_chat_conv
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = None
            conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = (
                initial_stop_str.copy()
            )  # Template's default stop strings
            conv_mock.return_value = conv_ins

            # First request with additional stop string
            req1 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "First request"}],
                stop=["CUSTOM_STOP"],
            )

            # Call the actual _apply_conversation_template method (not mocked)
            result1 = self.chat._apply_conversation_template(req1, is_multimodal=False)

            # Verify first request has both stop strings
            expected_stop1 = initial_stop_str + ["CUSTOM_STOP"]
            self.assertEqual(result1.stop, expected_stop1)

            # Verify the original template's stop_str wasn't mutated after first request
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

            # Second request without additional stop string
            req2 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "Second request"}],
                # No custom stop strings
            )
            result2 = self.chat._apply_conversation_template(req2, is_multimodal=False)

            # Verify second request only has original stop strings (no CUSTOM_STOP from req1)
            self.assertEqual(result2.stop, initial_stop_str)
            self.assertNotIn("CUSTOM_STOP", result2.stop)
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

    # ------------- sampling-params -------------
    def test_sampling_param_build(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.8,
            max_tokens=150,
            min_tokens=5,
            top_p=0.9,
            stop=["</s>"],
        )
        with patch.object(
            self.chat,
            "_process_messages",
            return_value=("Prompt", [1], None, None, [], ["</s>"], None),
        ):
            params = self.chat._build_sampling_params(req, ["</s>"], None)
            self.assertEqual(params["temperature"], 0.8)
            self.assertEqual(params["max_new_tokens"], 150)
            self.assertEqual(params["min_new_tokens"], 5)
            self.assertEqual(params["stop"], ["</s>"])


class APIKeyAuthenticationTestCase(unittest.TestCase):
    """Tests for API key authentication middleware functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)
        self.api_key = "test-api-key-12345"

    # ------------- Authentication Tests -------------
    def test_api_key_authentication_success(self):
        """Test successful authentication with valid API key."""
        from sglang.srt.utils import AuthenticationMiddleware
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route
        import asyncio

        async def endpoint(request):
            return PlainTextResponse("Success")

        app = Starlette(routes=[Route("/test", endpoint)])
        app.add_middleware(AuthenticationMiddleware, api_token=self.api_key)

        # Test with valid Bearer token
        async def test_valid_token():
            from starlette.testclient import TestClient
            with TestClient(app) as client:
                response = client.get(
                    "/test",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.text, "Success")

        asyncio.run(test_valid_token())

    def test_api_key_authentication_failure(self):
        """Test authentication failure without API key."""
        from sglang.srt.utils import AuthenticationMiddleware
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route
        import asyncio

        async def endpoint(request):
            return PlainTextResponse("Success")

        app = Starlette(routes=[Route("/test", endpoint)])
        app.add_middleware(AuthenticationMiddleware, api_token=self.api_key)

        # Test without token
        async def test_no_token():
            from starlette.testclient import TestClient
            with TestClient(app) as client:
                response = client.get("/test")
                self.assertEqual(response.status_code, 401)
                self.assertIn("Unauthorized", response.json()["detail"])

        # Test with invalid token
        async def test_invalid_token():
            from starlette.testclient import TestClient
            with TestClient(app) as client:
                response = client.get(
                    "/test",
                    headers={"Authorization": "Bearer wrong-token"}
                )
                self.assertEqual(response.status_code, 401)
                self.assertIn("Unauthorized", response.json()["detail"])

        asyncio.run(test_no_token())
        asyncio.run(test_invalid_token())

    def test_skip_conditions(self):
        """Test that OPTIONS requests and health/metrics endpoints bypass auth."""
        from sglang.srt.utils import AuthenticationMiddleware
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route
        import asyncio

        async def endpoint(request):
            return PlainTextResponse("Success")

        app = Starlette(routes=[
            Route("/test", endpoint, methods=["GET", "OPTIONS"]),
            Route("/health", endpoint),
            Route("/v1/health", endpoint),
            Route("/metrics", endpoint),
            Route("/v1/metrics", endpoint),
        ])
        app.add_middleware(AuthenticationMiddleware, api_token=self.api_key)

        async def test_skip_auth():
            from starlette.testclient import TestClient
            with TestClient(app) as client:
                # OPTIONS request should bypass auth
                response = client.options("/test")
                self.assertEqual(response.status_code, 200)

                # Health endpoints should bypass auth
                response = client.get("/health")
                self.assertEqual(response.status_code, 200)
                
                response = client.get("/v1/health")
                self.assertEqual(response.status_code, 200)

                # Metrics endpoints should bypass auth
                response = client.get("/metrics")
                self.assertEqual(response.status_code, 200)
                
                response = client.get("/v1/metrics")
                self.assertEqual(response.status_code, 200)

        asyncio.run(test_skip_auth())

    def test_concurrent_authenticated_requests(self):
        """Test concurrent requests with API key authentication."""
        from sglang.srt.utils import AuthenticationMiddleware
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route
        import asyncio
        import time

        request_count = 0
        request_times = []

        async def endpoint(request):
            nonlocal request_count
            request_count += 1
            request_times.append(time.time())
            await asyncio.sleep(0.01)  # Simulate some processing time
            return PlainTextResponse(f"Request {request_count}")

        app = Starlette(routes=[Route("/test", endpoint)])
        app.add_middleware(AuthenticationMiddleware, api_token=self.api_key)

        async def make_request(client, request_id):
            """Make a single authenticated request."""
            response = await client.get(
                "/test",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return response

        async def test_concurrent():
            from httpx import AsyncClient
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                # Create 10 concurrent requests
                tasks = [make_request(client, i) for i in range(10)]
                responses = await asyncio.gather(*tasks)
                
                # All requests should succeed
                for response in responses:
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("Request", response.text)
                
                # Verify all requests were processed
                self.assertEqual(request_count, 10)
                
                # Verify requests were processed concurrently (not sequentially)
                # If processed sequentially, total time would be ~0.1s (10 * 0.01s)
                # If concurrent, should be much less
                total_time = max(request_times) - min(request_times)
                self.assertLess(total_time, 0.05)  # Should be much less than 0.1s

        asyncio.run(test_concurrent())

    def test_mixed_concurrent_requests(self):
        """Test concurrent requests with mixed authentication status."""
        from sglang.srt.utils import AuthenticationMiddleware
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route
        import asyncio

        async def endpoint(request):
            return PlainTextResponse("Success")

        async def health_endpoint(request):
            return PlainTextResponse("Healthy")

        app = Starlette(routes=[
            Route("/test", endpoint),
            Route("/health", health_endpoint),
        ])
        app.add_middleware(AuthenticationMiddleware, api_token=self.api_key)

        async def test_mixed():
            from httpx import AsyncClient
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                # Mix of authenticated, unauthenticated, and health check requests
                tasks = []
                
                # 5 authenticated requests (should succeed)
                for i in range(5):
                    tasks.append(client.get(
                        "/test",
                        headers={"Authorization": f"Bearer {self.api_key}"}
                    ))
                
                # 5 unauthenticated requests (should fail)
                for i in range(5):
                    tasks.append(client.get("/test"))
                
                # 5 health check requests (should succeed without auth)
                for i in range(5):
                    tasks.append(client.get("/health"))
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check authenticated requests succeeded
                for i in range(5):
                    self.assertEqual(responses[i].status_code, 200)
                    self.assertEqual(responses[i].text, "Success")
                
                # Check unauthenticated requests failed
                for i in range(5, 10):
                    self.assertEqual(responses[i].status_code, 401)
                
                # Check health requests succeeded
                for i in range(10, 15):
                    self.assertEqual(responses[i].status_code, 200)
                    self.assertEqual(responses[i].text, "Healthy")

        asyncio.run(test_mixed())


if __name__ == "__main__":
    unittest.main(verbosity=2)
