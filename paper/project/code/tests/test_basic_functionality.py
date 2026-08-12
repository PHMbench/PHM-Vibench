#!/usr/bin/env python3
"""
Basic Functionality Tests

Tests the core functionality of the LLM-Enhanced Fault Diagnosis Toolkit.
"""

import unittest
import sys
import os
import numpy as np
import torch

# Add the toolkit to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from llm_explainable_toolkit import DiagnosticSystem, create_toolkit


class TestDiagnosticSystem(unittest.TestCase):
    """Test cases for the DiagnosticSystem class."""

    def setUp(self):
        """Set up test fixtures."""
        self.system = create_toolkit()
        self.sample_signal = self.create_test_signal()

    def create_test_signal(self):
        """Create a test signal for testing."""
        fs = 1024
        duration = 1
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = np.sin(2 * np.pi * 30 * t) + 0.1 * np.random.randn(len(t))
        return signal

    def test_system_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system)
        self.assertIsNotNone(self.system.explainer)
        self.assertIsNotNone(self.system.conversation_agent)

    def test_basic_diagnosis(self):
        """Test basic diagnosis functionality."""
        result = self.system.diagnose(self.sample_signal)

        # Check result structure
        self.assertIn("timestamp", result)
        self.assertIn("signal_info", result)
        self.assertIn("model_prediction", result)
        self.assertIn("explanation", result)

        # Check prediction structure
        prediction = result["model_prediction"]
        self.assertIn("fault_type", prediction)
        self.assertIn("confidence", prediction)

        # Check explanation structure
        explanation = result["explanation"]
        self.assertIn("fault_info", explanation)
        self.assertIn("signal_analysis", explanation)

    def test_conversation_start(self):
        """Test conversation start functionality."""
        device_info = {
            "device_type": "motor",
            "operating_speed": 1800
        }

        session_result = self.system.start_conversation(
            self.sample_signal, device_info=device_info
        )

        # Check session result structure
        self.assertIn("session_id", session_result)
        self.assertIn("greeting", session_result)
        self.assertIn("initial_diagnosis", session_result)

        # Check session creation
        session_id = session_result["session_id"]
        self.assertIn(session_id, self.system.get_active_sessions())

    def test_conversation_continuation(self):
        """Test conversation continuation."""
        # Start conversation
        session_result = self.system.start_conversation(self.sample_signal)
        session_id = session_result["session_id"]

        # Continue conversation
        response = self.system.continue_conversation(
            session_id, "What is the cause of this fault?"
        )

        self.assertIn("session_id", response)
        self.assertIn("response", response)

    def test_conversation_end(self):
        """Test conversation ending."""
        # Start conversation
        session_result = self.system.start_conversation(self.sample_signal)
        session_id = session_result["session_id"]

        # End conversation
        conclusion = self.system.end_conversation(session_id)

        self.assertIn("session_id", conclusion)
        self.assertIn("conclusion", conclusion)
        self.assertIn("duration_seconds", conclusion)

    def test_batch_diagnosis(self):
        """Test batch diagnosis functionality."""
        signals = [self.create_test_signal() for _ in range(3)]

        results = self.system.batch_diagnose(signals)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn("batch_index", result)
            if "error" not in result:
                self.assertIn("model_prediction", result)

    def test_diagnostic_history(self):
        """Test diagnostic history functionality."""
        # Perform some diagnoses
        for _ in range(5):
            self.system.diagnose(self.sample_signal)

        # Get history
        history = self.system.get_diagnostic_history(limit=3)

        self.assertLessEqual(len(history), 3)
        self.assertGreater(len(history), 0)

        for item in history:
            self.assertIn("timestamp", item)
            self.assertIn("model_prediction", item)

    def test_export_functionality(self):
        """Test data export functionality."""
        # Perform some diagnoses
        self.system.diagnose(self.sample_signal)

        # Test export
        export_path = "./test_export.json"
        try:
            self.system.export_data(export_path)
            self.assertTrue(os.path.exists(export_path))
        finally:
            # Clean up
            if os.path.exists(export_path):
                os.remove(export_path)


class TestLLMEnhancedExplainer(unittest.TestCase):
    """Test cases for the LLMEnhancedExplainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.explainer = create_toolkit().explainer
        self.sample_signal = self.create_test_signal()
        self.mock_prediction = {
            "fault_type": "内圈故障",
            "confidence": 0.85,
            "probabilities": [0.1, 0.05, 0.85, 0.0, 0.0]
        }

    def create_test_signal(self):
        """Create a test signal for testing."""
        fs = 1024
        duration = 1
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = np.sin(2 * np.pi * 30 * t) + 0.1 * np.random.randn(len(t))
        return signal

    def test_explanation_generation(self):
        """Test basic explanation generation."""
        explanation = self.explainer.explain(
            self.sample_signal,
            self.mock_prediction
        )

        # Check explanation structure
        self.assertIn("timestamp", explanation)
        self.assertIn("fault_info", explanation)
        self.assertIn("signal_analysis", explanation)
        self.assertIn("recommendations", explanation)

        # Check fault info
        fault_info = explanation["fault_info"]
        self.assertEqual(fault_info["fault_type"], "内圈故障")
        self.assertEqual(fault_info["confidence"], 0.85)

    def test_different_explanation_styles(self):
        """Test different explanation styles."""
        styles = ["standard", "detailed", "simple", "expert"]

        for style in styles:
            explanation = self.explainer.explain(
                self.sample_signal,
                self.mock_prediction,
                style=style
            )

            self.assertIn("metadata", explanation)
            self.assertEqual(explanation["metadata"]["explanation_style"], style)

    def test_explanation_with_query(self):
        """Test explanation generation with user query."""
        user_query = "What is the cause of this fault?"
        explanation = self.explainer.explain(
            self.sample_signal,
            self.mock_prediction,
            user_query=user_query
        )

        self.assertIn("fault_info", explanation)
        self.assertIn("signal_analysis", explanation)

    def test_conversation_response(self):
        """Test conversation response generation."""
        context = {
            "session_info": {"session_id": "test_session"},
            "diagnostic_context": {"fault_type": "内圈故障"},
            "conversation_history": []
        }

        response = self.explainer.explain_conversation(
            "test_session",
            "What is the maintenance procedure?",
            context
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_component_info(self):
        """Test component information retrieval."""
        info = self.explainer.get_component_info()

        self.assertIn("llm_available", info)
        self.assertIn("knowledge_graph_available", info)
        self.assertIn("explanation_history_size", info)

    def test_explanation_history(self):
        """Test explanation history functionality."""
        # Generate some explanations
        for _ in range(3):
            self.explainer.explain(self.sample_signal, self.mock_prediction)

        # Get history
        history = self.explainer.get_explanation_history(limit=2)

        self.assertLessEqual(len(history), 2)
        self.assertGreater(len(history), 0)

        for item in history:
            self.assertIn("timestamp", item)
            self.assertIn("fault_type", item)


def run_basic_tests():
    """Run all basic functionality tests."""
    print("🧪 运行基础功能测试...")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDiagnosticSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMEnhancedExplainer))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ 所有测试通过！")
        print(f"运行测试: {result.testsRun}")
        print(f"执行时间: {result.duration:.2f} 秒")
    else:
        print("❌ 部分测试失败")
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")

        # Print failure details
        for test, traceback in result.failures:
            print(f"\n❌ {test}: {traceback}")

        for test, traceback in result.errors:
            print(f"\n🔥 {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)