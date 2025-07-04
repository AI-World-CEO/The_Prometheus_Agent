# prometheus_agent/tests/unit/test_personal_gui.py

import unittest
import sys
from unittest.mock import MagicMock, patch

# A-1: We must import Qt Test utilities for simulating user input.
# We also need a QApplication instance for any widget test to run.
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from prometheus_agent.PersonalGUI import PersonalGUI

# This structure assumes the test is run from the project root directory


# This is necessary to run a Qt Application in a test environment
app = QApplication(sys.argv)


class TestPersonalGUI(unittest.TestCase):
    """
    A comprehensive unit test suite for the PersonalGUI component.

    This suite validates the GUI's ability to initialize, respond to user input,
    and correctly display data received from the agent's signals, ensuring the
    symbiotic interface is wired correctly and behaves as expected.
    """

    def setUp(self):
        """
        Set up a mocked agent and event loop, and create an instance of the GUI
        for each test.
        """
        # --- Mock Dependencies ---
        self.mock_agent = MagicMock()
        self.mock_agent.version = "Test.v1"
        self.mock_loop = MagicMock()

        # --- Instantiate the GUI ---
        # The GUI is the "class under test"
        self.gui = PersonalGUI(agent=self.mock_agent, loop=self.mock_loop)

    def tearDown(self):
        """Clean up the GUI instance after each test."""
        self.gui.close()
        self.gui = None

    def test_01_initialization_and_widget_creation(self):
        """
        Test (Golden Path): Verifies that the GUI initializes without errors
        and all key widgets are created.
        """
        # --- Assert ---
        self.assertEqual(self.gui.windowTitle(), "Prometheus Agent vTest.v1")
        self.assertIsNotNone(self.gui.output_area)
        self.assertIsNotNone(self.gui.plan_table)
        self.assertIsNotNone(self.gui.input_field)
        self.assertIsNotNone(self.gui.status_label)
        self.assertEqual(self.gui.status_label.text(), "Status: Initializing...")

    def test_02_display_agent_response(self):
        """
        Test (Signal Handling): Verifies that emitting the `response_received`
        signal correctly updates the output area and plan table.
        """
        # --- Arrange ---
        mock_response = {
            "response": "This is the final answer.",
            "confidence_score": 0.95,
            "persona_archetype": "The Analyst",
            "cognitive_plan": {
                "thought_process": "A brilliant thought process.",
                "plan": [
                    {"step_number": 1, "tool_name": "tool_A", "objective": "Objective A"},
                    {"step_number": 2, "tool_name": "tool_B", "objective": "Objective B"}
                ]
            }
        }

        # --- Act ---
        # Directly call the slot method to simulate the signal being received
        self.gui._display_agent_response(mock_response)

        # --- Assert ---
        # 1. Check the rich text output area
        output_html = self.gui.output_area.toHtml()
        self.assertIn("Persona: The Analyst", output_html)
        self.assertIn("Confidence: 0.95", output_html)
        self.assertIn("This is the final answer.", output_html)
        self.assertIn("A brilliant thought process.", output_html)

        # 2. Check the plan table
        self.assertEqual(self.gui.plan_table.rowCount(), 2)
        self.assertEqual(self.gui.plan_table.item(0, 1).text(), "tool_A")
        self.assertEqual(self.gui.plan_table.item(1, 2).text(), "Objective B")

    def test_03_display_error_message(self):
        """
        Test (Signal Handling): Verifies that emitting the `error_occurred`
        signal correctly displays an error message.
        """
        # --- Arrange ---
        error_message = "A critical failure has occurred."

        # --- Act ---
        self.gui._display_error(error_message)

        # --- Assert ---
        error_html = self.gui.output_area.toHtml()
        self.assertIn("Agent Error", error_html)
        self.assertIn(error_message, error_html)
        # The plan table should be cleared on error
        self.assertEqual(self.gui.plan_table.rowCount(), 0)

    @patch('asyncio.run_coroutine_threadsafe')
    def test_04_handle_user_input_triggers_agent_query(self, mock_run_coro):
        """
        Test (User Interaction): Simulates a user typing and pressing Enter,
        and verifies that it triggers the async agent query.
        """
        # --- Arrange ---
        prompt = "What is the meaning of life?"

        # --- Act ---
        # Use QTest to simulate user typing into the QLineEdit
        QTest.keyClicks(self.gui.input_field, prompt)
        # Simulate pressing the Enter key
        QTest.keyClick(self.gui.input_field, Qt.Key_Return)

        # --- Assert ---
        # 1. Verify the thinking message is displayed
        self.assertIn("Agent is thinking...", self.gui.output_area.toHtml())

        # 2. Verify the input field was cleared
        self.assertEqual(self.gui.input_field.text(), "")

        # 3. Verify that the async task was scheduled on the event loop
        mock_run_coro.assert_called_once()
        # Check that it was called with the correct event loop
        self.assertEqual(mock_run_coro.call_args[0][1], self.mock_loop)

        # The coroutine object itself is the first argument
        coro = mock_run_coro.call_args[0][0]
        # We can't easily inspect the coroutine, but we can check the agent method it calls
        self.assertIn("_process_query", str(coro))

    def test_05_handle_empty_user_input(self):
        """
        Test (Input Validation): Ensures that submitting an empty prompt
        displays an error and does not trigger an agent query.
        """
        # --- Act ---
        QTest.keyClick(self.gui.input_field, Qt.Key_Return)

        # --- Assert ---
        self.assertIn("Input prompt cannot be empty", self.gui.output_area.toHtml())
        # Ensure the agent was not called
        self.mock_agent.reflexive_thought.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
