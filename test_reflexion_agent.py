import unittest
from unittest.mock import patch, MagicMock
import os

# Set dummy environment variables before importing ReflexionAgent components
# as the agent file might try to load them at import time.
os.environ["OPENAI_API_KEY"] = "test_openai_key"
os.environ["TAVILY_API_KEY"] = "test_tavily_key"

# From the agent file (assuming ReflexionAgent.py is in the same directory or accessible)
from ReflexionAgent import ReflexionState, plan_step, execute_step, reflect_step, llm, executor_agent, planner

class TestReflexionAgentComponents(unittest.TestCase):

    def_state_kwargs = {
        "plan": [], 
        "executed_steps": [], 
        "reflections": [], 
        "response": "", 
        "num_revisions": 0
    }

    @patch.object(planner, 'invoke')
    def test_plan_step_returns_plan(self, mock_planner_invoke):
        # Arrange
        mock_planner_invoke.return_value = MagicMock(steps=["Mocked Step 1", "Mocked Step 2"])
        sample_state = ReflexionState(input="Test objective for planner", **self.def_state_kwargs)

        # Act
        result = plan_step(sample_state)

        # Assert
        self.assertIn("plan", result)
        self.assertIsInstance(result["plan"], list)
        self.assertEqual(result["plan"], ["Mocked Step 1", "Mocked Step 2"])
        mock_planner_invoke.assert_called_once_with({"objective": "Test objective for planner"})

    @patch.object(executor_agent, 'invoke')
    def test_execute_step_returns_observation(self, mock_executor_agent_invoke):
        # Arrange
        mock_executor_agent_invoke.return_value = {"output": "Mocked observation from executor"}
        sample_state = ReflexionState(
            input="Test objective for executor",
            plan=["Test task 1", "Test task 2"],
            executed_steps=[],
            reflections=[],
            response="",
            num_revisions=0
        )

        # Act
        result = execute_step(sample_state)

        # Assert
        self.assertIn("executed_steps", result)
        self.assertIsInstance(result["executed_steps"], list)
        self.assertEqual(len(result["executed_steps"]), 1)
        self.assertEqual(result["executed_steps"][0], ("Test task 1", "Mocked observation from executor"))
        
        self.assertIn("plan", result)
        self.assertEqual(result["plan"], ["Test task 2"]) # Remaining plan
        
        expected_agent_input = {
            "input": "Test task 1",
            "previous_steps": "No steps executed yet.",
            "reflection": "No reflection on previous step."
        }
        mock_executor_agent_invoke.assert_called_once_with(expected_agent_input)

    @patch.object(llm, 'invoke')
    def test_reflect_step_returns_reflection(self, mock_llm_invoke):
        # Arrange
        mock_llm_invoke.return_value = MagicMock(content="Mocked reflection text PROCEED:")
        sample_state = ReflexionState(
            input="Test objective for reflector",
            plan=["Remaining task"],
            executed_steps=[("Performed Action 1", "Observed Result 1")],
            reflections=[],
            response="",
            num_revisions=0
        )

        # Act
        result = reflect_step(sample_state)

        # Assert
        self.assertIn("reflections", result)
        self.assertIsInstance(result["reflections"], list)
        self.assertEqual(len(result["reflections"]), 1)
        self.assertEqual(result["reflections"][0], "Mocked reflection text PROCEED:")
        
        # Check that llm.invoke was called with the correctly formatted prompt
        args, _ = mock_llm_invoke.call_args
        prompt_arg = args[0] # The prompt text is the first positional argument
        self.assertIn("Objective: Test objective for reflector", prompt_arg)
        self.assertIn("Current Plan: Remaining task", prompt_arg)
        self.assertIn("Last Executed Action: Performed Action 1", prompt_arg)
        self.assertIn("Observation from Last Action: Observed Result 1", prompt_arg)
        mock_llm_invoke.assert_called_once()

    @patch.object(llm, 'invoke')
    def test_reflect_step_handles_no_executed_steps(self, mock_llm_invoke):
        # Arrange
        # This case should ideally not be hit if graph calls reflect_step only after execute_step,
        # but the function has a guard for it.
        sample_state = ReflexionState(
            input="Test objective for reflector, no steps",
            plan=["Some task"],
            executed_steps=[], # No executed steps
            reflections=[],
            response="",
            num_revisions=0
        )

        # Act
        result = reflect_step(sample_state)

        # Assert
        self.assertIn("reflections", result)
        self.assertEqual(result["reflections"], ["No action taken yet to reflect upon."])
        mock_llm_invoke.assert_not_called() # LLM should not be called if there are no steps to reflect on

if __name__ == '__main__':
    unittest.main()
