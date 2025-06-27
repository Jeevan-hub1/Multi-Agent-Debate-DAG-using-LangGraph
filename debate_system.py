#!/usr/bin/env python3
"""
Multi-Agent Debate DAG using LangGraph
A structured debate system with two AI agents, memory management, and automated judging.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict
from transformers import pipeline 

class DebateWorkflowState(TypedDict):
    topic: str
    debate_state: 'DebateState'


# Configuration and Data Classes
class AgentType(Enum):
    SCIENTIST = "scientist"
    PHILOSOPHER = "philosopher"


@dataclass
class DebateArgument:
    round_num: int
    agent: AgentType
    argument: str
    timestamp: str


@dataclass
class DebateState:
    topic: str
    current_round: int
    current_agent: AgentType
    arguments: List[DebateArgument]
    memory_summary: str
    is_complete: bool
    winner: Optional[str] = None
    judge_reasoning: Optional[str] = None


class DebateLogger:
    """Handles all logging for the debate system"""
    
    def __init__(self, log_file: str = "debate_log.txt"):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_state_transition(self, from_node: str, to_node: str, state: 'DebateState'):
        """Log state transitions between nodes"""
        self.logger.info(f"TRANSITION: {from_node} -> {to_node}")
        if state is not None:
            self.logger.info(f"STATE: Round {state.current_round}, Agent: {state.current_agent.value}")
        else:
            self.logger.info("STATE: (not initialized)")
        
    def log_argument(self, argument: DebateArgument):
        """Log individual arguments"""
        self.logger.info(f"[Round {argument.round_num}] {argument.agent.value.title()}: {argument.argument}")
        
    def log_memory_update(self, memory: str):
        """Log memory updates"""
        self.logger.info(f"MEMORY UPDATE: {memory[:100]}...")
        
    def log_final_verdict(self, winner: str, reasoning: str):
        """Log the final judgment"""
        self.logger.info(f"FINAL VERDICT - Winner: {winner}")
        self.logger.info(f"REASONING: {reasoning}")


class DebateAgents:
    """Container for debate agent configurations and prompts"""
    
    SCIENTIST_PROMPT = """
    You are a renowned scientist participating in a structured debate. Your approach is:
    - Evidence-based reasoning
    - Empirical data and research citations
    - Risk assessment and safety considerations
    - Practical implications for society
    
    Topic: {topic}
    Current round: {round_num}/4
    
    Previous arguments summary: {memory}
    
    Provide a clear, logical argument from a scientific perspective. Be concise but compelling.
    Avoid repeating previous points. Build upon the debate progression.
    """
    
    PHILOSOPHER_PROMPT = """
    You are a distinguished philosopher participating in a structured debate. Your approach is:
    - Ethical and moral reasoning
    - Conceptual analysis and definitions
    - Historical context and precedent
    - Human autonomy and freedom considerations
    
    Topic: {topic}
    Current round: {round_num}/4
    
    Previous arguments summary: {memory}
    
    Provide a clear, logical argument from a philosophical perspective. Be concise but compelling.
    Avoid repeating previous points. Build upon the debate progression.
    """
    
    JUDGE_PROMPT = """
    You are an impartial judge evaluating a structured debate between a Scientist and a Philosopher.
    
    Topic: {topic}
    
    Full debate transcript:
    {full_transcript}
    
    Evaluate the debate based on:
    1. Logical coherence and consistency
    2. Strength of evidence and reasoning
    3. Address of counterarguments
    4. Overall persuasiveness
    5. Factual accuracy
    
    Provide:
    1. A comprehensive summary of the debate
    2. Declare a winner (Scientist or Philosopher)
    3. Clear reasoning for your decision
    
    Format your response as:
    SUMMARY: [Your summary here]
    WINNER: [Scientist or Philosopher]
    REASONING: [Your detailed reasoning]
    """


class DebateDAG:
    """Main DAG implementation for the debate system"""
    
    def __init__(self, api_key: str = None):
        self.logger = DebateLogger()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=api_key or os.getenv("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")
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph DAG structure"""
        workflow = StateGraph(DebateWorkflowState)
        
        # Add nodes
        workflow.add_node("user_input", self.user_input_node)
        workflow.add_node("agent_a", self.agent_a_node)  # Scientist
        workflow.add_node("agent_b", self.agent_b_node)  # Philosopher
        workflow.add_node("memory", self.memory_node)
        workflow.add_node("judge", self.judge_node)
        workflow.add_node("validation", self.validation_node)
        
        # Define edges
        workflow.add_edge("user_input", "agent_a")  # Start with Scientist
        
        # Conditional edges for alternating agents
        workflow.add_conditional_edges(
            "agent_a",
            self.route_after_agent,
            {
                "memory": "memory",
                "judge": "judge"
            }
        )
        
        workflow.add_conditional_edges(
            "agent_b", 
            self.route_after_agent,
            {
                "memory": "memory",
                "judge": "judge"
            }
        )
        
        workflow.add_conditional_edges(
            "memory",
            self.route_after_memory,
            {
                "agent_a": "agent_a",
                "agent_b": "agent_b",
                "validation": "validation"
            }
        )
        
        workflow.add_edge("validation", "judge")
        workflow.add_edge("judge", END)
        
        # Set entry point
        workflow.set_entry_point("user_input")
        
        return workflow.compile()
    
    def user_input_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user input for debate topic"""
        self.logger.log_state_transition("START", "user_input", None)
        
        if "topic" not in state:
            topic = input("Enter topic for debate: ").strip()
            state["topic"] = topic
        
        # Initialize debate state
        debate_state = DebateState(
            topic=state["topic"],
            current_round=1,
            current_agent=AgentType.SCIENTIST,
            arguments=[],
            memory_summary="",
            is_complete=False
        )
        
        state["debate_state"] = debate_state
        print(f"\nStarting debate between Scientist and Philosopher...")
        print(f"Topic: {state['topic']}\n")
        
        return state
    
    def agent_a_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Scientist agent node"""
        debate_state: DebateState = state["debate_state"]
        self.logger.log_state_transition("previous", "agent_a", debate_state)
        
        if debate_state.current_agent != AgentType.SCIENTIST:
            return state  # Skip if not scientist's turn
        
        # Generate argument
        prompt = DebateAgents.SCIENTIST_PROMPT.format(
            topic=debate_state.topic,
            round_num=debate_state.current_round,
            memory=debate_state.memory_summary or "No previous arguments"
        )
        
        response = self.llm.invoke([SystemMessage(content=prompt)])
        argument_text = response.content.strip()
        
        # Create argument record
        argument = DebateArgument(
            round_num=debate_state.current_round,
            agent=AgentType.SCIENTIST,
            argument=argument_text,
            timestamp=datetime.now().isoformat()
        )
        
        # Update state
        debate_state.arguments.append(argument)
        debate_state.current_agent = AgentType.PHILOSOPHER
        
        # Log and display
        self.logger.log_argument(argument)
        print(f"[Round {argument.round_num}] Scientist: {argument_text}\n")
        
        state["debate_state"] = debate_state
        return state
    
    def agent_b_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Philosopher agent node"""
        debate_state: DebateState = state["debate_state"]
        self.logger.log_state_transition("previous", "agent_b", debate_state)
        
        if debate_state.current_agent != AgentType.PHILOSOPHER:
            return state  # Skip if not philosopher's turn
        
        # Generate argument
        prompt = DebateAgents.PHILOSOPHER_PROMPT.format(
            topic=debate_state.topic,
            round_num=debate_state.current_round,
            memory=debate_state.memory_summary or "No previous arguments"
        )
        
        response = self.llm.invoke([SystemMessage(content=prompt)])
        argument_text = response.content.strip()
        
        # Create argument record
        argument = DebateArgument(
            round_num=debate_state.current_round,
            agent=AgentType.PHILOSOPHER,
            argument=argument_text,
            timestamp=datetime.now().isoformat()
        )
        
        # Update state
        debate_state.arguments.append(argument)
        debate_state.current_agent = AgentType.SCIENTIST
        debate_state.current_round += 1
        
        # Log and display
        self.logger.log_argument(argument)
        print(f"[Round {argument.round_num}] Philosopher: {argument_text}\n")
        
        state["debate_state"] = debate_state
        return state
    
    def memory_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update memory with structured summary"""
        debate_state: DebateState = state["debate_state"]
        self.logger.log_state_transition("previous", "memory", debate_state)
        
        # Create summary of recent arguments
        recent_args = debate_state.arguments[-2:] if len(debate_state.arguments) >= 2 else debate_state.arguments
        
        summary_parts = []
        for arg in recent_args:
            summary_parts.append(f"{arg.agent.value.title()} (R{arg.round_num}): {arg.argument[:100]}...")
        
        new_summary = " | ".join(summary_parts)
        
        # Update memory (keep it manageable)
        if debate_state.memory_summary:
            debate_state.memory_summary = f"{debate_state.memory_summary} | {new_summary}"
        else:
            debate_state.memory_summary = new_summary
        
        # Truncate if too long
        if len(debate_state.memory_summary) > 1000:
            debate_state.memory_summary = debate_state.memory_summary[-800:]
        
        self.logger.log_memory_update(debate_state.memory_summary)
        
        state["debate_state"] = debate_state
        return state
    
    def validation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate debate completion and coherence"""
        debate_state: DebateState = state["debate_state"]
        self.logger.log_state_transition("memory", "validation", debate_state)
        
        # Check if debate is complete (8 rounds total, 4 per agent)
        total_rounds = len(debate_state.arguments)
        if total_rounds >= 8:
            debate_state.is_complete = True
            self.logger.logger.info("Debate validation: COMPLETE - 8 rounds reached")
        else:
            self.logger.logger.info(f"Debate validation: ONGOING - {total_rounds}/8 rounds")
        
        state["debate_state"] = debate_state
        return state
    
    def judge_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Judge the debate and declare winner"""
        debate_state: DebateState = state["debate_state"]
        self.logger.log_state_transition("validation", "judge", debate_state)
        
        # Create full transcript
        transcript_parts = []
        for arg in debate_state.arguments:
            transcript_parts.append(f"[Round {arg.round_num}] {arg.agent.value.title()}: {arg.argument}")
        
        full_transcript = "\n".join(transcript_parts)
        
        # Generate judgment
        judge_prompt = DebateAgents.JUDGE_PROMPT.format(
            topic=debate_state.topic,
            full_transcript=full_transcript
        )
        
        response = self.llm.invoke([SystemMessage(content=judge_prompt)])
        judgment = response.content.strip()
        
        # Parse judgment
        lines = judgment.split('\n')
        summary = ""
        winner = ""
        reasoning = ""
        
        current_section = ""
        for line in lines:
            if line.startswith("SUMMARY:"):
                current_section = "summary"
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("WINNER:"):
                current_section = "winner"
                winner = line.replace("WINNER:", "").strip()
            elif line.startswith("REASONING:"):
                current_section = "reasoning" 
                reasoning = line.replace("REASONING:", "").strip()
            elif current_section == "summary":
                summary += " " + line
            elif current_section == "reasoning":
                reasoning += " " + line
        
        # Update state
        debate_state.winner = winner
        debate_state.judge_reasoning = reasoning
        
        # Display results
        print("[Judge] Summary of debate:")
        print(summary)
        print(f"\n[Judge] Winner: {winner}")
        print(f"Reason: {reasoning}")
        
        # Log final verdict
        self.logger.log_final_verdict(winner, reasoning)
        
        state["debate_state"] = debate_state
        return state
    
    def route_after_agent(self, state: Dict[str, Any]) -> str:
        """Route after agent speaks"""
        debate_state: DebateState = state["debate_state"]
        
        # Always go to memory first, then check if complete
        if len(debate_state.arguments) >= 8:
            return "judge"
        return "memory"
    
    def route_after_memory(self, state: Dict[str, Any]) -> str:
        """Route after memory update"""
        debate_state: DebateState = state["debate_state"]
        
        # Check if we need to end the debate
        if len(debate_state.arguments) >= 8:
            return "validation"
        
        # Route to next agent
        if debate_state.current_agent == AgentType.SCIENTIST:
            return "agent_a"
        else:
            return "agent_b"
    
    def run_debate(self, topic: str = None) -> DebateState:
        """Run the complete debate"""
        initial_state = {}
        if topic:
            initial_state["topic"] = topic
        
        try:
            result = self.graph.invoke(initial_state)
            return result["debate_state"]
        except Exception as e:
            self.logger.logger.error(f"Error running debate: {e}")
            raise


def main():
    """Main CLI interface"""
    print("Multi-Agent Debate System")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it or the system will prompt for it.")
    
    try:
        # Initialize debate system
        debate_system = DebateDAG()
        
        # Run debate
        final_state = debate_system.run_debate()
        
        print("\n" + "=" * 50)
        print("Debate completed successfully!")
        print(f"Check 'debate_log.txt' for full transcript.")
        
        # Save final state to JSON
        with open("debate_results.json", "w") as f:
            json.dump(asdict(final_state), f, indent=2, default=str)
        
        print("Results saved to 'debate_results.json'")
        
    except KeyboardInterrupt:
        print("\nDebate interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())